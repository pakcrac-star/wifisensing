/**
 * @file wifi_csi.c
 * @brief Production-Grade Radio Hardware Abstraction Layer (HAL) for ESP32 Wi-Fi & CSI.
 *
 * EXCLUSIVE RESPONSIBILITIES:
 * - Hardware initialization, configuration, and teardown
 * - AP connection & auto-reconnection management
 * - CSI & Promiscuous ISR/callback frame extraction
 * - Hardware timestamping
 * - Thread-safe transport queues/ringbuffers to Rust FFI
 * - Passive channel hopping & hybrid scanning management
 * 
 * STRICT PROHIBITIONS:
 * - No DSP, filtering, or smoothing
 * - No phase/amplitude math
 * - No blocking operations in ISR or Wi-Fi callback contexts
 */

#include <string.h>
#include <stdbool.h>
#include <stdatomic.h>
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "esp_attr.h"
#include "esp_netif.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/ringbuf.h"
#include "freertos/queue.h"

#include "wifi_csi.h"

/* =========================================================================
 * Fallbacks & Guarded Definitions
 * ========================================================================= */



#ifndef CONFIG_CSI_MAX_SAMPLES
#define CONFIG_CSI_MAX_SAMPLES 384
#endif

#ifndef CONFIG_CSI_RINGBUF_DEPTH
#define CONFIG_CSI_RINGBUF_DEPTH 32
#endif

#ifndef CONFIG_METADATA_QUEUE_DEPTH
#define CONFIG_METADATA_QUEUE_DEPTH 16
#endif

#ifndef CONFIG_CSI_CHANNEL_HOP_INTERVAL_MS
#define CONFIG_CSI_CHANNEL_HOP_INTERVAL_MS 250
#endif

#ifndef CONFIG_CSI_HYBRID_SCAN_DWELL_MS
#define CONFIG_CSI_HYBRID_SCAN_DWELL_MS 120
#endif

#ifndef CONFIG_CSI_HYBRID_SCAN_INTERVAL_MS
#define CONFIG_CSI_HYBRID_SCAN_INTERVAL_MS 5000
#endif

#define MAX_SCANNED_APS 20 

static const char *TAG __attribute__((unused)) = "WIFI_CSI_HAL";

/* =========================================================================
 * Internal HAL State & Synchronization
 * ========================================================================= */

typedef struct {
    wifi_csi_mode_t current_mode;
    volatile bool is_initialized;
    volatile bool is_started;
    volatile bool is_connected;
    uint8_t current_channel;
    
    // Config storage
    app_wifi_csi_config_t config;
    
    // ESP-IDF handles
    esp_netif_t *sta_netif;
    TaskHandle_t channel_hop_task_handle;
    TaskHandle_t hybrid_scan_task_handle;
    SemaphoreHandle_t lock;

    // Event instance handlers
    esp_event_handler_instance_t wifi_event_instance;
    esp_event_handler_instance_t ip_event_instance;

    // Buffers for Rust FFI
    RingbufHandle_t csi_ringbuf;
    QueueHandle_t metadata_queue;

    // Statistics (Updated atomically)
    wifi_csi_stats_t stats;
} hal_state_t;

static hal_state_t s_hal = {0};

/* =========================================================================
 * Helper Functions
 * ========================================================================= */

static inline bool is_passive_mode(wifi_csi_mode_t mode)
{
    // Passive observer mode is represented by WIFI_CSI_MODE_UNKNOWN (0).
    // Do NOT treat WIFI_CSI_MODE_HYBRID as passive here.
    return (mode == WIFI_CSI_MODE_UNKNOWN);
}

/* =========================================================================
 * Forward Declarations 
 * ========================================================================= */

static void wifi_csi_cb(void *ctx, wifi_csi_info_t *info);
static void wifi_promiscuous_cb(void *buf, wifi_promiscuous_pkt_type_t type);
static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data);
static void ip_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data);
static void channel_hop_task(void *pvParameters);
static void hybrid_scan_task(void *pvParameters);
static esp_err_t apply_radio_config(void);

/* =========================================================================
 * High-Priority CSI Callback
 * ========================================================================= */

static void IRAM_ATTR wifi_csi_cb(void *ctx, wifi_csi_info_t *info)
{
    if (unlikely(!info || !info->buf || info->len == 0)) {
        __atomic_fetch_add(&s_hal.stats.invalid_frames, 1, __ATOMIC_RELAXED);
        return;
    }

    // Static buffer to prevent stack frame overflow in task context
    static csi_frame_t frame;
    
    frame.timestamp_us = (uint64_t)esp_timer_get_time();

    memcpy(frame.mac_address, info->mac, 6);
    frame.rssi = info->rx_ctrl.rssi;
    frame.noise_floor = info->rx_ctrl.noise_floor;
    frame.channel = info->rx_ctrl.channel;
    frame.bandwidth = info->rx_ctrl.cwb;
    frame.phy_mode = info->rx_ctrl.sig_mode;
    
    uint16_t copy_len = (info->len > CONFIG_CSI_MAX_SAMPLES) ? CONFIG_CSI_MAX_SAMPLES : info->len;
    frame.csi_length = copy_len;
    memcpy(frame.csi_data, info->buf, copy_len);

    BaseType_t higher_priority_task_woken = pdFALSE;
    BaseType_t ret = xRingbufferSendFromISR(
        s_hal.csi_ringbuf,
        &frame,
        sizeof(csi_frame_t),
        &higher_priority_task_woken
    );

    if (likely(ret == pdTRUE)) {
        __atomic_fetch_add(&s_hal.stats.csi_frames_received, 1, __ATOMIC_RELAXED);
    } else {
        __atomic_fetch_add(&s_hal.stats.dropped_frames, 1, __ATOMIC_RELAXED);
    }

    if (higher_priority_task_woken == pdTRUE) {
        portYIELD_FROM_ISR();
    }
}

/* =========================================================================
 * Promiscuous Packet Callback (Metadata Extraction)
 * ========================================================================= */

static void IRAM_ATTR wifi_promiscuous_cb(void *buf, wifi_promiscuous_pkt_type_t type)
{
    if (!is_passive_mode(s_hal.current_mode) || unlikely(!buf)) {
        return;
    }

    const wifi_promiscuous_pkt_t *pkt = (const wifi_promiscuous_pkt_t *)buf;
    const wifi_pkt_rx_ctrl_t *ctrl = &pkt->rx_ctrl;

    static metadata_frame_t meta;
    memset(&meta, 0, sizeof(metadata_frame_t));

    meta.timestamp_us = (uint64_t)esp_timer_get_time();
    meta.rssi = ctrl->rssi;
    meta.channel = ctrl->channel;
    meta.bandwidth = ctrl->cwb;
    meta.phy_mode = ctrl->sig_mode;
    meta.noise = ctrl->noise_floor;
    meta.security_type = 0; 
    meta.beacon_interval = 0;

    if (ctrl->sig_len >= 24) {
        memcpy(meta.bssid, &pkt->payload[10], 6);
    }

    BaseType_t higher_priority_task_woken = pdFALSE;
    xQueueSendFromISR(s_hal.metadata_queue, &meta, &higher_priority_task_woken);

    if (higher_priority_task_woken == pdTRUE) {
        portYIELD_FROM_ISR();
    }
}

/* =========================================================================
 * Channel Manager & Scheduler (Passive Mode)
 * ========================================================================= */

static void channel_hop_task(void *pvParameters)
{
    uint8_t channel_idx = 0;
    const uint8_t channels[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    const size_t total_channels = sizeof(channels) / sizeof(channels[0]);

    while (s_hal.is_started) {
        if (is_passive_mode(s_hal.current_mode)) {
            uint8_t next_chan = channels[channel_idx];
            if (esp_wifi_set_channel(next_chan, WIFI_SECOND_CHAN_NONE) == ESP_OK) {
                s_hal.current_channel = next_chan;
            }
            channel_idx = (channel_idx + 1) % total_channels;
        }
        vTaskDelay(pdMS_TO_TICKS(CONFIG_CSI_CHANNEL_HOP_INTERVAL_MS));
    }
    
    s_hal.channel_hop_task_handle = NULL;
    vTaskDelete(NULL);
}

/* =========================================================================
 * Hybrid Manager (Background Scanning)
 * ========================================================================= */

static void hybrid_scan_task(void *pvParameters)
{
    wifi_scan_config_t scan_config = {
        .ssid = NULL,
        .bssid = NULL,
        .channel = 0,
        .show_hidden = true,
        .scan_type = WIFI_SCAN_TYPE_PASSIVE,
        .scan_time.passive = CONFIG_CSI_HYBRID_SCAN_DWELL_MS
    };

    static wifi_ap_record_t s_ap_records[MAX_SCANNED_APS];

    while (s_hal.is_started) {
        if (s_hal.current_mode == WIFI_CSI_MODE_HYBRID && s_hal.is_connected) {
            
            s_hal.stats.scan_count++;
            esp_err_t err = esp_wifi_scan_start(&scan_config, true);
            
            if (err == ESP_OK) {
                uint16_t ap_count = MAX_SCANNED_APS;
                
                if (esp_wifi_scan_get_ap_records(&ap_count, s_ap_records) == ESP_OK) {
                    uint64_t now = (uint64_t)esp_timer_get_time();
                    
                    for (int i = 0; i < ap_count; i++) {
                        metadata_frame_t meta;
                        memset(&meta, 0, sizeof(metadata_frame_t));
                        
                        meta.timestamp_us = now;
                        memcpy(meta.bssid, s_ap_records[i].bssid, 6);
                        strncpy(meta.ssid, (char *)s_ap_records[i].ssid, 32);
                        meta.ssid[32] = '\0';
                        meta.rssi = s_ap_records[i].rssi;
                        meta.channel = s_ap_records[i].primary;
                        meta.security_type = s_ap_records[i].authmode;
                        meta.phy_mode = s_ap_records[i].phy_11b | (s_ap_records[i].phy_11g << 1) | (s_ap_records[i].phy_11n << 2);
                        meta.bandwidth = s_ap_records[i].bandwidth;
                        meta.noise = 0;
                        meta.beacon_interval = 0;

                        xQueueSend(s_hal.metadata_queue, &meta, 0);
                    }
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(CONFIG_CSI_HYBRID_SCAN_INTERVAL_MS));
    }

    s_hal.hybrid_scan_task_handle = NULL;
    vTaskDelete(NULL);
}

/* =========================================================================
 * Connection Manager & Event Handlers
 * ========================================================================= */

static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
            case WIFI_EVENT_STA_START:
                if (s_hal.is_started && !is_passive_mode(s_hal.current_mode)) {
                    esp_wifi_connect();
                }
                break;
            case WIFI_EVENT_STA_DISCONNECTED:
                s_hal.is_connected = false;
                __atomic_fetch_add(&s_hal.stats.disconnect_count, 1, __ATOMIC_RELAXED);
                if (s_hal.is_started && !is_passive_mode(s_hal.current_mode)) {
                    __atomic_fetch_add(&s_hal.stats.reconnect_count, 1, __ATOMIC_RELAXED);
                    esp_wifi_connect();
                }
                break;
            default:
                break;
        }
    }
}

static void ip_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        s_hal.is_connected = true;
    }
}

/* =========================================================================
 * HAL Lifecycle & Configuration
 * ========================================================================= */

esp_err_t wifi_csi_hal_init(const app_wifi_csi_config_t *config)
{
    if (!config) return ESP_ERR_INVALID_ARG;

    if (s_hal.lock == NULL) {
        s_hal.lock = xSemaphoreCreateMutex();
        if (!s_hal.lock) return ESP_ERR_NO_MEM;
    }

    if (xSemaphoreTake(s_hal.lock, pdMS_TO_TICKS(1000)) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }

    if (s_hal.is_initialized) {
        xSemaphoreGive(s_hal.lock);
        return ESP_ERR_INVALID_STATE;
    }

    memcpy(&s_hal.config, config, sizeof(app_wifi_csi_config_t));
    s_hal.current_mode = config->mode;

    s_hal.csi_ringbuf = xRingbufferCreate(sizeof(csi_frame_t) * CONFIG_CSI_RINGBUF_DEPTH, RINGBUF_TYPE_NOSPLIT);
    s_hal.metadata_queue = xQueueCreate(CONFIG_METADATA_QUEUE_DEPTH, sizeof(metadata_frame_t));

    if (!s_hal.csi_ringbuf || !s_hal.metadata_queue) {
        if (s_hal.csi_ringbuf) vRingbufferDelete(s_hal.csi_ringbuf);
        if (s_hal.metadata_queue) vQueueDelete(s_hal.metadata_queue);
        s_hal.csi_ringbuf = NULL;
        s_hal.metadata_queue = NULL;
        xSemaphoreGive(s_hal.lock);
        return ESP_ERR_NO_MEM;
    }

    s_hal.sta_netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    if (s_hal.sta_netif == NULL) {
        s_hal.sta_netif = esp_netif_create_default_wifi_sta();
    }

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_err_t ret = esp_wifi_init(&cfg);
    if (ret != ESP_OK) {
        vRingbufferDelete(s_hal.csi_ringbuf);
        vQueueDelete(s_hal.metadata_queue);
        s_hal.csi_ringbuf = NULL;
        s_hal.metadata_queue = NULL;
        xSemaphoreGive(s_hal.lock);
        return ret;
    }

    ret = esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &s_hal.wifi_event_instance);
    if (ret != ESP_OK) goto cleanup;

    ret = esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &ip_event_handler, NULL, &s_hal.ip_event_instance);
    if (ret != ESP_OK) goto cleanup;

    ret = esp_wifi_set_storage(WIFI_STORAGE_RAM);
    if (ret != ESP_OK) goto cleanup;
    
    s_hal.is_initialized = true;
    xSemaphoreGive(s_hal.lock);
    return ESP_OK;

cleanup:
    if (s_hal.wifi_event_instance) {
        esp_event_handler_instance_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID, s_hal.wifi_event_instance);
        s_hal.wifi_event_instance = NULL;
    }
    esp_wifi_deinit();
    vRingbufferDelete(s_hal.csi_ringbuf);
    vQueueDelete(s_hal.metadata_queue);
    s_hal.csi_ringbuf = NULL;
    s_hal.metadata_queue = NULL;
    xSemaphoreGive(s_hal.lock);
    return ret;
}

static esp_err_t apply_radio_config(void)
{
    esp_err_t ret = esp_wifi_set_mode(WIFI_MODE_STA);
    if (ret != ESP_OK) return ret;
    
    if (!is_passive_mode(s_hal.current_mode)) {
        wifi_config_t sta_config = {0};
        strncpy((char *)sta_config.sta.ssid, s_hal.config.ssid, sizeof(sta_config.sta.ssid) - 1);
        strncpy((char *)sta_config.sta.password, s_hal.config.password, sizeof(sta_config.sta.password) - 1);
        
        ret = esp_wifi_set_config(WIFI_IF_STA, &sta_config);
        if (ret != ESP_OK) return ret;
    }

    // 1. Start Wi-Fi Driver
    ret = esp_wifi_start();
    if (ret != ESP_OK && ret != ESP_ERR_WIFI_STATE) return ret;

    // 2. CRITICAL: Disable Modem Power Save to guarantee uninterrupted CSI interrupts
    ret = esp_wifi_set_ps(WIFI_PS_NONE);
    if (ret != ESP_OK) return ret;

    // 3. Configure Promiscuous Filter (Passive Mode)
    if (is_passive_mode(s_hal.current_mode)) {
        wifi_promiscuous_filter_t filter = {
            .filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT | WIFI_PROMIS_FILTER_MASK_DATA
        };
        ret = esp_wifi_set_promiscuous_filter(&filter);
        if (ret != ESP_OK) return ret;

        ret = esp_wifi_set_promiscuous_rx_cb(&wifi_promiscuous_cb);
        if (ret != ESP_OK) return ret;

        ret = esp_wifi_set_promiscuous(true);
        if (ret != ESP_OK) return ret;
    }

    // 4. Configure & Enable CSI Sampling
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = 0,
    };

    ret = esp_wifi_set_csi_config(&csi_config);
    if (ret != ESP_OK) return ret;

    ret = esp_wifi_set_csi_rx_cb(&wifi_csi_cb, NULL);
    if (ret != ESP_OK) return ret;

    ret = esp_wifi_set_csi(true);
    if (ret != ESP_OK) return ret;

    return ESP_OK;
}

esp_err_t wifi_csi_hal_start(void)
{
    if (xSemaphoreTake(s_hal.lock, pdMS_TO_TICKS(1000)) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }

    if (!s_hal.is_initialized || s_hal.is_started) {
        xSemaphoreGive(s_hal.lock);
        return ESP_ERR_INVALID_STATE;
    }

    esp_err_t ret = apply_radio_config();
    if (ret != ESP_OK) {
        xSemaphoreGive(s_hal.lock);
        return ret;
    }

    s_hal.is_started = true;

    if (is_passive_mode(s_hal.current_mode)) {
        xTaskCreate(channel_hop_task, "csi_chan_hop", 3072, NULL, 3, &s_hal.channel_hop_task_handle);
    } else if (s_hal.current_mode == WIFI_CSI_MODE_HYBRID) {
        xTaskCreate(hybrid_scan_task, "csi_hybrid_scan", 4096, NULL, 2, &s_hal.hybrid_scan_task_handle);
    }

    xSemaphoreGive(s_hal.lock);
    return ESP_OK;
}

esp_err_t wifi_csi_hal_stop(void)
{
    if (xSemaphoreTake(s_hal.lock, pdMS_TO_TICKS(1000)) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }

    if (!s_hal.is_started) {
        xSemaphoreGive(s_hal.lock);
        return ESP_ERR_INVALID_STATE;
    }

    // Signal tasks to stop gracefully
    s_hal.is_started = false;

    esp_wifi_set_csi(false);
    esp_wifi_stop();

    xSemaphoreGive(s_hal.lock);
    return ESP_OK;
}

esp_err_t wifi_csi_hal_deinit(void)
{
    if (s_hal.lock == NULL) return ESP_ERR_INVALID_STATE;

    if (xSemaphoreTake(s_hal.lock, pdMS_TO_TICKS(1000)) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }

    if (!s_hal.is_initialized) {
        xSemaphoreGive(s_hal.lock);
        return ESP_ERR_INVALID_STATE;
    }

    if (s_hal.is_started) {
        s_hal.is_started = false;
        esp_wifi_set_csi(false);
        esp_wifi_stop();
    }

    if (s_hal.wifi_event_instance) {
        esp_event_handler_instance_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID, s_hal.wifi_event_instance);
        s_hal.wifi_event_instance = NULL;
    }

    if (s_hal.ip_event_instance) {
        esp_event_handler_instance_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP, s_hal.ip_event_instance);
        s_hal.ip_event_instance = NULL;
    }

    if (s_hal.csi_ringbuf) {
        vRingbufferDelete(s_hal.csi_ringbuf);
        s_hal.csi_ringbuf = NULL;
    }

    if (s_hal.metadata_queue) {
        vQueueDelete(s_hal.metadata_queue);
        s_hal.metadata_queue = NULL;
    }

    esp_wifi_deinit();

    s_hal.is_initialized = false;
    SemaphoreHandle_t lock = s_hal.lock;
    s_hal.lock = NULL;
    
    xSemaphoreGive(lock);
    vSemaphoreDelete(lock);

    return ESP_OK;
}

/* =========================================================================
 * Thread-Safe Rust FFI Buffer Consumption API
 * ========================================================================= */

bool wifi_csi_pop_frame(csi_frame_t *out_frame, uint32_t timeout_ms)
{
    if (!out_frame || !s_hal.csi_ringbuf) return false;

    size_t item_size = 0;
    TickType_t ticks_to_wait = (timeout_ms == 0xFFFFFFFF) ? portMAX_DELAY : pdMS_TO_TICKS(timeout_ms);
    
    csi_frame_t *item = (csi_frame_t *)xRingbufferReceive(s_hal.csi_ringbuf, &item_size, ticks_to_wait);

    if (item != NULL) {
        memcpy(out_frame, item, sizeof(csi_frame_t));
        vRingbufferReturnItem(s_hal.csi_ringbuf, (void *)item);
        return true;
    }

    return false;
}

bool wifi_csi_pop_metadata(metadata_frame_t *out_meta, uint32_t timeout_ms)
{
    if (!out_meta || !s_hal.metadata_queue) return false;

    TickType_t ticks_to_wait = (timeout_ms == 0xFFFFFFFF) ? portMAX_DELAY : pdMS_TO_TICKS(timeout_ms);

    return (xQueueReceive(s_hal.metadata_queue, out_meta, ticks_to_wait) == pdTRUE);
}

/* =========================================================================
 * Statistics API
 * ========================================================================= */

void wifi_csi_get_stats(wifi_csi_stats_t *out_stats)
{
    if (!out_stats) return;
    
    if (xSemaphoreTake(s_hal.lock, pdMS_TO_TICKS(100)) == pdTRUE) {
        out_stats->csi_frames_received = __atomic_load_n(&s_hal.stats.csi_frames_received, __ATOMIC_RELAXED);
        out_stats->dropped_frames = __atomic_load_n(&s_hal.stats.dropped_frames, __ATOMIC_RELAXED);
        out_stats->invalid_frames = __atomic_load_n(&s_hal.stats.invalid_frames, __ATOMIC_RELAXED);
        out_stats->reconnect_count = __atomic_load_n(&s_hal.stats.reconnect_count, __ATOMIC_RELAXED);
        out_stats->disconnect_count = __atomic_load_n(&s_hal.stats.disconnect_count, __ATOMIC_RELAXED);
        out_stats->scan_count = s_hal.stats.scan_count;
        
        xSemaphoreGive(s_hal.lock);
    }
}

void wifi_csi_reset_stats(void)
{
    if (xSemaphoreTake(s_hal.lock, pdMS_TO_TICKS(100)) == pdTRUE) {
        __atomic_store_n(&s_hal.stats.csi_frames_received, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&s_hal.stats.dropped_frames, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&s_hal.stats.invalid_frames, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&s_hal.stats.reconnect_count, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&s_hal.stats.disconnect_count, 0, __ATOMIC_RELAXED);
        s_hal.stats.scan_count = 0;

        xSemaphoreGive(s_hal.lock);
    }
}
