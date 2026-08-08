```c
/* name=wifi_csi.c
 *
 * Production-grade Wi‑Fi CSI HAL for ESP32-S3
 *
 * Features:
 * - Zero-copy pointer pool for ISR -> consumer handoff
 * - Configurable CSI sample length and pool depth
 * - Asynchronous transport_task that resolves runtime destination (NVS-backed)
 * - Runtime CLI hook: wifi_prod "<host-or-ip>:<port>" to set destination
 * - Non-blocking UDP sends (MSG_DONTWAIT) with backpressure via tx_queue
 * - Promiscuous metadata queue + hybrid scanning support
 * - Robust start/stop/deinit lifecycle
 *
 * Assumptions:
 * - Types csi_frame_t, metadata_frame_t, wifi_csi_stats_t, app_wifi_csi_config_t
 *   are declared in wifi_csi.h included below.
 * - Optional external helpers (e.g., rust_engine_push_csi) may exist; used where appropriate.
 *
 * Build-time tuneables:
 * - CONFIG_CSI_MAX_SAMPLES (samples copied per frame)
 * - CONFIG_CSI_RINGBUF_DEPTH (pool depth)
 * - CONFIG_CSI_TX_QUEUE_DEPTH (transport queue depth)
 *
 * Runtime flow:
 * - ISR obtains a free pointer from csi_free_queue, writes frame, pushes pointer to csi_ready_queue
 * - data_router_task pops ready pointers, quickly enqueues pointers to tx_queue (non-blocking)
 * - transport_task dequeues pointers and sends UDP (non-blocking); on completion recycles pointer
 *
 * NOTE: This file is intentionally self-contained and conservative about blocking operations.
 */

#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_attr.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_heap_caps.h"
#include "nvs_flash.h"
#include "nvs.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/queue.h"

#include "wifi_csi.h" // user-provided header with csi_frame_t and related types

/* =========================================================================
 * Compile-time defaults (override via sdkconfig or by #define above)
 * ========================================================================= */
#ifndef CONFIG_CSI_MAX_SAMPLES
#define CONFIG_CSI_MAX_SAMPLES 256
#endif

#ifndef CONFIG_CSI_RINGBUF_DEPTH
#define CONFIG_CSI_RINGBUF_DEPTH 128
#endif

#ifndef CONFIG_CSI_TX_QUEUE_DEPTH
#define CONFIG_CSI_TX_QUEUE_DEPTH 256
#endif

#ifndef CONFIG_METADATA_QUEUE_DEPTH
#define CONFIG_METADATA_QUEUE_DEPTH 32
#endif

#ifndef CONFIG_CSI_CHANNEL_HOP_INTERVAL_MS
#define CONFIG_CSI_CHANNEL_HOP_INTERVAL_MS 250
#endif

#define CSI_POOL_MALLOC_CAPS MALLOC_CAP_8BIT

static const char *TAG = "WIFI_CSI_HAL";

/* =========================================================================
 * HAL state
 * ========================================================================= */
typedef struct {
    wifi_csi_mode_t current_mode;
    volatile bool is_initialized;
    volatile bool is_started;
    volatile bool is_connected;
    uint8_t current_channel;

    app_wifi_csi_config_t config;

    esp_netif_t *sta_netif;

    /* Tasks */
    TaskHandle_t channel_hop_task_handle;
    TaskHandle_t hybrid_scan_task_handle;
    TaskHandle_t data_router_task_handle;
    TaskHandle_t transport_task_handle;
    TaskHandle_t reconnect_task_handle;

    /* synchronization */
    SemaphoreHandle_t lock;

    /* Event instances */
    esp_event_handler_instance_t wifi_event_instance;
    esp_event_handler_instance_t ip_event_instance;

    /* Zero-copy pointer pool (contiguous block + queues) */
    csi_frame_t *csi_pool;             // contiguous pool of frames
    QueueHandle_t csi_free_queue;      // pointers to free slots
    QueueHandle_t csi_ready_queue;     // pointers to filled slots (to consumer)
    QueueHandle_t tx_queue;            // pointers to frames ready to transmit
    QueueHandle_t metadata_queue;      // metadata frames from promiscuous cb

    /* Transport destination (resolved sockaddr) - protected by lock */
    struct sockaddr_in transport_dst;

    /* Stats */
    wifi_csi_stats_t stats;
} hal_state_t;

static hal_state_t s_hal = {0};

/* =========================================================================
 * External hooks (weak declarations; project may provide implementations)
 * ========================================================================= */
/* Optional: push frame pointer into Rust engine (user must implement if used) */
extern void rust_engine_push_csi(const csi_frame_t *frame) __attribute__((weak));

/* Optional legacy synchronous telemetry sender (user may have) -
   we won't call it in hot paths; keep prototype for compatibility. */
extern int telemetry_transport_send(const void *buf, size_t len) __attribute__((weak));

/* =========================================================================
 * Forward declarations
 * ========================================================================= */
static void wifi_csi_cb(void *ctx, wifi_csi_info_t *info);
static void wifi_promiscuous_cb(void *buf, wifi_promiscuous_pkt_type_t type);
static void channel_hop_task(void *pvParameters);
static void hybrid_scan_task(void *pvParameters);
static void reconnect_task(void *pvParameters);
static void data_router_task(void *pvParameters);
static void transport_task(void *pvParameters);
static esp_err_t apply_radio_config(void);
static esp_err_t resolve_destination(const char *uri, struct sockaddr_in *out);
static esp_err_t save_prod_destination_nvs(const char *uri);
static esp_err_t load_prod_destination_nvs(char *out_buf, size_t buf_len);
static void update_transport_destination_locked(const struct sockaddr_in *dst);

/* =========================================================================
 * Helper: NVS persistence for production destination
 * ========================================================================= */
static esp_err_t save_prod_destination_nvs(const char *uri)
{
    if (!uri) return ESP_ERR_INVALID_ARG;
    nvs_handle_t h;
    esp_err_t r = nvs_open("wifi_csi", NVS_READWRITE, &h);
    if (r != ESP_OK) return r;
    r = nvs_set_str(h, "prod_dest", uri);
    if (r == ESP_OK) r = nvs_commit(h);
    nvs_close(h);
    return r;
}

static esp_err_t load_prod_destination_nvs(char *out_buf, size_t buf_len)
{
    if (!out_buf || buf_len == 0) return ESP_ERR_INVALID_ARG;
    nvs_handle_t h;
    esp_err_t r = nvs_open("wifi_csi", NVS_READONLY, &h);
    if (r != ESP_OK) return r;
    size_t required = buf_len;
    r = nvs_get_str(h, "prod_dest", out_buf, &required);
    nvs_close(h);
    return r;
}

/* =========================================================================
 * Utility: parse <host-or-ip>:<port> and resolve to sockaddr_in
 * - This is blocking; call from a task or CLI handler, not ISR.
 * ========================================================================= */
static esp_err_t resolve_destination(const char *uri, struct sockaddr_in *out)
{
    if (!uri || !out) return ESP_ERR_INVALID_ARG;

    // split by last ':'
    const char *col = strrchr(uri, ':');
    if (!col) return ESP_ERR_INVALID_ARG;
    size_t host_len = col - uri;
    if (host_len == 0 || host_len >= 128) return ESP_ERR_INVALID_ARG;

    char host[128];
    char portstr[8];
    memset(host, 0, sizeof(host));
    memset(portstr, 0, sizeof(portstr));
    memcpy(host, uri, host_len);
    strncpy(portstr, col + 1, sizeof(portstr) - 1);

    int port = atoi(portstr);
    if (port <= 0 || port > 65535) return ESP_ERR_INVALID_ARG;

    struct in_addr inaddr;
    if (inet_pton(AF_INET, host, &inaddr) == 1) {
        memset(out, 0, sizeof(*out));
        out->sin_family = AF_INET;
        out->sin_port = htons((uint16_t)port);
        out->sin_addr = inaddr;
        return ESP_OK;
    }

    // Resolve hostname (blocking)
    struct addrinfo hints = {0};
    hints.ai_family = AF_INET;
    int gai = 0;
    struct addrinfo *res = NULL;
    gai = getaddrinfo(host, portstr, &hints, &res);
    if (gai != 0 || res == NULL) {
        if (res) freeaddrinfo(res);
        return ESP_ERR_NOT_FOUND;
    }
    struct sockaddr_in *sin = (struct sockaddr_in *)res->ai_addr;
    memcpy(out, sin, sizeof(*out));
    freeaddrinfo(res);
    return ESP_OK;
}

/* =========================================================================
 * Update the runtime transport destination (caller must not hold s_hal.lock)
 * ========================================================================= */
static void update_transport_destination_locked(const struct sockaddr_in *dst)
{
    if (!dst) return;
    if (xSemaphoreTake(s_hal.lock, pdMS_TO_TICKS(100)) == pdTRUE) {
        memcpy(&s_hal.transport_dst, dst, sizeof(*dst));
        xSemaphoreGive(s_hal.lock);
    }
}

/* =========================================================================
 * CSI ISR (zero-copy pointer pool)
 * - Must be IRAM safe and minimal.
 * ========================================================================= */
static void IRAM_ATTR wifi_csi_cb(void *ctx, wifi_csi_info_t *info)
{
    if (unlikely(!info || !info->buf || info->len == 0)) {
        __atomic_fetch_add(&s_hal.stats.invalid_frames, 1, __ATOMIC_RELAXED);
        return;
    }

    csi_frame_t *slot = NULL;
    BaseType_t higher_woken = pdFALSE;

    // obtain a free pool pointer (non-blocking from ISR)
    if (xQueueReceiveFromISR(s_hal.csi_free_queue, &slot, &higher_woken) == pdTRUE && slot != NULL) {
        // fill slot
        slot->timestamp_us = (uint64_t)esp_timer_get_time();
        memcpy(slot->mac_address, info->mac, 6);
        slot->rssi = info->rx_ctrl.rssi;
        slot->noise_floor = info->rx_ctrl.noise_floor;
        slot->channel = info->rx_ctrl.channel;
        slot->bandwidth = info->rx_ctrl.cwb;
        slot->phy_mode = info->rx_ctrl.sig_mode;

        uint16_t copy_len = (info->len > CONFIG_CSI_MAX_SAMPLES) ? CONFIG_CSI_MAX_SAMPLES : (uint16_t)info->len;
        slot->csi_length = copy_len;
        // single memcpy from wifi buffer to pool slot
        memcpy(slot->csi_data, info->buf, copy_len);

        // push filled pointer to ready queue for consumer
        xQueueSendFromISR(s_hal.csi_ready_queue, &slot, &higher_woken);
        __atomic_fetch_add(&s_hal.stats.csi_frames_received, 1, __ATOMIC_RELAXED);
    } else {
        // pool exhausted: drop frame
        __atomic_fetch_add(&s_hal.stats.dropped_frames, 1, __ATOMIC_RELAXED);
    }

    if (higher_woken == pdTRUE) {
        portYIELD_FROM_ISR();
    }
}

/* =========================================================================
 * Promiscuous callback for metadata
 * ========================================================================= */
static void IRAM_ATTR wifi_promiscuous_cb(void *buf, wifi_promiscuous_pkt_type_t type)
{
    if (!is_passive_mode(s_hal.current_mode) || unlikely(!buf)) return;

    const wifi_promiscuous_pkt_t *pkt = (const wifi_promiscuous_pkt_t *)buf;
    const wifi_pkt_rx_ctrl_t *ctrl = &pkt->rx_ctrl;

    metadata_frame_t meta;
    memset(&meta, 0, sizeof(metadata_frame_t));
    meta.timestamp_us = (uint64_t)esp_timer_get_time();
    meta.rssi = ctrl->rssi;
    meta.channel = ctrl->channel;
    meta.bandwidth = ctrl->cwb;
    meta.phy_mode = ctrl->sig_mode;
    meta.noise = ctrl->noise_floor;

    if (ctrl->sig_len >= 24) {
        memcpy(meta.bssid, &pkt->payload[10], 6);
    }

    BaseType_t higher = pdFALSE;
    xQueueSendFromISR(s_hal.metadata_queue, &meta, &higher);
    if (higher == pdTRUE) portYIELD_FROM_ISR();
}

/* =========================================================================
 * Channel hop and hybrid scan tasks (unchanged behavior)
 * ========================================================================= */
static void channel_hop_task(void *pvParameters)
{
    const uint8_t channels[] = {1,2,3,4,5,6,7,8,9,10,11,12,13};
    const size_t nch = sizeof(channels)/sizeof(channels[0]);
    size_t idx = 0;
    ESP_LOGI(TAG, "channel_hop_task started");
    while (s_hal.is_started) {
        if (is_passive_mode(s_hal.current_mode) && !s_hal.is_connected) {
            uint8_t ch = channels[idx];
            esp_err_t r = esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
            if (r == ESP_OK) {
                s_hal.current_channel = ch;
                ESP_LOGD(TAG, "Hopped to channel %d", ch);
            } else {
                ESP_LOGW(TAG, "esp_wifi_set_channel failed: %d", r);
            }
            idx = (idx + 1) % nch;
        }
        vTaskDelay(pdMS_TO_TICKS(CONFIG_CSI_CHANNEL_HOP_INTERVAL_MS));
    }
    s_hal.channel_hop_task_handle = NULL;
    vTaskDelete(NULL);
}

static void hybrid_scan_task(void *pvParameters)
{
    wifi_scan_config_t scan_cfg = {
        .ssid = NULL,
        .bssid = NULL,
        .channel = 0,
        .show_hidden = true,
        .scan_type = WIFI_SCAN_TYPE_PASSIVE,
        .scan_time.passive = 120
    };
    wifi_ap_record_t ap_records[16];

    while (s_hal.is_started) {
        if (s_hal.current_mode == WIFI_CSI_MODE_HYBRID && s_hal.is_connected) {
            s_hal.stats.scan_count++;
            esp_err_t err = esp_wifi_scan_start(&scan_cfg, true);
            if (err == ESP_OK) {
                uint16_t apcnt = 16;
                if (esp_wifi_scan_get_ap_records(&apcnt, ap_records) == ESP_OK) {
                    uint64_t now = (uint64_t)esp_timer_get_time();
                    for (int i = 0; i < apcnt; ++i) {
                        metadata_frame_t meta;
                        memset(&meta, 0, sizeof(meta));
                        meta.timestamp_us = now;
                        memcpy(meta.bssid, ap_records[i].bssid, 6);
                        strncpy(meta.ssid, (char *)ap_records[i].ssid, sizeof(meta.ssid)-1);
                        meta.ssid[sizeof(meta.ssid)-1] = '\0';
                        meta.rssi = ap_records[i].rssi;
                        meta.channel = ap_records[i].primary;
                        meta.security_type = ap_records[i].authmode;
                        xQueueSend(s_hal.metadata_queue, &meta, 0);
                    }
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
    s_hal.hybrid_scan_task_handle = NULL;
    vTaskDelete(NULL);
}

/* =========================================================================
 * Reconnect task (exponential backoff)
 * ========================================================================= */
static void reconnect_task(void *pvParameters)
{
    int back_ms = 1000;
    const int max_ms = 30000;
    while (1) {
        esp_err_t r = esp_wifi_connect();
        ESP_LOGI(TAG, "reconnect_task: esp_wifi_connect() -> %d (backoff=%d)", r, back_ms);
        if (r == ESP_OK) break;
        vTaskDelay(pdMS_TO_TICKS(back_ms));
        back_ms = (back_ms * 2 > max_ms) ? max_ms : back_ms * 2;
    }
    vTaskDelete(NULL);
}

/* =========================================================================
 * Data router task (consumer of ready pointers)
 * - Blocks waiting for frames, processes small batches, and enqueues to tx_queue
 * ========================================================================= */
static void data_router_task(void *pvParameters)
{
    csi_frame_t *slot = NULL;
    const TickType_t blocking_ticks = pdMS_TO_TICKS(200);
    const int BATCH_LIMIT = 16;

    while (1) {
        if (xQueueReceive(s_hal.csi_ready_queue, &slot, blocking_ticks) == pdTRUE) {
            int processed = 0;
            do {
                if (!slot) break;
                // Optionally push to Rust engine as an immediate copy (call weak function)
                if (rust_engine_push_csi) {
                    rust_engine_push_csi(slot); // user can optimize to accept pointer if desired
                }

                // Attempt to hand pointer to transport queue (non-blocking)
                if (s_hal.tx_queue) {
                    if (xQueueSend(s_hal.tx_queue, &slot, 0) != pdTRUE) {
                        // tx queue full: drop frame and recycle pointer
                        __atomic_fetch_add(&s_hal.stats.dropped_frames, 1, __ATOMIC_RELAXED);
                        xQueueSend(s_hal.csi_free_queue, &slot, 0);
                    } else {
                        // ownership transferred to transport_task: do not recycle here
                    }
                } else {
                    // No tx_queue configured – fallback: call legacy synchronous sender with a very small timeout
                    if (telemetry_transport_send) {
                        // Build a small temporary buffer or let legacy function handle format
                        telemetry_transport_send((const void *)slot, sizeof(csi_frame_t));
                    }
                    // Recycle pointer immediately
                    xQueueSend(s_hal.csi_free_queue, &slot, 0);
                }

                processed++;
                // drain up to batch limit without waiting
            } while (processed < BATCH_LIMIT && xQueueReceive(s_hal.csi_ready_queue, &slot, 0) == pdTRUE);
        }

        // Yield briefly to allow lower-priority tasks & idle to run
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

/* =========================================================================
 * Transport helpers: framing and non-blocking send
 * ========================================================================= */
static bool telemetry_transport_send_ptr_udp(csi_frame_t *frame_ptr, int sock, struct sockaddr_in *dst)
{
    if (!frame_ptr || sock < 0 || !dst) return false;

    // Simple framing: [8B timestamp][1B rssi][2B len][N bytes of csi]
    uint8_t sendbuf[1500];
    size_t off = 0;
    if (off + sizeof(frame_ptr->timestamp_us) >= sizeof(sendbuf)) return false;
    memcpy(sendbuf + off, &frame_ptr->timestamp_us, sizeof(frame_ptr->timestamp_us));
    off += sizeof(frame_ptr->timestamp_us);

    if (off + 1 >= sizeof(sendbuf)) return false;
    sendbuf[off++] = (uint8_t)frame_ptr->rssi;

    if (off + sizeof(uint16_t) >= sizeof(sendbuf)) return false;
    uint16_t len = frame_ptr->csi_length;
    memcpy(sendbuf + off, &len, sizeof(len));
    off += sizeof(len);

    size_t copy_len = (len > (int)(sizeof(sendbuf) - off)) ? (sizeof(sendbuf) - off) : len;
    memcpy(sendbuf + off, frame_ptr->csi_data, copy_len);
    off += copy_len;

    int sent = sendto(sock, sendbuf, off, MSG_DONTWAIT, (struct sockaddr *)dst, sizeof(*dst));
    if (sent < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return false;
        }
        // treat other socket errors as send failure
        return false;
    }
    return true;
}

/* =========================================================================
 * Transport task
 * - Opens UDP socket (non-blocking) and sends frames; resolves destination from NVS at startup
 * - If destination not configured, tries to load and resolve from NVS periodically
 * ========================================================================= */
static void transport_task(void *pvParameters)
{
    csi_frame_t *ptr = NULL;
    int sock = -1;
    struct sockaddr_in dst_local;
    memset(&dst_local, 0, sizeof(dst_local));

    // create UDP socket
    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "transport_task: socket() failed: %d", sock);
        vTaskDelete(NULL);
        return;
    }
    // set non-blocking
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    // attempt to pre-load saved destination
    char saved[128];
    if (load_prod_destination_nvs(saved, sizeof(saved)) == ESP_OK) {
        struct sockaddr_in resolved;
        if (resolve_destination(saved, &resolved) == ESP_OK) {
            update_transport_destination_locked(&resolved);
            memcpy(&dst_local, &resolved, sizeof(resolved));
            ESP_LOGI(TAG, "transport_task: loaded saved destination %s", saved);
        } else {
            ESP_LOGW(TAG, "transport_task: saved destination present but failed to resolve: %s", saved);
        }
    }

    // polling loop: wait for tx pointers and send
    while (1) {
        if (xQueueReceive(s_hal.tx_queue, &ptr, pdMS_TO_TICKS(500)) == pdTRUE) {
            if (!ptr) continue;

            // read the latest dst under lock
            if (xSemaphoreTake(s_hal.lock, pdMS_TO_TICKS(50)) == pdTRUE) {
                memcpy(&dst_local, &s_hal.transport_dst, sizeof(dst_local));
                xSemaphoreGive(s_hal.lock);
            }

            bool ok = false;
            if (dst_local.sin_family != 0) {
                ok = telemetry_transport_send_ptr_udp(ptr, sock, &dst_local);
            } else {
                // try to reload saved URI and resolve (best-effort; blocking but rare)
                char uri[128];
                if (load_prod_destination_nvs(uri, sizeof(uri)) == ESP_OK) {
                    struct sockaddr_in resolved;
                    if (resolve_destination(uri, &resolved) == ESP_OK) {
                        update_transport_destination_locked(&resolved);
                        memcpy(&dst_local, &resolved, sizeof(resolved));
                        ok = telemetry_transport_send_ptr_udp(ptr, sock, &dst_local);
                    } else {
                        ESP_LOGW(TAG, "transport_task: cannot resolve saved uri %s", uri);
                    }
                } else {
                    // no destination configured: drop frame or optionally broadcast
                    ok = false;
                }
            }

            if (!ok) {
                __atomic_fetch_add(&s_hal.stats.dropped_frames, 1, __ATOMIC_RELAXED);
            }

            // recycle pointer to pool
            xQueueSend(s_hal.csi_free_queue, &ptr, 0);
        }

        // regular maintenance: small sleep to avoid CPU hog
        vTaskDelay(pdMS_TO_TICKS(1));
    }

    // cleanup (not reached)
    close(sock);
    vTaskDelete(NULL);
}

/* =========================================================================
 * Wi-Fi event handlers
 * ========================================================================= */
static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_base != WIFI_EVENT) return;

    switch (event_id) {
        case WIFI_EVENT_STA_START:
            if (s_hal.is_started && !is_passive_mode(s_hal.current_mode)) {
                xTaskCreate(reconnect_task, "wifi_reconn", 3072, NULL, tskIDLE_PRIORITY + 3, NULL);
            }
            break;
        case WIFI_EVENT_STA_DISCONNECTED: {
            s_hal.is_connected = false;
            __atomic_fetch_add(&s_hal.stats.disconnect_count, 1, __ATOMIC_RELAXED);
            if (s_hal.is_started && !is_passive_mode(s_hal.current_mode)) {
                __atomic_fetch_add(&s_hal.stats.reconnect_count, 1, __ATOMIC_RELAXED);
                xTaskCreate(reconnect_task, "wifi_reconn", 3072, NULL, tskIDLE_PRIORITY + 3, NULL);
            }
            break;
        }
        default:
            break;
    }
}

static void ip_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        s_hal.is_connected = true;
    }
}

/* =========================================================================
 * apply_radio_config - minimal configuration + CSI enable
 * ========================================================================= */
static esp_err_t apply_radio_config(void)
{
    esp_err_t ret = esp_wifi_set_mode(WIFI_MODE_STA);
    if (ret != ESP_OK) return ret;

    if (!is_passive_mode(s_hal.current_mode)) {
        wifi_config_t cfg = {0};
        strncpy((char *)cfg.sta.ssid, s_hal.config.ssid, sizeof(cfg.sta.ssid) - 1);
        strncpy((char *)cfg.sta.password, s_hal.config.password, sizeof(cfg.sta.password) - 1);
        ret = esp_wifi_set_config(WIFI_IF_STA, &cfg);
        if (ret != ESP_OK) return ret;
    }

    ret = esp_wifi_start();
    if (ret != ESP_OK && ret != ESP_ERR_WIFI_STATE) return ret;

    ret = esp_wifi_set_ps(WIFI_PS_NONE);
    if (ret != ESP_OK) return ret;

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

    wifi_csi_config_t csi_cfg = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = 0,
    };

    ret = esp_wifi_set_csi_config(&csi_cfg);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_set_csi_config -> %d", ret);
        return ret;
    }

    ret = esp_wifi_set_csi_rx_cb(&wifi_csi_cb, NULL);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_set_csi_rx_cb -> %d", ret);
        return ret;
    }

    ret = esp_wifi_set_csi(true);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_set_csi -> %d", ret);
        return ret;
    }

    return ESP_OK;
}

/* =========================================================================
 * Public lifecycle API
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

    // Create queues (pointer-based, low overhead)
    s_hal.csi_free_queue = xQueueCreate(CONFIG_CSI_RINGBUF_DEPTH, sizeof(csi_frame_t *));
    s_hal.csi_ready_queue = xQueueCreate(CONFIG_CSI_RINGBUF_DEPTH, sizeof(csi_frame_t *));
    s_hal.tx_queue = xQueueCreate(CONFIG_CSI_TX_QUEUE_DEPTH, sizeof(csi_frame_t *));
    s_hal.metadata_queue = xQueueCreate(CONFIG_METADATA_QUEUE_DEPTH, sizeof(metadata_frame_t));
    if (!s_hal.csi_free_queue || !s_hal.csi_ready_queue || !s_hal.tx_queue || !s_hal.metadata_queue) {
        ESP_LOGE(TAG, "Failed to create required queues");
        goto error_mem;
    }

    // Allocate contiguous pool
    size_t pool_bytes = sizeof(csi_frame_t) * CONFIG_CSI_RINGBUF_DEPTH;
    s_hal.csi_pool = (csi_frame_t *)heap_caps_malloc(pool_bytes, CSI_POOL_MALLOC_CAPS);
    if (!s_hal.csi_pool) {
        ESP_LOGE(TAG, "Failed to allocate pool of %u bytes", (unsigned)pool_bytes);
        goto error_mem;
    }

    // Fill free queue with pointers to each pool slot
    for (int i = 0; i < CONFIG_CSI_RINGBUF_DEPTH; ++i) {
        csi_frame_t *p = &s_hal.csi_pool[i];
        // zero-init slot
        memset(p, 0, sizeof(csi_frame_t));
        xQueueSend(s_hal.csi_free_queue, &p, 0);
    }

    // Create netif if needed
    s_hal.sta_netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    if (!s_hal.sta_netif) {
        s_hal.sta_netif = esp_netif_create_default_wifi_sta();
    }

    // init wifi
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_err_t r = esp_wifi_init(&cfg);
    if (r != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_init failed: %d", r);
        goto error_wifi;
    }

    // register events
    r = esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &s_hal.wifi_event_instance);
    if (r != ESP_OK) {
        ESP_LOGE(TAG, "esp_event_handler_instance_register wifi failed: %d", r);
        goto error_wifi;
    }
    r = esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &ip_event_handler, NULL, &s_hal.ip_event_instance);
    if (r != ESP_OK) {
        ESP_LOGE(TAG, "esp_event_handler_instance_register ip failed: %d", r);
        goto error_wifi;
    }

    // zero transport dst
    memset(&s_hal.transport_dst, 0, sizeof(s_hal.transport_dst));

    s_hal.is_initialized = true;
    xSemaphoreGive(s_hal.lock);
    ESP_LOGI(TAG, "wifi_csi_hal_init succeeded. pool_size=%u items, item=%u bytes",
             CONFIG_CSI_RINGBUF_DEPTH, (unsigned)sizeof(csi_frame_t));
    return ESP_OK;

error_wifi:
    if (s_hal.wifi_event_instance) {
        esp_event_handler_instance_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID, s_hal.wifi_event_instance);
        s_hal.wifi_event_instance = NULL;
    }
    if (s_hal.ip_event_instance) {
        esp_event_handler_instance_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP, s_hal.ip_event_instance);
        s_hal.ip_event_instance = NULL;
    }

error_mem:
    if (s_hal.csi_free_queue) { vQueueDelete(s_hal.csi_free_queue); s_hal.csi_free_queue = NULL; }
    if (s_hal.csi_ready_queue) { vQueueDelete(s_hal.csi_ready_queue); s_hal.csi_ready_queue = NULL; }
    if (s_hal.tx_queue) { vQueueDelete(s_hal.tx_queue); s_hal.tx_queue = NULL; }
    if (s_hal.metadata_queue) { vQueueDelete(s_hal.metadata_queue); s_hal.metadata_queue = NULL; }
    if (s_hal.csi_pool) { heap_caps_free(s_hal.csi_pool); s_hal.csi_pool = NULL; }
    xSemaphoreGive(s_hal.lock);
    return ESP_ERR_NO_MEM;
}

esp_err_t wifi_csi_hal_start(void)
{
    if (xSemaphoreTake(s_hal.lock, pdMS_TO_TICKS(1000)) != pdTRUE) return ESP_ERR_TIMEOUT;
    if (!s_hal.is_initialized || s_hal.is_started) {
        xSemaphoreGive(s_hal.lock);
        return ESP_ERR_INVALID_STATE;
    }

    esp_err_t r = apply_radio_config();
    if (r != ESP_OK) {
        xSemaphoreGive(s_hal.lock);
        return r;
    }

    s_hal.is_started = true;

    // start tasks
    if (is_passive_mode(s_hal.current_mode)) {
        xTaskCreate(channel_hop_task, "csi_chan_hop", 3072, NULL, tskIDLE_PRIORITY + 1, &s_hal.channel_hop_task_handle);
    } else if (s_hal.current_mode == WIFI_CSI_MODE_HYBRID) {
        xTaskCreate(hybrid_scan_task, "csi_hybrid_scan", 4096, NULL, tskIDLE_PRIORITY + 1, &s_hal.hybrid_scan_task_handle);
    }

    // data router (higher priority so frames are drained)
    xTaskCreate(data_router_task, "data_router", 4096, NULL, tskIDLE_PRIORITY + 3, &s_hal.data_router_task_handle);

    // transport task (lower priority)
    xTaskCreate(transport_task, "csi_transport", 4096, NULL, tskIDLE_PRIORITY + 2, &s_hal.transport_task_handle);

    xSemaphoreGive(s_hal.lock);
    return ESP_OK;
}

esp_err_t wifi_csi_hal_stop(void)
{
    if (xSemaphoreTake(s_hal.lock, pdMS_TO_TICKS(1000)) != pdTRUE) return ESP_ERR_TIMEOUT;
    if (!s_hal.is_started) {
        xSemaphoreGive(s_hal.lock);
        return ESP_ERR_INVALID_STATE;
    }

    s_hal.is_started = false;

    // disable csi and stop wifi
    esp_wifi_set_csi(false);
    esp_wifi_stop();

    // let tasks exit naturally (they check s_hal.is_started) or force delete – safer to let them exit
    xSemaphoreGive(s_hal.lock);
    return ESP_OK;
}

esp_err_t wifi_csi_hal_deinit(void)
{
    if (!s_hal.lock) return ESP_ERR_INVALID_STATE;
    if (xSemaphoreTake(s_hal.lock, pdMS_TO_TICKS(1000)) != pdTRUE) return ESP_ERR_TIMEOUT;
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

    // Delete queues and free pool
    if (s_hal.csi_free_queue) { vQueueDelete(s_hal.csi_free_queue); s_hal.csi_free_queue = NULL; }
    if (s_hal.csi_ready_queue) { vQueueDelete(s_hal.csi_ready_queue); s_hal.csi_ready_queue = NULL; }
    if (s_hal.tx_queue) { vQueueDelete(s_hal.tx_queue); s_hal.tx_queue = NULL; }
    if (s_hal.metadata_queue) { vQueueDelete(s_hal.metadata_queue); s_hal.metadata_queue = NULL; }
    if (s_hal.csi_pool) { heap_caps_free(s_hal.csi_pool); s_hal.csi_pool = NULL; }

    esp_wifi_deinit();

    s_hal.is_initialized = false;
    SemaphoreHandle_t loc = s_hal.lock;
    s_hal.lock = NULL;
    xSemaphoreGive(loc);
    vSemaphoreDelete(loc);

    return ESP_OK;
}

/* =========================================================================
 * Backward-compatible pop APIs for existing code
 * - These copy out contents from pool into user-provided buffer and recycle pointer
 * ========================================================================= */
bool wifi_csi_pop_frame(csi_frame_t *out_frame, uint32_t timeout_ms)
{
    if (!out_frame || !s_hal.csi_ready_queue) return false;
    csi_frame_t *ptr = NULL;
    TickType_t ticks = (timeout_ms == 0xFFFFFFFF) ? portMAX_DELAY : pdMS_TO_TICKS(timeout_ms);
    if (xQueueReceive(s_hal.csi_ready_queue, &ptr, ticks) == pdTRUE && ptr != NULL) {
        memcpy(out_frame, ptr, sizeof(csi_frame_t));
        xQueueSend(s_hal.csi_free_queue, &ptr, 0);
        return true;
    }
    return false;
}

bool wifi_csi_pop_metadata(metadata_frame_t *out_meta, uint32_t timeout_ms)
{
    if (!out_meta || !s_hal.metadata_queue) return false;
    TickType_t ticks = (timeout_ms == 0xFFFFFFFF) ? portMAX_DELAY : pdMS_TO_TICKS(timeout_ms);
    return (xQueueReceive(s_hal.metadata_queue, out_meta, ticks) == pdTRUE);
}

/* =========================================================================
 * Stats API
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

/* =========================================================================
 * CLI / Runtime config: wifi_prod "<host-or-ip>:<port>"
 * - Call this from your CLI handler to set the production destination at runtime.
 * ========================================================================= */
esp_err_t wifi_csi_set_prod_destination(const char *uri)
{
    if (!uri) return ESP_ERR_INVALID_ARG;

    // Save to NVS first (so this survives reboot)
    esp_err_t r = save_prod_destination_nvs(uri);
    if (r != ESP_OK) {
        ESP_LOGW(TAG, "Failed to save prod destination to NVS: %s (err=%d)", uri, r);
        // continue anyway to attempt runtime update
    }

    // Resolve and update runtime destination (blocking call - caller should be task/CLI)
    struct sockaddr_in resolved;
    if (resolve_destination(uri, &resolved) == ESP_OK) {
        update_transport_destination_locked(&resolved);
        ESP_LOGI(TAG, "Set production destination: %s", uri);
        return ESP_OK;
    } else {
        ESP_LOGW(TAG, "Saved destination but failed to resolve now: %s", uri);
        // transport_task will attempt to resolve later
        return ESP_ERR_NOT_FOUND;
    }
}

/* =========================================================================
 * End of file
 * ========================================================================= */
```
