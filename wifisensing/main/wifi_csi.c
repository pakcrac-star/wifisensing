/**
 * @file wifi_csi.c
 * @brief High-Performance ESP32-S3 WiFi CSI Acquisition Engine
 * @note Optimized for low-latency ISR copies and SIMD vector execution.
 */

#include "wifi_csi.h"
#include <string.h>
#include <stdlib.h>

#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_timer.h"

#include "freertos/FreeRTOS.h"
#include "freertos/portmacro.h"

#define TAG "WIFI_CSI"

// =====================================================
// ENGINE STATE DEFINITIONS
// =====================================================
typedef enum {
    CSI_STATE_UNINITIALIZED = 0,
    CSI_STATE_INITIALIZED,
    CSI_STATE_RUNNING
} csi_engine_state_t;

// =====================================================
// STATIC RUNTIME STORAGE
// =====================================================
static CSIEvent csi_buffer[CSI_BUFFER_SIZE];
static volatile uint32_t buffer_head = 0;
static volatile uint32_t buffer_tail = 0;
static volatile uint32_t dropped_frames = 0;

static portMUX_TYPE buffer_lock = portMUX_INITIALIZER_UNLOCKED;
static volatile csi_engine_state_t engine_state = CSI_STATE_UNINITIALIZED;

// =====================================================
// PRIVATE FORWARD DECLARATIONS
// =====================================================
static void wifi_csi_callback(void *ctx, wifi_csi_info_t *info);

// =====================================================
// API IMPLEMENTATIONS
// =====================================================

esp_err_t wifi_csi_init(void)
{
    if (engine_state != CSI_STATE_UNINITIALIZED) {
        ESP_LOGW(TAG, "Engine already initialized");
        return ESP_ERR_INVALID_STATE;
    }

    ESP_LOGI(TAG, "Initializing WiFi CSI Hardware Layer...");

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_err_t ret = esp_wifi_init(&cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to init WiFi core: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Flush and reset memory structures
    portENTER_CRITICAL(&buffer_lock);
    buffer_head = 0;
    buffer_tail = 0;
    dropped_frames = 0;
    memset(csi_buffer, 0, sizeof(csi_buffer));
    engine_state = CSI_STATE_INITIALIZED;
    portEXIT_CRITICAL(&buffer_lock);

    ESP_LOGI(TAG, "WiFi layer active. CSI Engine Ready.");
    return ESP_OK;
}

esp_err_t wifi_csi_start(void)
{
    if (engine_state != CSI_STATE_INITIALIZED) {
        ESP_LOGE(TAG, "Invalid state for start transition: %d", engine_state);
        return ESP_ERR_INVALID_STATE;
    }

    wifi_csi_config_t config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = 0
    };

    esp_err_t ret;
    ret = esp_wifi_set_csi_config(&config);
    if (ret != ESP_OK) return ret;

    ret = esp_wifi_set_csi_rx_cb(wifi_csi_callback, NULL);
    if (ret != ESP_OK) return ret;

    ret = esp_wifi_set_csi(true);
    if (ret != ESP_OK) return ret;

    engine_state = CSI_STATE_RUNNING;
    ESP_LOGI(TAG, "CSI Acquisition Engine actively streaming.");
    return ESP_OK;
}

void wifi_csi_stop(void)
{
    if (engine_state != CSI_STATE_RUNNING) {
        return;
    }

    esp_wifi_set_csi(false);
    esp_wifi_set_csi_rx_cb(NULL, NULL);
    
    engine_state = CSI_STATE_INITIALIZED;
    ESP_LOGI(TAG, "CSI Acquisition Engine paused.");
}

/**
 * @brief Thread-safe high-speed driver callback executing within RX Interrupt context.
 */
static void IRAM_ATTR wifi_csi_callback(void *ctx, wifi_csi_info_t *info)
{
    if (unlikely(info == NULL || info->buf == NULL || engine_state != CSI_STATE_RUNNING)) {
        return;
    }

    uint32_t next = (buffer_head + 1) % CSI_BUFFER_SIZE;

    portENTER_CRITICAL_ISR(&buffer_lock);

    if (next != buffer_tail) {
        /* Reserve the slot index by advancing head while inside critical section,
         * then perform the relatively expensive copy outside the critical region. */
        uint32_t slot_index = buffer_head;
        buffer_head = next;
        portEXIT_CRITICAL_ISR(&buffer_lock);

        CSIEvent *slot = &csi_buffer[slot_index];
        slot->rssi = info->rx_ctrl.rssi;
        slot->channel = info->rx_ctrl.channel;

        uint16_t copy_length = info->len;
        if (copy_length > CSI_RAW_MAX_LEN) {
            copy_length = CSI_RAW_MAX_LEN;
        }
        slot->len = copy_length;

        /* Copy payload outside critical section to reduce ISR latency */
        memcpy(slot->csiRaw, info->buf, copy_length);

        /* Timestamp will be set by the consumer when dequeuing to avoid expensive calls in ISR */
    } else {
        dropped_frames++;
        portEXIT_CRITICAL_ISR(&buffer_lock);
    }
}

bool wifi_csi_available(void)
{
    return buffer_head != buffer_tail;
}

bool wifi_csi_get_event(CSIEvent *event)
{
    if (unlikely(event == NULL)) {
        return false;
    }

    bool data_extracted = false;
    portENTER_CRITICAL(&buffer_lock);

    if (buffer_head != buffer_tail) {
        memcpy(event, &csi_buffer[buffer_tail], sizeof(CSIEvent));
        buffer_tail = (buffer_tail + 1) % CSI_BUFFER_SIZE;
        data_extracted = true;
    }

    portEXIT_CRITICAL(&buffer_lock);
    if (data_extracted) {
        /* Set more accurate timestamp outside the critical section and avoid esp_timer_get_time() in ISR */
        event->timestamp = (uint32_t)(esp_timer_get_time() / 1000);
    }
    return data_extracted;
}

// =====================================================
// VECTOR INFERENCE DSP PIPELINE
// =====================================================

void wifi_csi_convert(const CSIEvent *__restrict__ event, CSIFrame *__restrict__ frame)
{
    if (unlikely(!event || !frame)) return;

    frame->rssi = event->rssi;
    frame->channel = event->channel;
    frame->timestamp = event->timestamp;

    const size_t available_elements = event->len;

    // Hints compiler that pointers do not alias and structures are aligned
    #pragma GCC unroll 4
    for (size_t i = 0; i < CSI_SUBCARRIERS; i++) {
        if (i < available_elements) {
            int8_t raw = event->csiRaw[i];
            frame->amplitude[i] = (int16_t)abs(raw);
            frame->phase[i] = (int16_t)raw;
        } else {
            frame->amplitude[i] = 0;
            frame->phase[i] = 0;
        }
    }
}

bool wifi_csi_read(CSIFrame *frame)
{
    CSIEvent event;

    if (!wifi_csi_get_event(&event)) {
        return false;
    }

    wifi_csi_convert(&event, frame);

    wifi_csi_denoise(frame);
    wifi_csi_normalize(frame);
    wifi_csi_phase_correct(frame);

    return true;
}

void wifi_csi_denoise(CSIFrame *__restrict__ frame)
{
    if (unlikely(!frame)) return;

    // Local pointers explicitly tell compiler memory passes are uncontested
    int16_t *__restrict__ amp = frame->amplitude;

    #pragma GCC unroll 4
    for (size_t i = 1; i < (CSI_SUBCARRIERS - 1); i++) {
        amp[i] = (int16_t)((amp[i - 1] + amp[i] + amp[i + 1]) / 3);
    }
}

void wifi_csi_normalize(CSIFrame *__restrict__ frame)
{
    if (unlikely(!frame)) return;

    int16_t *__restrict__ amp = frame->amplitude;
    int32_t accumulation = 0;

    #pragma GCC unroll 4
    for (size_t i = 0; i < CSI_SUBCARRIERS; i++) {
        accumulation += amp[i];
    }

    const int16_t arithmetic_mean = (int16_t)(accumulation / CSI_SUBCARRIERS);

    #pragma GCC unroll 4
    for (size_t i = 0; i < CSI_SUBCARRIERS; i++) {
        amp[i] -= arithmetic_mean;
    }
}

void wifi_csi_phase_correct(CSIFrame *__restrict__ frame)
{
    if (unlikely(!frame)) return;

    int16_t *__restrict__ phase = frame->phase;
    const int16_t fundamental_reference = phase[0];

    #pragma GCC unroll 4
    for (size_t i = 0; i < CSI_SUBCARRIERS; i++) {
        phase[i] -= fundamental_reference;
    }
}

uint32_t wifi_csi_get_dropped_frames(void)
{
    return dropped_frames;
}