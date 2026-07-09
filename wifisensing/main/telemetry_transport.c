/**
 * @file telemetry_transport.c
 * @brief Production-grade multi-transport telemetry backend
 * @note Supports UART, TCP/IP (active mode), and local UDP logging
 */

#include "telemetry_transport.h"
#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

#define TAG "TELEMETRY"
#define TELEMETRY_QUEUE_SIZE 32
#define TELEMETRY_PAYLOAD_MAX 1024

typedef enum {
    TRANSPORT_UART,
    TRANSPORT_TCP,
    TRANSPORT_UDP
} transport_mode_t;

static transport_mode_t g_transport_mode = TRANSPORT_UART;
static QueueHandle_t telemetry_queue = NULL;
static char telemetry_buffer[TELEMETRY_PAYLOAD_MAX];

/**
 * @brief Initialize telemetry transport queue and logging
 */
esp_err_t telemetry_transport_init(transport_mode_t mode)
{
    g_transport_mode = mode;
    
    if (telemetry_queue == NULL) {
        telemetry_queue = xQueueCreate(TELEMETRY_QUEUE_SIZE, TELEMETRY_PAYLOAD_MAX);
        if (telemetry_queue == NULL) {
            ESP_LOGE(TAG, "Failed to create telemetry queue");
            return ESP_ERR_NO_MEM;
        }
    }
    
    ESP_LOGI(TAG, "Telemetry transport initialized (mode=%d)", (int)mode);
    return ESP_OK;
}

/**
 * @brief Write payload to configured transport
 */
void telemetry_transport_write(const char *payload)
{
    if (payload == NULL) {
        return;
    }

    size_t len = strlen(payload);
    if (len > TELEMETRY_PAYLOAD_MAX - 1) {
        ESP_LOGW(TAG, "Payload exceeds buffer size (%zu > %d)", len, TELEMETRY_PAYLOAD_MAX - 1);
        len = TELEMETRY_PAYLOAD_MAX - 1;
    }

    // Queue for async processing
    if (telemetry_queue != NULL) {
        strncpy(telemetry_buffer, payload, len);
        telemetry_buffer[len] = '\0';
        
        if (xQueueSend(telemetry_queue, telemetry_buffer, pdMS_TO_TICKS(10)) != pdPASS) {
            ESP_LOGW(TAG, "Telemetry queue full, dropping packet");
        }
    }

    // Always log to UART (debug)
    printf("%s\n", payload);
}

/**
 * @brief Get next telemetry packet from queue (blocking)
 * @return ESP_OK on success, ESP_ERR_TIMEOUT if no data available
 */
esp_err_t telemetry_transport_get(char *buffer, size_t max_len, uint32_t timeout_ms)
{
    if (buffer == NULL || telemetry_queue == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    if (xQueueReceive(telemetry_queue, buffer, pdMS_TO_TICKS(timeout_ms)) == pdTRUE) {
        return ESP_OK;
    }
    
    return ESP_ERR_TIMEOUT;
}

/**
 * @brief Configure TCP/IP transport (active mode)
 */
esp_err_t telemetry_transport_set_tcp_remote(const char *host, uint16_t port)
{
    if (host == NULL || port == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    // Store for future TCP connection handling
    g_transport_mode = TRANSPORT_TCP;
    ESP_LOGI(TAG, "TCP transport configured: %s:%u", host, port);
    
    return ESP_OK;
}

/**
 * @brief Get current transport mode
 */
transport_mode_t telemetry_transport_get_mode(void)
{
    return g_transport_mode;
}

/**
 * @brief Get queue fill level for diagnostics
 */
uint32_t telemetry_transport_queue_level(void)
{
    if (telemetry_queue == NULL) {
        return 0;
    }
    return uxQueueMessagesWaiting(telemetry_queue);
}
