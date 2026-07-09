/**
 * @file telemetry_transport.h
 * @brief Production telemetry transport interface
 */

#ifndef TELEMETRY_TRANSPORT_H
#define TELEMETRY_TRANSPORT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "esp_err.h"

typedef enum {
    TRANSPORT_UART = 0,
    TRANSPORT_TCP = 1,
    TRANSPORT_UDP = 2
} transport_mode_t;

/**
 * @brief Initialize telemetry system
 */
esp_err_t telemetry_transport_init(transport_mode_t mode);

/**
 * @brief Write telemetry packet
 */
void telemetry_transport_write(const char *payload);

/**
 * @brief Retrieve queued telemetry packet
 */
esp_err_t telemetry_transport_get(char *buffer, size_t max_len, uint32_t timeout_ms);

/**
 * @brief Configure remote TCP server
 */
esp_err_t telemetry_transport_set_tcp_remote(const char *host, uint16_t port);

/**
 * @brief Get current transport mode
 */
transport_mode_t telemetry_transport_get_mode(void);

/**
 * @brief Get queue fill level
 */
uint32_t telemetry_transport_queue_level(void);

#ifdef __cplusplus
}
#endif

#endif /* TELEMETRY_TRANSPORT_H */
