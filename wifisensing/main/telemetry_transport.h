
#ifndef TELEMETRY_TRANSPORT_H
#define TELEMETRY_TRANSPORT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TELEMETRY_MODE_USB = 0,
    TELEMETRY_MODE_UDP = 1
} telemetry_mode_t;

// Opaque frame wrapper matching your codebase
typedef struct {
    uint8_t data[1024];
    size_t length;
} telemetry_payload_t;

/**
 * @brief Initializes the telemetry transport subsystem (defaults to USB mode).
 */
bool telemetry_transport_init(const char *server_ip, uint16_t port);

/**
 * @brief Dynamic runtime mode switch to Wireless UDP.
 */
esp_err_t telemetry_transport_set_udp(const char *ip_str, uint16_t port);

/**
 * @brief Dynamic runtime mode switch back to USB Serial.
 */
void telemetry_transport_set_usb(void);

/**
 * @brief Returns active telemetry transport mode.
 */
telemetry_mode_t telemetry_transport_get_mode(void);

/**
 * @brief Sends telemetry payload over the currently active transport (USB or UDP).
 */
int telemetry_transport_send(const void *payload, size_t size);

/**
 * @brief Cleanly shuts down active socket connections.
 */
void telemetry_transport_deinit(void);

#ifdef __cplusplus
}
#endif

#endif // TELEMETRY_TRANSPORT_H
