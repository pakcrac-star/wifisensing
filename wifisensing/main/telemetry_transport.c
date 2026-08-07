#include "telemetry_transport.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <lwip/sockets.h>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static const char *TAG = "TELEMETRY_TRANSPORT";

static int s_sock = -1;
static struct sockaddr_in s_dest_addr;
static telemetry_mode_t s_active_mode = TELEMETRY_MODE_USB;
static SemaphoreHandle_t s_transport_lock = NULL;

// Binary Sync Word for USB output stream
static const uint8_t SYNC_WORD[4] = {0xDE, 0xAD, 0xBE, 0xEF};

bool telemetry_transport_init(const char *server_ip, uint16_t port)
{
    if (s_transport_lock == NULL) {
        s_transport_lock = xSemaphoreCreateMutex();
    }

    // Disable stdout buffering for immediate USB streaming
    setvbuf(stdout, NULL, _IONBF, 0);

    // If initial IP provided, attempt UDP configuration, otherwise stay USB
    if (server_ip && strlen(server_ip) > 0 && strcmp(server_ip, "0.0.0.0") != 0) {
        telemetry_transport_set_udp(server_ip, port);
    } else {
        telemetry_transport_set_usb();
    }

    return true;
}

esp_err_t telemetry_transport_set_udp(const char *ip_str, uint16_t port)
{
    if (!ip_str || port == 0) return ESP_ERR_INVALID_ARG;

    xSemaphoreTake(s_transport_lock, portMAX_DELAY);

    // Close existing socket if open
    if (s_sock >= 0) {
        close(s_sock);
        s_sock = -1;
    }

    memset(&s_dest_addr, 0, sizeof(s_dest_addr));
    s_dest_addr.sin_family = AF_INET;
    s_dest_addr.sin_port = htons(port);
    inet_pton(AF_INET, ip_str, &s_dest_addr.sin_addr);

    s_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (s_sock < 0) {
        ESP_LOGE(TAG, "Failed to create UDP socket (errno %d). Falling back to USB.", errno);
        s_active_mode = TELEMETRY_MODE_USB;
        xSemaphoreGive(s_transport_lock);
        return ESP_FAIL;
    }

    // Set non-blocking socket so packet drops never delay the CSI pipeline
    int flags = fcntl(s_sock, F_GETFL, 0);
    fcntl(s_sock, F_SETFL, flags | O_NONBLOCK);

    s_active_mode = TELEMETRY_MODE_UDP;
    ESP_LOGI(TAG, "-> Switched Active Transport to UDP Wireless [%s:%d]", ip_str, port);

    xSemaphoreGive(s_transport_lock);
    return ESP_OK;
}

void telemetry_transport_set_usb(void)
{
    xSemaphoreTake(s_transport_lock, portMAX_DELAY);

    if (s_sock >= 0) {
        close(s_sock);
        s_sock = -1;
    }

    s_active_mode = TELEMETRY_MODE_USB;
    ESP_LOGI(TAG, "-> Switched Active Transport to USB Serial Cable Mode");

    xSemaphoreGive(s_transport_lock);
}

telemetry_mode_t telemetry_transport_get_mode(void)
{
    return s_active_mode;
}

int telemetry_transport_send(const void *payload, size_t size)
{
    if (unlikely(payload == NULL || size == 0)) return -1;

    // ---------------------------------------------------------
    // MODE 1: USB Serial Mode (Default / Cable Attached)
    // ---------------------------------------------------------
    if (s_active_mode == TELEMETRY_MODE_USB) {
        fwrite(SYNC_WORD, 1, sizeof(SYNC_WORD), stdout);
        uint16_t frame_size = (uint16_t)size;
        fwrite(&frame_size, 1, sizeof(frame_size), stdout);
        size_t written = fwrite(payload, 1, size, stdout);
        fflush(stdout);
        return (int)written;
    }

    // ---------------------------------------------------------
    // MODE 2: Wireless UDP Mode (Detached / Network Streaming)
    // ---------------------------------------------------------
    if (s_active_mode == TELEMETRY_MODE_UDP && s_sock >= 0) {
        int bytes_sent = sendto(s_sock, payload, size, 0,
                               (struct sockaddr *)&s_dest_addr, sizeof(s_dest_addr));
        if (bytes_sent > 0) return bytes_sent;

        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return 0; // Buffer full, drop frame silently to prevent lagging
        }
    }

    return -1;
}

void telemetry_transport_deinit(void)
{
    xSemaphoreTake(s_transport_lock, portMAX_DELAY);
    if (s_sock >= 0) {
        close(s_sock);
        s_sock = -1;
    }
    s_active_mode = TELEMETRY_MODE_USB;
    xSemaphoreGive(s_transport_lock);
}
