/**
 * @file wifi_csi.h
 * @brief Radio Hardware Abstraction Layer (HAL) for ESP32 Wi-Fi & CSI.
 *
 * Defines the strict C/Rust Foreign Function Interface (FFI) boundary.
 * Exposes hardware configuration, event loops, ISR-safe buffers, and raw frame structures.
 * 
 * STRICT RULE: ZERO signal processing, DSP, or state estimation in this header/driver.
 */

#ifndef WIFI_CSI_HAL_H
#define WIFI_CSI_HAL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"


/* =========================================================================
 * Hardware & Transport Configuration Defaults
 * ========================================================================= */

#ifndef CONFIG_CSI_MAX_SAMPLES
#define CONFIG_CSI_MAX_SAMPLES 384 // Max raw CSI I/Q subcarrier bytes
#endif

#ifndef CONFIG_CSI_RINGBUF_DEPTH
#define CONFIG_CSI_RINGBUF_DEPTH 32 // Depth of lock-free CSI frame ringbuffer
#endif

#ifndef CONFIG_METADATA_QUEUE_DEPTH
#define CONFIG_METADATA_QUEUE_DEPTH 64 // Depth of background AP metadata queue
#endif

#ifndef CONFIG_CSI_CHANNEL_HOP_INTERVAL_MS
#define CONFIG_CSI_CHANNEL_HOP_INTERVAL_MS 100 // Dwell time per channel in Passive mode
#endif

#ifndef CONFIG_CSI_HYBRID_SCAN_INTERVAL_MS
#define CONFIG_CSI_HYBRID_SCAN_INTERVAL_MS 5000 // Interval between hybrid background scans
#endif

#ifndef CONFIG_CSI_HYBRID_SCAN_DWELL_MS
#define CONFIG_CSI_HYBRID_SCAN_DWELL_MS 120 // Passive scan dwell time per channel in Hybrid mode
#endif

/* =========================================================================
 * Operational Modes
 * ========================================================================= */

/**
 * @brief Defines the physical operating topology of the Wi-Fi Radio.
 */
typedef enum {
    /** 
     * Mode 1: Passive observer.
     * No AP connection, no IP, channel hopping enabled, promiscuous mode active.
     */
    WIFI_CSI_MODE_UNKNOWN = 0,

    /** 
     * Mode 2: Station connected.
     * Connected to target AP. Channel locked to AP. Continuous traditional CSI.
     */
    WIFI_CSI_MODE_KNOWN,

    /** 
     * Mode 3: Hybrid sensing.
     * Connected to target AP (unbroken CSI/IP). Background task passively gathers
     * surrounding metadata without dropping the association.
     */
    WIFI_CSI_MODE_HYBRID
} wifi_csi_mode_t;

/* wifi_csi.h — add after wifi_csi_mode_t definition */
static inline bool is_passive_mode(wifi_csi_mode_t mode)
{
    return mode == WIFI_CSI_MODE_UNKNOWN;
}

/* =========================================================================
 * FFI Memory Layout Structures (Rust #[repr(C)] Compatible)
 * ========================================================================= */

/**
 * @brief Fast-path CSI Frame structure.
 * Populated directly inside the CSI ISR and pushed to the ringbuffer.
 */
typedef struct {
    uint64_t timestamp_us;                      // Hardware monotonic time (microseconds)
    uint8_t  mac_address[6];                    // Transmitter MAC / BSSID
    int8_t   rssi;                              // Received Signal Strength Indicator (dBm)
    int8_t   noise_floor;                       // Receiver noise floor (dBm)
    uint8_t  channel;                           // Wi-Fi Channel
    uint8_t  bandwidth;                         // 0: 20MHz, 1: 40MHz
    uint8_t  phy_mode;                          // PHY mode (11b, 11g, 11n, etc.)
    uint16_t csi_length;                        // Valid byte length in csi_data
    int8_t   csi_data[CONFIG_CSI_MAX_SAMPLES];   // Raw unparsed I/Q subcarrier array
} csi_frame_t;

/**
 * @brief Slow-path Radio Metadata Frame structure.
 * Populated by promiscuous callback or hybrid background scanner.
 */
typedef struct {
    uint64_t timestamp_us;                      // Capture timestamp (microseconds)
    uint8_t  bssid[6];                          // Target AP MAC address
    char     ssid[33];                          // Null-terminated SSID
    int8_t   rssi;                              // Received Signal Strength Indicator (dBm)
    int8_t   noise;                             // Noise floor (dBm)
    uint8_t  channel;                           // Wi-Fi Channel
    uint8_t  bandwidth;                         // Channel bandwidth
    uint8_t  phy_mode;                          // PHY standard
    uint8_t  security_type;                     // Auth mode (WPA2, WPA3, Open, etc.)
    uint16_t beacon_interval;                   // Beacon interval (TU)
} metadata_frame_t;

/* =========================================================================
 * Hardware HAL Configuration & Diagnostics
 * ========================================================================= */

/**
 * @brief Initialization parameters for wifi_csi.c HAL.
 * Renamed to prevent collision with ESP-IDF's internal `wifi_csi_config_t`.
 */
typedef struct {
    wifi_csi_mode_t mode;                       // Chosen operational mode
    char ssid[33];                              // Target AP SSID (Ignored in UNKNOWN mode)
    char password[65];                          // Target AP Password (Ignored in UNKNOWN mode)
} app_wifi_csi_config_t;

/**
 * @brief Radio and Transport Diagnostic Statistics.
 */
typedef struct {
    volatile uint32_t csi_frames_received;     // Total frames written to ringbuffer
    volatile uint32_t dropped_frames;          // Frames dropped due to ringbuffer overflow
    volatile uint32_t invalid_frames;          // Corrupted or zero-length hardware payloads
    uint32_t reconnect_count;                   // Station reconnection attempts
    uint32_t disconnect_count;                  // Unexpected disconnections
    uint32_t scan_count;                        // Background scan executions
} wifi_csi_stats_t;

/* =========================================================================
 * Public HAL API Declarations
 * ========================================================================= */

/**
 * @brief Initializes Wi-Fi hardware, netif stack, queues, and locks.
 * 
 * @param config Pointer to the configuration parameters.
 * @return ESP_OK on success, or appropriate esp_err_t error code.
 */
esp_err_t wifi_csi_hal_init(const app_wifi_csi_config_t *config);

/**
 * @brief Configures radio registers, starts Wi-Fi, and spawns channel/scan tasks.
 * 
 * @return ESP_OK on success, or appropriate esp_err_t error code.
 */
esp_err_t wifi_csi_hal_start(void);

/**
 * @brief Disables CSI, stops Wi-Fi hardware, and terminates tasks cleanly.
 * 
 * @return ESP_OK on success, or appropriate esp_err_t error code.
 */
esp_err_t wifi_csi_hal_stop(void);

/**
 * @brief Pops a single raw CSI frame from the lock-free ringbuffer.
 * Non-blocking. Designed for polling by Rust `acquisition.rs`.
 * 
 * @param out_frame Pointer to caller-allocated csi_frame_t memory.
 * @return true if a frame was copied, false if the buffer was empty.
 */
bool wifi_csi_pop_frame(csi_frame_t *out_frame, uint32_t timeout_ms);

/**
 * @brief Pops a single Radio Metadata frame from the queue.
 * Non-blocking.
 * 
 * @param out_meta Pointer to caller-allocated metadata_frame_t memory.
 * @return true if metadata was copied, false if the queue was empty.
 */
bool wifi_csi_pop_metadata(metadata_frame_t *out_meta, uint32_t timeout_ms);

/**
 * @brief Thread-safe retrieval of current HAL counters and diagnostics.
 * 
 * @param out_stats Pointer to write stats struct into.
 */
void wifi_csi_get_stats(wifi_csi_stats_t *out_stats);

/**
 * @brief Resets all diagnostic counters to zero.
 */
void wifi_csi_reset_stats(void);

#ifdef __cplusplus
}
#endif

#endif // WIFI_CSI_HAL_H
