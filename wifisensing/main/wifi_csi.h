/*
 * wifi_csi.h
 *
 * ESP32-S3 WiFi CSI Acquisition Engine
 *
 * Driver
 *   |
 *   v
 * CSI Ring Buffer
 *   |
 *   v
 * CSI Processing
 *   |
 *   v
 * CSIFrame
 *
 */


#ifndef WIFI_CSI_H
#define WIFI_CSI_H


#ifdef __cplusplus
extern "C" {
#endif



#include <stdint.h>
#include <stdbool.h>

#include "esp_err.h"

#include "model.h"





// =====================================================
// CONFIGURATION
// =====================================================


#define CSI_BUFFER_SIZE     64


#define CSI_RAW_MAX_LEN     384





// =====================================================
// RAW CSI EVENT
//
// Direct snapshot from WiFi driver callback
//
// This structure is copied into ring buffer
// =====================================================


typedef struct
{

    int8_t rssi;


    uint8_t channel;


    uint32_t timestamp;



    uint16_t len;



    /*
     * Raw CSI payload from ESP WiFi driver
     *
     * ESP32-S3 CSI buffer
     */

    int8_t csiRaw[CSI_RAW_MAX_LEN];


} CSIEvent;





// =====================================================
// INITIALIZATION
// =====================================================


/*
 * Initialize WiFi driver
 *
 * Must be called before wifi_csi_start()
 */

esp_err_t wifi_csi_init(void);





// =====================================================
// CSI CONTROL
// =====================================================


/*
 * Enable CSI capture
 */

esp_err_t wifi_csi_start(void);





/*
 * Disable CSI capture
 */

void wifi_csi_stop(void);





// =====================================================
// DATA ACCESS
// =====================================================



/*
 * Check if CSI data available
 */

bool wifi_csi_available(void);





/*
 * Read processed CSI frame
 *
 * Pipeline:
 *
 * Raw CSI
 *    |
 * Convert
 *    |
 * Denoise
 *    |
 * Normalize
 *    |
 * Phase correction
 *
 */

bool wifi_csi_read(
        CSIFrame *frame
);





/*
 * Read raw CSI event
 *
 * Normally used internally
 * or for debugging
 */

bool wifi_csi_get_event(
        CSIEvent *event
);





// =====================================================
// PROCESSING FUNCTIONS
// =====================================================



/*
 * Convert driver CSI event
 * into ML frame
 */

void wifi_csi_convert(
        const CSIEvent *event,
        CSIFrame *frame
);





/*
 * Noise reduction
 */

void wifi_csi_denoise(
        CSIFrame *frame
);





/*
 * Remove amplitude DC offset
 */

void wifi_csi_normalize(
        CSIFrame *frame
);





/*
 * Remove phase reference offset
 */

void wifi_csi_phase_correct(
        CSIFrame *frame
);





// =====================================================
// DIAGNOSTICS
// =====================================================



/*
 * Number of CSI packets dropped
 * because ring buffer was full
 */

uint32_t wifi_csi_get_dropped_frames(void);





#ifdef __cplusplus
}
#endif


#endif /* WIFI_CSI_H */