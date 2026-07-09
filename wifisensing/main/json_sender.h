#ifndef JSON_SENDER_H
#define JSON_SENDER_H

#include "esp_err.h"
#include "model.h"
#include <stdbool.h>

/*
 * json_sender_send:
 *   features - extracted engineering features
 *   result   - prediction result
 *   frame    - the original CSIFrame (used to include RSSI + timestamp)
 *   active   - whether we are in ACTIVE (connected) mode or PASSIVE mode
 */
esp_err_t json_sender_send(
        const CSIFeatures *features,
        const TinyMLResult *result,
        const CSIFrame *frame,
        bool active
);

#endif