#include "json_sender.h"
#include <stdio.h>
#include "esp_log.h"
#include "esp_attr.h"
#include "esp_compiler.h"
#include "cJSON.h"

#define TAG "JSON_SENDER"
#define JSON_BUFFER_SIZE 1024

static char json_buffer[JSON_BUFFER_SIZE];

extern void telemetry_transport_write(const char *payload);

esp_err_t json_sender_send(const CSIFeatures *features, const TinyMLResult *result, const CSIFrame *frame, bool active)
{
    if (unlikely(features == NULL || result == NULL || frame == NULL)) {
        ESP_LOGE(TAG, "Invalid data reference passed to sender pipeline.");
        return ESP_ERR_INVALID_ARG;
    }

    cJSON *root = cJSON_CreateObject();
    if (unlikely(root == NULL)) {
        ESP_LOGE(TAG, "Failed to allocate JSON root object via heap.");
        return ESP_ERR_NO_MEM;
    }

    // Prediction object (unchanged)
    cJSON *ml_res = cJSON_CreateObject();
    if (unlikely(ml_res == NULL)) {
        ESP_LOGE(TAG, "Failed to allocate prediction item allocation slot.");
        cJSON_Delete(root);
        return ESP_ERR_NO_MEM;
    }
    cJSON_AddItemToObject(root, "prediction", ml_res);

    cJSON_AddBoolToObject(ml_res, "motion_detected", result->motion_detected);
    cJSON_AddBoolToObject(ml_res, "person_present",  result->person_present);
    cJSON_AddNumberToObject(ml_res, "confidence",     (double)result->confidence);
    cJSON_AddNumberToObject(ml_res, "motion_score",   (double)result->motion_score);
    cJSON_AddNumberToObject(ml_res, "timestamp_ms",   (double)result->timestamp_ms);

    // Features object (unchanged)
    cJSON *eng_feats = cJSON_CreateObject();
    if (unlikely(eng_feats == NULL)) {
        ESP_LOGE(TAG, "Failed to allocate features sub-object.");
        cJSON_Delete(root);
        return ESP_ERR_NO_MEM;
    }
    cJSON_AddItemToObject(root, "features", eng_feats);

    cJSON_AddNumberToObject(eng_feats, "amplitude_mean",  (double)features->amplitude_mean);
    cJSON_AddNumberToObject(eng_feats, "amplitude_var",   (double)features->amplitude_var);
    cJSON_AddNumberToObject(eng_feats, "phase_mean",      (double)features->phase_mean);
    cJSON_AddNumberToObject(eng_feats, "phase_stability", (double)features->phase_stability);
    cJSON_AddNumberToObject(eng_feats, "motion_energy",   (double)features->motion_energy);
    cJSON_AddNumberToObject(eng_feats, "entropy",         (double)features->entropy);
    cJSON_AddNumberToObject(eng_feats, "band_low",        (double)features->band_low);
    cJSON_AddNumberToObject(eng_feats, "band_high",       (double)features->band_high);
    cJSON_AddNumberToObject(eng_feats, "snr_estimate",    (double)features->snr_estimate);

    // Meta object includes RSSI and mode so downstream can treat PASSIVE vs ACTIVE differently
    cJSON *meta = cJSON_CreateObject();
    if (unlikely(meta == NULL)) {
        ESP_LOGE(TAG, "Failed to allocate meta sub-object.");
        cJSON_Delete(root);
        return ESP_ERR_NO_MEM;
    }
    cJSON_AddItemToObject(root, "meta", meta);

    cJSON_AddNumberToObject(meta, "rssi", (double)frame->rssi);
    cJSON_AddStringToObject(meta, "mode", active ? "active" : "passive");

    // Serialize and transmit
    esp_err_t ret_status = ESP_OK;
#if defined(cJSON_PrintPreallocated)
    if (!cJSON_PrintPreallocated(root, json_buffer, JSON_BUFFER_SIZE, 0)) {
        ESP_LOGE(TAG, "JSON serialization exceeded buffer size");
        ret_status = ESP_ERR_NO_MEM;
    } else {
        telemetry_transport_write(json_buffer);
    }
#else
    char *json_string = cJSON_PrintUnformatted(root);
    if (likely(json_string != NULL)) {
        telemetry_transport_write(json_string);
        cJSON_free(json_string);
    } else {
        ESP_LOGE(TAG, "Failed to run unformatted serialization layout pass.");
        ret_status = ESP_ERR_NO_MEM;
    }
#endif

    cJSON_Delete(root);
    return ret_status;
}