/**
 * @file model.h
 * @brief ESP32-S3 WiFi CSI Radar TinyML Processing Interface
 * * Production-grade feature extraction and inference pipeline optimized for
 * ESP32-S3 Xtensa LX7 vector instructions.
 */

#ifndef MODEL_H
#define MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"

// =====================================================
// MODEL CONFIGURATION & CONSTANTS
// =====================================================

#define CSI_SUBCARRIERS         128
#define FEATURE_VECTOR_SIZE      64
#define WINDOW_SIZE               8

/**
 * @brief Memory alignment macro for ESP32-S3 SIMD/Vector instruction optimization
 */
#define TINYML_ALIGN            __attribute__((aligned(16)))

// =====================================================
// INTERFACE STRUCTS
// =====================================================

/**
 * @brief Raw CSI Frame container passed from the Wi-Fi hardware layer
 */
typedef struct {
    int8_t rssi;
    uint8_t channel;
    uint32_t timestamp;
    int16_t amplitude[CSI_SUBCARRIERS] TINYML_ALIGN;
    int16_t phase[CSI_SUBCARRIERS] TINYML_ALIGN;
} CSIFrame;

/**
 * @brief Structured Engineering Features extracted from temporal CSI data
 */
typedef struct {
    float amplitude_mean;
    float amplitude_var;
    float phase_mean;
    float phase_stability;
    float motion_energy;
    float entropy;
    float band_low;
    float band_high;
    float snr_estimate;

    /** Flattened array matching standard input shapes for TFLite Micro / ONNX */
    float feature_vector[FEATURE_VECTOR_SIZE] TINYML_ALIGN;
} CSIFeatures;

/**
 * @brief Final prediction output from the edge classifier
 */
typedef struct {
    bool motion_detected;
    bool person_present;
    float confidence;
    float motion_score;
    uint32_t timestamp_ms;
} TinyMLResult;

/**
 * @brief TinyML Engine Runtime Context Instance
 * @note Removes global state from .c files, ensuring reentrancy and thread-safety.
 */
typedef struct {
    float history[WINDOW_SIZE][CSI_SUBCARRIERS] TINYML_ALIGN;
    size_t history_index;
    bool initialized;
} tinyml_ctx_t;

// =====================================================
// API FUNCTIONS
// =====================================================

/**
 * @brief Initializes a TinyML context instance
 * * @param[out] ctx Pointer to user-allocated context structure
 * @return esp_err_t ESP_OK on success, or ESP_ERR_INVALID_ARG if ctx is NULL
 */
esp_err_t tinyml_init(tinyml_ctx_t *ctx);

/**
 * @brief Extracts features from a incoming CSI Frame using temporal history
 * * @param[in,out] ctx     Pointer to an initialized runtime context
 * @param[in]     frame   Pointer to the newly acquired raw CSI data frame
 * @param[out]    out_features Pointer to target feature destination struct
 * @return esp_err_t ESP_OK on success
 */
esp_err_t tinyml_extract_features(
    tinyml_ctx_t *ctx,
    const CSIFrame *frame,
    CSIFeatures *out_features
);

/**
 * @brief Runs edge inference on an extracted feature vector
 * * @param[in]  features Pointer to fully populated features struct
 * @param[out] out_result Pointer to prediction output structure
 * @return esp_err_t ESP_OK on success
 */
esp_err_t tinyml_predict(
    const CSIFeatures *features,
    TinyMLResult *out_result
);

// =====================================================
// SIGNAL PROCESSING UTILITIES
// =====================================================

/**
 * @brief Computes normalized energy across a slice of subcarriers
 * * @param[in] signal    Pointer to aligned signal data array
 * @param[in] sig_len   Total length of the signal array
 * @param[in] start_idx Inclusive starting index of the band
 * @param[in] end_idx   Exclusive ending index of the band
 * @return float Calculated average energy metric
 */
float compute_band_energy(
    const int16_t *signal,
    size_t sig_len,
    size_t start_idx,
    size_t end_idx
);

/**
 * @brief Performs zero-mean normalization on an array in-place
 * * @param[in,out] data   Pointer to float array data
 * @param[in]     length Number of elements to normalize
 */
void normalize_features(
    float *data,
    size_t length
);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_H */