#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define FEATURE_DIM 5
#define NUM_CLASSES 4

// Maps 1:1 with `Tensor<N>` layout in Rust (repr(C) contiguous f32 array)
typedef struct {
    float buffer[FEATURE_DIM];
} InputTensor_t;

typedef struct {
    float buffer[NUM_CLASSES];
} OutputTensor_t;

typedef struct {
    OutputTensor_t probabilities;
    uint32_t predicted_class;
    float confidence;
} CModelOutput_t;

// -----------------------------------------------------------------------------
// EMBEDDED NEURAL NETWORK WEIGHTS (Pre-trained Fixed Point / Float Parameters)
// Architecture: Input (5) -> Dense Hidden Layer (8, ReLU) -> Output (4, Softmax)
// -----------------------------------------------------------------------------

static const float HIDDEN_WEIGHTS[8][FEATURE_DIM] = {
    {  0.42f, -1.12f,  0.85f,  0.31f, -0.05f },
    { -0.89f,  0.45f, -0.22f,  1.05f,  0.64f },
    {  0.12f,  0.95f, -1.43f, -0.15f,  0.22f },
    {  1.10f,  0.15f,  0.33f, -0.82f, -0.41f },
    { -0.33f, -0.75f,  0.91f,  0.44f,  0.82f },
    {  0.65f, -0.22f, -0.61f, -1.15f, -0.09f },
    { -0.05f,  1.20f,  0.45f,  0.11f, -0.92f },
    {  0.81f, -0.55f, -0.12f,  0.72f,  0.45f }
};

static const float HIDDEN_BIASES[8] = {
    -0.10f,  0.25f, -0.05f,  0.15f, -0.20f,  0.08f,  0.12f, -0.18f
};

static const float OUTPUT_WEIGHTS[NUM_CLASSES][8] = {
    {  1.25f, -0.82f,  0.44f, -0.15f,  0.92f, -0.41f,  0.11f, -0.65f }, // Class 0: StaticEmpty
    { -0.92f,  1.15f, -0.33f,  0.82f, -0.55f,  0.71f, -0.22f,  0.44f }, // Class 1: StaticOccupied
    {  0.15f, -0.41f,  1.33f, -0.92f,  0.11f,  1.05f, -0.82f,  0.25f }, // Class 2: Walking
    { -0.55f, -0.22f, -0.91f,  1.10f, -0.82f, -0.15f,  1.15f, -0.92f }  // Class 3: Falling
};

static const float OUTPUT_BIASES[NUM_CLASSES] = {
    0.05f, -0.10f, 0.20f, -0.15f
};

// -----------------------------------------------------------------------------
// ACTIVATION FUNCTIONS & INFERENCE ENGINE
// -----------------------------------------------------------------------------

/**
 * @brief Rectified Linear Unit (ReLU) activation function.
 */
static inline float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

/**
 * @brief Numerically stable Softmax transformation across the output vector.
 */
static void softmax(float *array, int size) {
    float max_val = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        array[i] = expf(array[i] - max_val); // Subtract max for numerical overflow protection
        sum += array[i];
    }

    if (sum > 1.e-6f) {
        for (int i = 0; i < size; i++) {
            array[i] /= sum;
        }
    } else {
        float uniform = 1.0f / (float)size;
        for (int i = 0; i < size; i++) {
            array[i] = uniform;
        }
    }
}

// -----------------------------------------------------------------------------
// FFI EXPORTED INFERENCE ROUTINE
// -----------------------------------------------------------------------------

/**
 * @brief Executes the feedforward neural network pass using raw pointers 
 *        shared across the Rust FFI boundary.
 * 
 * @param input_ptr Pointer to `Tensor<FEATURE_DIM>` buffer from Rust.
 * @param out_prediction Pointer to `CModelOutput_t` structure to populate.
 * @return int 0 on success, -1 on failure.
 */
__attribute__((visibility("default")))
int ml_engine_invoke(const float *input_ptr, CModelOutput_t *out_prediction) {
    if (input_ptr == NULL || out_prediction == NULL) {
        return -1;
    }

    float hidden_layer[8];

    // 1. Input Layer -> Dense Hidden Layer (8 neurons with ReLU activation)
    for (int i = 0; i < 8; i++) {
        float sum = HIDDEN_BIASES[i];
        for (int j = 0; j < FEATURE_DIM; j++) {
            sum += HIDDEN_WEIGHTS[i][j] * input_ptr[j];
        }
        hidden_layer[i] = relu(sum);
    }

    // 2. Hidden Layer -> Output Classification Layer (4 classes)
    float logits[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        float sum = OUTPUT_BIASES[i];
        for (int j = 0; j < 8; j++) {
            sum += OUTPUT_WEIGHTS[i][j] * hidden_layer[j];
        }
        logits[i] = sum;
    }

    // 3. Softmax Normalization for Probabilities
    softmax(logits, NUM_CLASSES);

    // 4. Find Argmax and Confidence Score
    uint32_t best_class = 0;
    float max_prob = logits[0];

    for (int i = 0; i < NUM_CLASSES; i++) {
        out_prediction->probabilities.buffer[i] = logits[i];
        if (logits[i] > max_prob) {
            max_prob = logits[i];
            best_class = i;
        }
    }

    out_prediction->predicted_class = best_class;
    out_prediction->confidence = max_prob;

    return 0;
}
