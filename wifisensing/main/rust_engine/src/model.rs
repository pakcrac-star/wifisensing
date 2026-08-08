//! ============================================================================
//! Module: TinyML Model & Feature Extraction (`model.rs`)
//! ============================================================================
//!
//! Purpose
//! -------
//! Prepares and flattens unified environmental scenes into normalized 
//! inference feature tensors suitable for classification models.
//! Also includes the pure-Rust neural network inference engine (ported from C).
//!
//! Design Guarantees
//! -----------------
//! • Zero heap allocation.
//! • Fully compatible with `#![no_std]` (inherits crate-level configuration).
//! • Native drop-in replacement for the legacy C engine via ABI exports.
//! ============================================================================

use crate::types::{UnifiedScene, InferenceTensor, MODEL_FEATURES};
use crate::recovery::NumAnalyzer;

/// TinyML model staging and feature extraction helper.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelHandler;

impl ModelHandler {
    /// Extracts a normalized inference feature tensor from a unified scene,
    /// mapping fields precisely to the `UnifiedScene` structure definition.
    #[inline]
    pub fn extract_features(scene: &UnifiedScene) -> InferenceTensor {
        let mut features = [0.0f32; MODEL_FEATURES];

        // Map fields safely using exact nested structures from UnifiedScene
        features[0] = NumAnalyzer::sanitize(scene.channel_quality.signal_to_noise_ratio_db);
        features[1] = NumAnalyzer::sanitize(scene.channel_quality.rssi_variance);
        features[2] = NumAnalyzer::sanitize(scene.kinematics.peak_doppler_shift_hz);
        features[3] = NumAnalyzer::sanitize(scene.kinematics.estimated_velocity_m_s);
        features[4] = NumAnalyzer::sanitize(scene.kinematics.velocity_variance);
        features[5] = NumAnalyzer::sanitize(scene.biometrics.breathing_rate_hz);
        features[6] = NumAnalyzer::sanitize(scene.biometrics.heartbeat_rate_hz);
        features[7] = NumAnalyzer::sanitize(scene.biometrics.chest_displacement_mm);
        features[8] = NumAnalyzer::sanitize(scene.biometrics.signal_confidence);
        features[9] = NumAnalyzer::sanitize(scene.entropy.spatial_entropy);
        features[10] = NumAnalyzer::sanitize(scene.entropy.temporal_entropy);
        features[11] = NumAnalyzer::sanitize(scene.entropy.spectral_entropy);
        
        // Remaining slots padding for 16-feature tensor alignment
        for i in 12..MODEL_FEATURES {
            features[i] = 0.0;
        }

        // Sanitize all features in-place to ensure absolute numerical stability
        NumAnalyzer::sanitize_array(&mut features);

        InferenceTensor {
            features,
            feature_count: MODEL_FEATURES as u8,
        }
    }
}

// ============================================================================
// PURE RUST INFERENCE ENGINE (Ported from legacy ml_engine.c)
// ============================================================================

const FEATURE_DIM: usize = 16;
const NUM_CLASSES: usize = 4;

/// Represents the OutputTensor_t memory layout from the C header
#[repr(C)]
pub struct OutputTensorT {
    pub buffer: [f32; NUM_CLASSES],
}

/// Represents the CModelOutput_t memory layout from the C header
#[repr(C)]
pub struct CModelOutputT {
    pub probabilities: OutputTensorT,
    pub predicted_class: u32,
    pub confidence: f32,
}

const HIDDEN_WEIGHTS: [[f32; FEATURE_DIM]; 8] = [
    [  0.42, -1.12,  0.85,  0.31, -0.05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],
    [ -0.89,  0.45, -0.22,  1.05,  0.64,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],
    [  0.12,  0.95, -1.43, -0.15,  0.22,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],
    [  1.10,  0.15,  0.33, -0.82, -0.41,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],
    [ -0.33, -0.75,  0.91,  0.44,  0.82,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],
    [  0.65, -0.22, -0.61, -1.15, -0.09,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],
    [ -0.05,  1.20,  0.45,  0.11, -0.92,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],
    [  0.81, -0.55, -0.12,  0.72,  0.45,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ]
];

const HIDDEN_BIASES: [f32; 8] = [
    -0.10,  0.25, -0.05,  0.15, -0.20,  0.08,  0.12, -0.18
];

const OUTPUT_WEIGHTS: [[f32; 8]; NUM_CLASSES] = [
    [  1.25, -0.82,  0.44, -0.15,  0.92, -0.41,  0.11, -0.65 ], // Class 0: StaticEmpty
    [ -0.92,  1.15, -0.33,  0.82, -0.55,  0.71, -0.22,  0.44 ], // Class 1: StaticOccupied
    [  0.15, -0.41,  1.33, -0.92,  0.11,  1.05, -0.82,  0.25 ], // Class 2: Walking
    [ -0.55, -0.22, -0.91,  1.10, -0.82, -0.15,  1.15, -0.92 ]  // Class 3: Falling
];

const OUTPUT_BIASES: [f32; NUM_CLASSES] = [
    0.05, -0.10, 0.20, -0.15
];

#[inline(always)]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

extern "C" {
    // Hooks directly into the system's math library (`libm` or toolchain native)
    // preserving `no_std` compliance without needing external crate dependencies.
    fn expf(x: f32) -> f32;
}

/// Executes the feedforward neural network pass natively in Rust, maintaining
/// the exact same ABI signature and behaviour as the legacy C library.
/// Other Rust modules utilizing `extern "C" fn ml_engine_invoke` will transparently
/// resolve to this implementation at link-time.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_invoke(
    input_ptr: *const f32,
    out_prediction: *mut CModelOutputT,
) -> i32 {
    if input_ptr.is_null() || out_prediction.is_null() {
        return -1;
    }

    // 1. Input Layer -> Dense Hidden Layer (8 neurons with ReLU activation)
    let input = core::slice::from_raw_parts(input_ptr, FEATURE_DIM);
    let mut hidden_layer = [0.0f32; 8];

    for i in 0..8 {
        let mut sum = HIDDEN_BIASES[i];
        for j in 0..FEATURE_DIM {
            sum += HIDDEN_WEIGHTS[i][j] * input[j];
        }
        hidden_layer[i] = relu(sum);
    }

    // 2. Hidden Layer -> Output Classification Layer (4 classes)
    let mut logits = [0.0f32; NUM_CLASSES];
    for i in 0..NUM_CLASSES {
        let mut sum = OUTPUT_BIASES[i];
        for j in 0..8 {
            sum += OUTPUT_WEIGHTS[i][j] * hidden_layer[j];
        }
        logits[i] = sum;
    }

    // 3. Softmax Normalization for Probabilities
    let mut max_val = logits[0];
    for i in 1..NUM_CLASSES {
        if logits[i] > max_val {
            max_val = logits[i];
        }
    }

    let mut sum_exp = 0.0f32;
    for i in 0..NUM_CLASSES {
        logits[i] = expf(logits[i] - max_val); // Safe fallback via linked math
        sum_exp += logits[i];
    }

    if sum_exp > 1e-6 {
        for i in 0..NUM_CLASSES {
            logits[i] /= sum_exp;
        }
    } else {
        let uniform = 1.0 / (NUM_CLASSES as f32);
        for i in 0..NUM_CLASSES {
            logits[i] = uniform;
        }
    }

    // 4. Find Argmax and Confidence Score
    let mut best_class = 0;
    let mut max_prob = logits[0];

    let out = &mut *out_prediction;
    for i in 0..NUM_CLASSES {
        out.probabilities.buffer[i] = logits[i];
        if logits[i] > max_prob {
            max_prob = logits[i];
            best_class = i as u32;
        }
    }

    out.predicted_class = best_class;
    out.confidence = max_prob;

    0
}
