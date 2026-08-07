//! ============================================================================
//! Module: TinyML Model & Feature Extraction (`model.rs`)
//! ============================================================================
//!
//! Purpose
//! -------
//! Prepares and flattens unified environmental scenes into normalized 
//! inference feature tensors suitable for classification models.
//!
//! Design Guarantees
//! -----------------
//! • Zero heap allocation.
//! • Fully compatible with `#![no_std]` (inherits crate-level configuration).
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
