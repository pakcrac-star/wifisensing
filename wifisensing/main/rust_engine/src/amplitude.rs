//! ============================================================================
//! Module 8: Amplitude & Attenuation Physics (`amplitude.rs`)
//! ============================================================================
//! 
//! Physical Model:
//! In free space, received power P_r follows the Friis Transmission Equation:
//! 
//!     P_r = P_t * G_t * G_r * ( λ / (4 * π * R) )²
//! 
//! The magnitude A(f) of the complex channel response H(f) = I(f) + j*Q(f) is:
//! 
//!     A(f) = |H(f)| = √( I(f)² + Q(f)² )
//! 
//! Physical Interpretation of Amplitude Metrics:
//! 1. Mean (μ): Baseline path loss and steady-state attenuation.
//! 2. Variance (σ²): Frequency-selective fading and multipath chaos.
//! 3. RMS Amplitude: Total electromagnetic energy coupling across the band.
//! 4. Dynamic Range: Maximum attenuation spread caused by deep multipath nulls.
//! 5. Relative Distance / Shadowing Metric: Inverse-square attenuation proxy.
//! 
//! Invariant Guarantees:
//! 1. NO Phase Operation: Never computes atan2(Q, I), phase slope, or phase unwrap.
//! 2. Zero Filtering: Does not apply low-pass/EMA filtering to smooth magnitudes.
//! 3. Zero Allocation: All outputs are fixed-size stack arrays.
//! ============================================================================

use crate::FloatExt;
use crate::types::ComplexF32;

/// Maximum subcarriers supported matching upstream architecture (`MAX_SUBCARRIERS = 256`)
pub const MAX_SUBCARRIERS: usize = 256;

/// Epsilon to prevent log(0) and division by zero in dynamic range calculations
const EPSILON: f32 = 1e-9;

/// Physical summary of amplitude statistics across the OFDM subcarrier band.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmplitudeMetrics {
    /// Arithmetic mean amplitude μ_A
    pub mean: f32,
    /// Variance σ²_A representing frequency-selective multipath spread
    pub variance: f32,
    /// Standard deviation σ_A
    pub std_dev: f32,
    /// Root Mean Square (RMS) amplitude across subcarriers
    pub rms: f32,
    /// Minimum detected subcarrier magnitude (deep fade depth)
    pub min_amplitude: f32,
    /// Maximum detected subcarrier magnitude
    pub max_amplitude: f32,
    /// Absolute dynamic range spread (Max - Min)
    pub dynamic_range_linear: f32,
    /// Dynamic range in decibels: 20 * log10(Max / (Min + ε))
    pub dynamic_range_db: f32,
    /// Relative path-loss proxy scale based on Friis equation (1 / RMS)
    pub relative_path_loss_indicator: f32,
}

impl Default for AmplitudeMetrics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            rms: 0.0,
            min_amplitude: 0.0,
            max_amplitude: 0.0,
            dynamic_range_linear: 0.0,
            dynamic_range_db: 0.0,
            relative_path_loss_indicator: 0.0,
        }
    }
}

/// Extracted subcarrier amplitudes and physical summary for a single CSI frame.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmplitudeFrame {
    /// Raw un-smoothed subcarrier magnitudes A_k = √(I_k² + Q_k²)
    pub magnitudes: [f32; MAX_SUBCARRIERS],
    /// Total number of valid subcarriers processed
    pub subcarrier_count: u16,
    /// Derived physical metrics across the spectrum
    pub metrics: AmplitudeMetrics,
}

impl Default for AmplitudeFrame {
    fn default() -> Self {
        Self {
            magnitudes: [0.0; MAX_SUBCARRIERS],
            subcarrier_count: 0,
            metrics: AmplitudeMetrics::default(),
        }
    }
}

impl AmplitudeFrame {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.subcarrier_count as usize
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.subcarrier_count == 0
    }
}

/// Normalizes subcarrier magnitudes directly from raw integer subcarrier arrays.
pub fn normalize_amplitude(subcarriers: &[ComplexF32; MAX_SUBCARRIERS]) -> [f32; MAX_SUBCARRIERS] {
    let mut normalized = [0.0f32; MAX_SUBCARRIERS];
    for k in 0..MAX_SUBCARRIERS {
        let i = subcarriers[k].re as f32;
        let q = subcarriers[k].im as f32;
        normalized[k] = ((i * i) + (q * q)).sqrt();
    }
    normalized
}

// ============================================================================
// Amplitude Processing Engine
// ============================================================================

pub struct AmplitudeEngine;

impl AmplitudeEngine {
    /// Extracts subcarrier magnitudes and computes physical metrics from complex channel estimates.
    /// 
    /// # Arguments
    /// * `i_samples` - Floating-point In-phase components (numerically recovered)
    /// * `q_samples` - Floating-point Quadrature components (numerically recovered)
    /// * `count` - Valid subcarrier count for this frame
    /// 
    /// # Constraints
    /// Strictly operates on complex magnitudes. Phase is completely ignored.
    pub fn process(
        i_samples: &[f32; MAX_SUBCARRIERS],
        q_samples: &[f32; MAX_SUBCARRIERS],
        count: usize,
    ) -> AmplitudeFrame {
        let valid_count = count.min(MAX_SUBCARRIERS);
        if valid_count == 0 {
            return AmplitudeFrame::default();
        }

        let mut frame = AmplitudeFrame::default();
        frame.subcarrier_count = valid_count as u16;

        let mut sum_amplitude = 0.0f32;
        let mut sum_power = 0.0f32;
        let mut min_amp = f32::MAX;
        let mut max_amp = 0.0f32;

        // 1. Compute per-subcarrier magnitudes and baseline accumulation
        for k in 0..valid_count {
            let i = i_samples[k];
            let q = q_samples[k];

            // Power P_k = I² + Q²
            let power = (i * i) + (q * q);
            
            // Native f32 square root (handled by FloatExt / libm in no_std)
            let mag = power.sqrt();

            frame.magnitudes[k] = mag;
            sum_amplitude += mag;
            sum_power += power;

            if mag < min_amp { min_amp = mag; }
            if mag > max_amp { max_amp = mag; }
        }

        let n = valid_count as f32;

        // 2. Compute Mean (μ) and RMS Amplitude
        let mean = sum_amplitude / n;
        let rms = (sum_power / n).sqrt();

        // 3. Compute Variance (σ²) using second pass for numerical stability
        let mut sum_squared_diff = 0.0f32;
        for k in 0..valid_count {
            let diff = frame.magnitudes[k] - mean;
            sum_squared_diff += diff * diff;
        }
        
        let variance = sum_squared_diff / n;
        let std_dev = variance.sqrt();

        // 4. Compute Dynamic Range (Linear & Decibel)
        let dynamic_range_linear = max_amp - min_amp;
        let ratio = max_amp / (min_amp + EPSILON);
        
        let dynamic_range_db = 20.0 * ratio.max(1.0).log10();

        // 5. Friis Path Loss Indicator (Inverse relationship to RMS power)
        let relative_path_loss_indicator = 1.0 / (rms + EPSILON);

        frame.metrics = AmplitudeMetrics {
            mean,
            variance,
            std_dev,
            rms,
            min_amplitude: min_amp,
            max_amplitude: max_amp,
            dynamic_range_linear,
            dynamic_range_db,
            relative_path_loss_indicator,
        };

        frame
    }
}
