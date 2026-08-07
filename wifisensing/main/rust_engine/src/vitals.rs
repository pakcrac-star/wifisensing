
//! ============================================================================
//! src/vitals.rs
//! ============================================================================
//!
//! Physics: Chest displacement and biometric extraction.
//! 
//! Extracts millimeter-scale phase shifts caused by respiration and cardiac 
//! cycles. Uses IIR band-pass filtering and autocorrelation to isolate 
//! periodic micro-movements.
//!
//! CONSTRAINTS:
//! - Strictly handles biometrics (Breathing: 0.1-0.5 Hz, Pulse: 0.8-2.0 Hz).
//! - NEVER detects falls or macro-kinematics.
//! - ZERO heap allocations (no_std).
//! ============================================================================



use crate::types::BiometricFeatures;

/// Underflow threshold to prevent denormalized numbers from stalling the ESP32 FPU.
const DENORMAL_THRESHOLD: f32 = 1e-15;

/// IIR Biquad Filter state for zero-allocation DSP.
#[derive(Debug, Clone, Copy)]
pub struct BiquadFilter {
    // Coefficients
    b0: f32, b1: f32, b2: f32,
    a1: f32, a2: f32,
    // History
    x1: f32, x2: f32,
    y1: f32, y2: f32,
}

impl BiquadFilter {
    pub const fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0, b1, b2, a1, a2,
            x1: 0.0, x2: 0.0,
            y1: 0.0, y2: 0.0,
        }
    }

    /// Processes a single sample through the Direct Form 1 difference equation.
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let mut y = (self.b0 * x) + (self.b1 * self.x1) + (self.b2 * self.x2)
                  - (self.a1 * self.y1) - (self.a2 * self.y2);

        // Anti-denormalization guard: Prevents extreme FPU lag on the ESP32
        if abs(y) < DENORMAL_THRESHOLD {
            y = 0.0;
        }

        // Shift history
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;

        y
    }
}

/// Computes the autocorrelation of a signal and returns the lag of the highest peak.
/// 
/// `min_lag` and `max_lag` bound the search to physically possible biological rates.
#[inline]
pub fn find_autocorrelation_peak(
    signal: &[f32], 
    min_lag: usize, 
    max_lag: usize
) -> (usize, f32) {
    let len = signal.len();
    
    // Guard against empty signals or impossible bounds
    if len == 0 || min_lag >= len {
        return (0, 0.0);
    }

    let safe_max_lag = if max_lag >= len { len - 1 } else { max_lag };
    
    let mut best_lag = min_lag;
    let mut max_rxx = core::f32::NEG_INFINITY;
    
    // Zero-cost iterator mean calculation
    let sum: f32 = signal.iter().sum();
    let mean = sum / (len as f32);

    for lag in min_lag..=safe_max_lag {
        let count = len - lag;
        if count == 0 {
            continue;
        }
        
        let mut rxx = 0.0f32;
        
        // Using iter().zip() eliminates bounds-checking overhead in the inner loop,
        // allowing the LLVM compiler to fully vectorize the math.
        let signal_base = &signal[..count];
        let signal_offset = &signal[lag..];
        
        for (a, b) in signal_base.iter().zip(signal_offset.iter()) {
            rxx += (*a - mean) * (*b - mean);
        }
        
        rxx /= count as f32;

        if rxx > max_rxx {
            max_rxx = rxx;
            best_lag = lag;
        }
    }

    (best_lag, max_rxx)
}

/// Extracts breathing and heart rate from a raw phase history buffer.
/// 
/// Expects `phase_history` to be pre-calibrated (SFO/CFO removed).
/// `fs` is the sampling rate in Hz (e.g., 20.0 for standard CSI streaming).
pub fn extract_vitals<const N: usize>(
    phase_history: &[f32; N], // Strictly typed to array of size N to match internal buffers
    fs: f32,
    mut breath_filter: BiquadFilter,
    mut heart_filter: BiquadFilter
) -> BiometricFeatures {
    
    let mut breath_signal = [0.0f32; N];
    let mut heart_signal = [0.0f32; N];

    // 1. Band-pass filtering via zero-cost zip iterators
    for (i, &phase) in phase_history.iter().enumerate() {
        breath_signal[i] = breath_filter.process(phase);
        heart_signal[i] = heart_filter.process(phase);
    }

    // 2. Bound constraints based on sampling rate and biological limits
    // Safe conversion to usize with zero-guards
    let breath_min_lag = if fs > 0.0 { (fs / 0.5) as usize } else { 1 }; 
    let breath_max_lag = if fs > 0.0 { (fs / 0.1) as usize } else { 1 }; 

    let heart_min_lag = if fs > 0.0 { (fs / 2.0) as usize } else { 1 };  
    let heart_max_lag = if fs > 0.0 { (fs / 0.8) as usize } else { 1 };  

    // 3. Autocorrelation & Peak Detection
    let (breath_lag, breath_rxx) = find_autocorrelation_peak(&breath_signal, breath_min_lag, breath_max_lag);
    let (heart_lag, heart_rxx) = find_autocorrelation_peak(&heart_signal, heart_min_lag, heart_max_lag);

    // 4. Convert lag to frequency (Hz), protected against division by zero
    let breathing_rate_hz = if breath_lag > 0 { fs / (breath_lag as f32) } else { 0.0 };
    let heartbeat_rate_hz = if heart_lag > 0 { fs / (heart_lag as f32) } else { 0.0 };

    // Pseudo-confidence based on the strength of the autocorrelation peak
    let signal_confidence = if breath_rxx > 0.0 && heart_rxx > 0.0 { 1.0 } else { 0.1 };

    let chest_displacement_mm = breath_rxx * 10.0;

    BiometricFeatures {
        breathing_rate_hz,
        heartbeat_rate_hz,
        chest_displacement_mm,
        signal_confidence,
    }
}

// ============================================================================
// Math Utilities
// ============================================================================

/// Fast absolute value via bitmasking.
/// Prevents the need to link `libm` or FPU overhead for a simple sign removal.
#[inline(always)]
fn abs(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}
