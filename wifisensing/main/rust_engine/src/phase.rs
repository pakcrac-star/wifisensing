//! ============================================================================
//! Module 3: Phase Processing & Hardware Calibration (phase.rs)
//! ============================================================================
//!
//! Pure phase mathematics.
//!
//! Implements:
//! • Phase unwrapping
//! • Carrier Frequency Offset (CFO) removal
//! • Sampling Frequency Offset (SFO) removal
//! • Circular statistics
//!
//! Invariants
//! ----------
//! • Phase only
//! • No amplitude processing
//! • No filtering
//! • No heap allocation
//! • Fully #![no_std] compatible
//! ============================================================================

use core::f32::consts::{PI, TAU};
use libm::{cosf, fabsf, sinf, sqrtf};

const EPSILON: f32 = 1e-6;

/// Unwrap adjacent subcarrier phases.
pub fn unwrap<const N: usize>(raw: &[f32; N]) -> [f32; N] {
    let mut out = [0.0; N];

    if N == 0 {
        return out;
    }

    out[0] = raw[0];

    // Using a standard `for` loop allows the compiler (LLVM) to completely 
    // elide bounds checking, making this C-level fast.
    for i in 1..N {
        let mut delta = raw[i] - raw[i - 1];

        while delta > PI {
            delta -= TAU;
        }

        while delta < -PI {
            delta += TAU;
        }

        out[i] = out[i - 1] + delta;
    }

    out
}

/// Remove linear phase caused by CFO and SFO.
pub fn mitigate_cfo_sfo<const N: usize>(
    phase: &[f32; N],
    subcarrier_index: &[f32; N],
) -> [f32; N] {
    if N == 0 {
        return [0.0; N];
    }

    // `fold` handles all the accumulation in one pass cleanly.
    let (sum_x, sum_y, sum_x2, sum_xy) = subcarrier_index
        .iter()
        .zip(phase.iter())
        .fold((0.0, 0.0, 0.0, 0.0), |(sx, sy, sx2, sxy), (&x, &y)| {
            (sx + x, sy + y, sx2 + x * x, sxy + x * y)
        });

    let n = N as f32;
    let denominator = n * sum_x2 - sum_x * sum_x;

    let (slope, intercept) = if fabsf(denominator) > EPSILON {
        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;
        (slope, intercept)
    } else {
        (0.0, 0.0)
    };
    
    let mut corrected = [0.0; N];
    
    // `.zip()` steps through both arrays simultaneously without bounds checks.
    for (i, (&p, &idx)) in phase.iter().zip(subcarrier_index.iter()).enumerate() {
        corrected[i] = p - (slope * idx + intercept);
    }

    corrected
}

/// Complete phase calibration pipeline.
#[inline]
pub fn calibrate<const N: usize>(
    raw_phase: &[f32; N],
    subcarrier_index: &[f32; N],
) -> [f32; N] {
    let unwrapped = unwrap(raw_phase);
    mitigate_cfo_sfo(&unwrapped, subcarrier_index)
}

/// Mean phase.
pub fn mean_phase<const N: usize>(phase: &[f32; N]) -> f32 {
    if N == 0 {
        return 0.0;
    }
    
    // Iterators replace the manual while-loop accumulator
    phase.iter().sum::<f32>() / (N as f32)
}

/// Circular variance.
///
/// 0 = perfectly coherent
/// 1 = completely random
pub fn circular_variance<const N: usize>(phase: &[f32; N]) -> f32 {
    if N == 0 {
        return 1.0;
    }

    // Fold over both trigonometric functions at the same time
    let (sum_cos, sum_sin) = phase.iter().fold((0.0, 0.0), |(c, s), &p| {
        (c + cosf(p), s + sinf(p))
    });

    let inv_n = 1.0 / (N as f32);
    let mean_cos = sum_cos * inv_n;
    let mean_sin = sum_sin * inv_n;

    let r = sqrtf(mean_cos * mean_cos + mean_sin * mean_sin);

    // .clamp() is available in `core` by default
    (1.0 - r).clamp(0.0, 1.0)
}

/// Phase coherence quality.
///
/// 1 = highly stable
/// 0 = highly unstable
#[inline]
pub fn phase_quality<const N: usize>(phase: &[f32; N]) -> f32 {
    1.0 - circular_variance(phase)
}
