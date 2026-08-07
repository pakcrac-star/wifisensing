//! ============================================================================
//! Module 3: Numerical Recovery & Stability (`recovery.rs`)
//! ============================================================================
//!
//! Purpose
//! -------
//! Provides numerical sanitization utilities for the DSP pipeline.
//! This module protects against floating-point exceptions, underflow,
//! NaN propagation, infinities, and unstable arithmetic.
//!
//! Mathematical Model
//! ------------------
//! 1. Non-finite recovery
//!
//!      x = NaN or ±∞  → fallback
//!
//! 2. Underflow protection
//!
//!      |x| < ε → fallback
//!
//! 3. Safe arithmetic
//!
//!      • division
//!      • square root
//!      • subtraction
//!
//! Design Guarantees
//! -----------------
//! • Zero allocation.
//! • Pure scalar mathematics.
//! • Fully compatible with `#![no_std]`.
//!
//! Invariants
//! ----------
//! • Never performs filtering.
//! • Never performs interpolation.
//! • Never modifies valid physical measurements.
//! • Only repairs numerically invalid values.
//! ============================================================================

/// Values smaller than this magnitude are considered numerical underflow.
pub const UNDERFLOW_THRESHOLD: f32 = 1.0e-32;

/// Stateless numerical stability helper.
#[derive(Debug, Clone, Copy, Default)]
pub struct NumAnalyzer;

impl NumAnalyzer {
    // ========================================================================
    // Validation
    // ========================================================================

    /// Returns true if a value is finite and above the underflow threshold.
    #[inline(always)]
    pub fn is_valid(x: f32) -> bool {
        x.is_finite() && Self::abs(x) >= UNDERFLOW_THRESHOLD
    }

    /// Sanitizes a floating-point value.
    ///
    /// Invalid values are replaced with zero.
    #[inline(always)]
    pub fn sanitize(x: f32) -> f32 {
        Self::sanitize_with_fallback(x, 0.0)
    }

    /// Sanitizes a floating-point value.
    ///
    /// Invalid values are replaced by `fallback`.
    #[inline(always)]
    pub fn sanitize_with_fallback(
        x: f32,
        fallback: f32,
    ) -> f32 {
        if Self::is_valid(x) {
            x
        } else {
            fallback
        }
    }

    /// Sanitizes an array in-place.
    #[inline(always)]
    pub fn sanitize_array<const N: usize>(
        values: &mut [f32; N],
    ) {
        for val in values.iter_mut() {
            *val = Self::sanitize(*val);
        }
    }

    // ========================================================================
    // Safe Arithmetic
    // ========================================================================

    /// Computes a protected division.
    #[inline(always)]
    pub fn safe_divide(
        numerator: f32,
        denominator: f32,
        fallback: f32,
    ) -> f32 {
        if Self::abs(denominator) < UNDERFLOW_THRESHOLD {
            return fallback;
        }

        Self::sanitize_with_fallback(
            numerator / denominator,
            fallback,
        )
    }

    /// Computes a protected square root.
    #[inline(always)]
    pub fn safe_sqrt(x: f32) -> f32 {
        if x <= UNDERFLOW_THRESHOLD {
            return 0.0;
        }

        Self::sanitize(Self::sqrtf(x))
    }

    /// Computes a protected subtraction.
    ///
    /// Very small differences caused by floating-point cancellation
    /// are flushed to zero.
    #[inline(always)]
    pub fn safe_subtract(
        x: f32,
        y: f32,
    ) -> f32 {
        let diff = x - y;

        if Self::abs(diff) < UNDERFLOW_THRESHOLD {
            0.0
        } else {
            Self::sanitize(diff)
        }
    }

    // ========================================================================
    // Internal Math Utilities
    // ========================================================================

    #[inline(always)]
    fn abs(x: f32) -> f32 {
        f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
    }

    #[inline(always)]
    fn sqrtf(x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            x.sqrt()
        }

        #[cfg(not(feature = "std"))]
        {
            libm::sqrtf(x)
        }
    }
}
