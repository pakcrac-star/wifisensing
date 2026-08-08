
//! ============================================================================
//! Module 9: Information Theory & Entropy (`entropy.rs`)
//! ============================================================================

/// Maximum subcarriers supported across 20MHz/40MHz OFDM frames
pub const MAX_SUBCARRIERS: usize = 384;

/// Temporal window size for computing time-series entropy
pub const TEMPORAL_WINDOW_SIZE: usize = 32;

/// Number of bins used to approximate the Probability Density Function (PDF) for Spatial Entropy
pub const SPATIAL_HISTOGRAM_BINS: usize = 16;

/// Epsilon to prevent log2(0)
const EPSILON: f32 = 1e-12;

/// Information theory summary of the electromagnetic field's chaos.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntropyMetrics {
    /// Normalized spectral entropy [0.0, 1.0]
    pub spectral_entropy: f32,
    /// Normalized spatial (magnitude distribution) entropy [0.0, 1.0]
    pub spatial_entropy: f32,
    /// Normalized temporal energy entropy [0.0, 1.0]
    pub temporal_entropy: f32,
}

impl Default for EntropyMetrics {
    fn default() -> Self {
        Self {
            spectral_entropy: 0.0,
            spatial_entropy: 0.0,
            temporal_entropy: 0.0,
        }
    }
}

// ============================================================================
// Information Theory Engine
// ============================================================================

pub struct EntropyEngine;

impl EntropyEngine {
    /// Computes the complete suite of Shannon entropy metrics for the channel.
    pub fn compute(
        magnitudes: &[f32; MAX_SUBCARRIERS],
        active_count: usize,
        temporal_energy_window: &[f32; TEMPORAL_WINDOW_SIZE],
    ) -> EntropyMetrics {
        let count = active_count.min(MAX_SUBCARRIERS);
        if count == 0 {
            return EntropyMetrics::default();
        }

        let spectral = Self::compute_spectral_entropy(magnitudes, count);
        let spatial = Self::compute_spatial_entropy(magnitudes, count);
        let temporal = Self::compute_temporal_entropy(temporal_energy_window);

        EntropyMetrics {
            spectral_entropy: spectral,
            spatial_entropy: spatial,
            temporal_entropy: temporal,
        }
    }

    /// Calculates normalized Spectral Entropy (Energy spread across frequencies).
    fn compute_spectral_entropy(magnitudes: &[f32; MAX_SUBCARRIERS], count: usize) -> f32 {
        let active = &magnitudes[..count];

        // Compute total power across the active spectrum
        let total_power: f32 = active.iter().map(|&m| m * m).sum();

        if total_power <= EPSILON {
            return 0.0;
        }

        let mut entropy = 0.0f32;

        // H = - Σ (p_i * log2(p_i))
        for &mag in active {
            let power_i = mag * mag;
            let p_i = power_i / total_power;

            if p_i > EPSILON {
                entropy -= p_i * log2f(p_i);
            }
        }

        // Normalize by maximum possible entropy: log2(N)
        let max_entropy = log2f(count as f32);
        if max_entropy > EPSILON {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Calculates normalized Spatial Entropy (Randomness of the physical multipath profile).
    fn compute_spatial_entropy(magnitudes: &[f32; MAX_SUBCARRIERS], count: usize) -> f32 {
        let active = &magnitudes[..count];

        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        for &mag in active {
            if mag < min_val { min_val = mag; }
            if mag > max_val { max_val = mag; }
        }

        let range = max_val - min_val;
        if range <= EPSILON {
            return 0.0; // Completely flat channel
        }

        // Build PDF histogram
        let mut bins = [0.0f32; SPATIAL_HISTOGRAM_BINS];
        let num_bins_f = SPATIAL_HISTOGRAM_BINS as f32;

        for &mag in active {
            // Safely clamp normalized float to [0.0, 1.0] to prevent out-of-bounds cast panics
            let normalized = ((mag - min_val) / range).clamp(0.0, 1.0);
            let mut bin_idx = (normalized * num_bins_f) as usize;
            if bin_idx >= SPATIAL_HISTOGRAM_BINS {
                bin_idx = SPATIAL_HISTOGRAM_BINS - 1;
            }
            bins[bin_idx] += 1.0;
        }

        let mut entropy = 0.0f32;
        let total_counts = count as f32;

        // H = - Σ (p_i * log2(p_i))
        for &bin_count in &bins {
            let p_i = bin_count / total_counts;
            if p_i > EPSILON {
                entropy -= p_i * log2f(p_i);
            }
        }

        // Normalize by log2(BINS)
        let max_entropy = log2f(num_bins_f);
        if max_entropy > EPSILON {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Calculates normalized Temporal Entropy (Randomness of signal over time).
    fn compute_temporal_entropy(energy_window: &[f32; TEMPORAL_WINDOW_SIZE]) -> f32 {
        let total_energy: f32 = energy_window.iter().sum();

        if total_energy <= EPSILON {
            return 0.0;
        }

        let mut entropy = 0.0f32;

        // H = - Σ (p_i * log2(p_i))
        for &e in energy_window {
            let p_i = e / total_energy;
            if p_i > EPSILON {
                entropy -= p_i * log2f(p_i);
            }
        }

        // Normalize by log2(WINDOW_SIZE)
        let max_entropy = log2f(TEMPORAL_WINDOW_SIZE as f32);
        if max_entropy > EPSILON {
            entropy / max_entropy
        } else {
            0.0
        }
    }
}

// ============================================================================
// Portable ESP-IDF / #![no_std] Math Utility
// ============================================================================

#[inline(always)]
fn log2f(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.log2()
    }
    #[cfg(not(feature = "std"))]
    {
        // ESP-IDF exports Newlib C libm functions directly.
        // This bindings avoids needing `libm` as an external Rust crate dependency.
        extern "C" {
            fn log2f(x: f32) -> f32;
        }
        unsafe { log2f(x) }
    }
}
