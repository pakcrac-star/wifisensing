
//! ============================================================================
//! Module 6: Interference & Channel Quality Evaluation (`interference.rs`)
//! ============================================================================
//! 
//! Physics & Statistical Model:
//! Evaluates the physical wireless channel quality using raw RF metrics across 
//! 1 to N spatial Wi-Fi interfaces. 
//! 
//! 1. Statistical Dispersion (Variance & MAD):
//!    Variance (σ²) measures broadband noise.
//!    Median Absolute Deviation (MAD) provides a robust measure of frequency-selective 
//!    fading, ignoring extreme outliers.
//!        MAD = median( |x_i - median(X)| )
//! 
//! 2. Z-Score Subcarrier Anomalies:
//!    Identifies narrow-band interference (spikes) by finding subcarriers where:
//!        z = |x_i - μ| / σ  > 3.0
//! 
//! 3. Packet & Burst Loss:
//!    Tracks IEEE 802.11 sequence numbers to quantify temporal link degradation.
//! 
//! Invariant Guarantees:
//! 1. NEVER filters CSI: Treats the `amplitudes` array as strictly read-only.
//! 2. Multi-Interface: Inherently supports spatial diversity (multiple antennas/radios).
//! 3. Zero Allocation: All statistical sorting uses in-place $O(N \log^2 N)$ algorithms.
//! ============================================================================

/// Maximum number of physical Wi-Fi interfaces/antennas supported.
pub const MAX_INTERFACES: usize = 4;

/// Threshold for Z-score to classify a subcarrier as anomalous (spike).
const Z_SCORE_SPIKE_THRESHOLD: f32 = 3.0;

/// Represents a raw physical observation from a single Wi-Fi interface.
#[derive(Debug, Clone)]
pub struct InterfaceObservation<'a, const SC: usize> {
    /// Hardware ID of the interface (e.g., 0 for Core 0 / Ant A)
    pub interface_id: u8,
    /// Received Signal Strength Indicator (dBm, typically -90 to -30)
    pub rssi_dbm: i8,
    /// Signal-to-Noise Ratio (dB, typically 0 to 40)
    pub snr_db: u8,
    /// IEEE 802.11 MAC Sequence Number
    pub sequence_number: u16,
    /// PHY layer CRC/FCS failure flag
    pub is_corrupted: bool,
    /// Read-only slice of CSI magnitudes
    pub amplitudes: &'a [f32; SC],
}

/// The synthesized evaluation of the wireless channel.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChannelQuality {
    /// Overall usability of the channel [0.0 (Dead) to 1.0 (Pristine)]
    pub overall_quality: f32,
    /// Packet delivery reliability [0.0 (High Loss) to 1.0 (Perfect)]
    pub packet_quality: f32,
    /// Detected RF interference [0.0 (Clear) to 1.0 (Severe Jamming/Noise)]
    pub interference_level: f32,
    /// Total count of anomalous subcarrier spikes across all interfaces
    pub total_subcarrier_spikes: u16,
}

#[derive(Debug, Clone, Copy)]
struct InterfaceState {
    last_seq: u16,
    packets_received: u32,
    packets_dropped: u32,
    burst_loss_events: u32,
}

impl Default for InterfaceState {
    fn default() -> Self {
        Self {
            last_seq: 0,
            packets_received: 0,
            packets_dropped: 0,
            burst_loss_events: 0,
        }
    }
}

/// Stateful evaluator for tracking temporal channel degradation and spatial interference.
pub struct InterferenceEvaluator {
    states: [InterfaceState; MAX_INTERFACES],
}

impl Default for InterferenceEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl InterferenceEvaluator {
    pub fn new() -> Self {
        Self {
            states: [InterfaceState::default(); MAX_INTERFACES],
        }
    }

    /// Evaluates the channel quality across one or more Wi-Fi interfaces simultaneously.
    pub fn evaluate<const SC: usize>(
        &mut self,
        observations: &[InterfaceObservation<'_, SC>],
    ) -> ChannelQuality {
        if observations.is_empty() {
            return ChannelQuality::default();
        }

        let mut best_interface_quality = 0.0f32;
        let mut aggregate_packet_quality = 0.0f32;
        let mut aggregate_interference = 0.0f32;
        let mut aggregate_spikes = 0;

        let num_obs = if observations.len() > MAX_INTERFACES {
            MAX_INTERFACES
        } else {
            observations.len()
        };

        for i in 0..num_obs {
            let obs = &observations[i];
            let state = &mut self.states[obs.interface_id as usize % MAX_INTERFACES];

            // 1. Packet & Burst Loss Evaluation
            let mut loss_penalty = 0.0f32;
            if state.packets_received > 0 {
                let seq_diff = obs.sequence_number.wrapping_sub(state.last_seq);
                if seq_diff > 1 && seq_diff < 1000 {
                    let dropped = (seq_diff - 1) as u32;
                    state.packets_dropped += dropped;
                    if dropped >= 5 {
                        state.burst_loss_events += 1;
                        loss_penalty += 0.4; // Heavy penalty for burst loss
                    } else {
                        loss_penalty += 0.1 * (dropped as f32);
                    }
                }
            }
            state.last_seq = obs.sequence_number;
            state.packets_received += 1;

            let corruption_penalty = if obs.is_corrupted { 0.5 } else { 0.0 };
            
            // Replaced `#![no_std]` hostile `.clamp()`
            let current_packet_quality = clamp_01(1.0 - loss_penalty - corruption_penalty);

            // 2. Statistical Subcarrier Evaluation
            let (mean, _variance, std_dev) = Self::compute_moments(obs.amplitudes);
            let spikes = Self::count_z_score_spikes(obs.amplitudes, mean, std_dev);
            let mad = Self::compute_mad(obs.amplitudes);

            // 3. Interference Estimation
            // High MAD relative to mean indicates frequency-selective interference.
            // Variance indicates broadband noise.
            let coefficient_of_variation = if mean > 1e-6 { std_dev / mean } else { 1.0 };
            let mad_ratio = if mean > 1e-6 { mad / mean } else { 1.0 };
            
            // Normalize SNR (0-40dB) to 0.0-1.0 inverted (interference proxy)
            let snr_factor = 1.0 - clamp_01(obs.snr_db as f32 / 40.0);
            
            let spike_factor = clamp_01(spikes as f32 / (SC as f32 * 0.1)); // 10% spikes = severe

            // Fuse interference metrics using intrinsic-free bounds
            let current_interference = (snr_factor * 0.4) 
                                     + (clamp_01(coefficient_of_variation) * 0.2)
                                     + (clamp_01(mad_ratio) * 0.2)
                                     + (spike_factor * 0.2);

            // 4. Interface Quality Synthesis
            // Convert RSSI (-90 to -30) to [0.0, 1.0]
            let rssi_norm = clamp_01((obs.rssi_dbm as f32 + 90.0) / 60.0);
            
            let interface_quality = (rssi_norm * 0.3)
                                  + ((1.0 - current_interference) * 0.4)
                                  + (current_packet_quality * 0.3);

            // "Feel their signals": Track the strongest/cleanest interface in the spatial array
            if interface_quality > best_interface_quality {
                best_interface_quality = interface_quality;
            }

            aggregate_packet_quality += current_packet_quality;
            aggregate_interference += current_interference;
            aggregate_spikes += spikes;
        }

        let float_count = num_obs as f32;

        ChannelQuality {
            overall_quality: best_interface_quality, // Selection Diversity: use the best spatial path
            packet_quality: aggregate_packet_quality / float_count,
            interference_level: aggregate_interference / float_count,
            total_subcarrier_spikes: aggregate_spikes,
        }
    }

    /// Computes Mean, Variance, and Standard Deviation in one pass.
    fn compute_moments<const SC: usize>(amplitudes: &[f32; SC]) -> (f32, f32, f32) {
        let mut sum = 0.0;
        for &val in amplitudes.iter() {
            sum += val;
        }
        let mean = sum / (SC as f32);

        let mut variance_sum = 0.0;
        for &val in amplitudes.iter() {
            let diff = val - mean;
            variance_sum += diff * diff;
        }
        let variance = variance_sum / (SC as f32);
        let std_dev = sqrtf(variance);

        (mean, variance, std_dev)
    }

    /// Counts subcarriers that violate the Z-score threshold.
    fn count_z_score_spikes<const SC: usize>(amplitudes: &[f32; SC], mean: f32, std_dev: f32) -> u16 {
        if std_dev < 1e-6 {
            return 0; // Avoid division by zero in perfectly flat channels
        }
        let mut spikes = 0;
        for &val in amplitudes.iter() {
            let z_score = abs_f32(val - mean) / std_dev;
            if z_score > Z_SCORE_SPIKE_THRESHOLD {
                spikes += 1;
            }
        }
        spikes
    }

    /// Computes the Median Absolute Deviation (MAD) using a zero-allocation Shell Sort.
    fn compute_mad<const SC: usize>(amplitudes: &[f32; SC]) -> f32 {
        let mut buffer = [0.0f32; SC];
        buffer.copy_from_slice(amplitudes);

        // 1. Find the median of X
        Self::shell_sort(&mut buffer);
        let median_x = Self::extract_median(&buffer);

        // 2. Compute absolute deviations |x_i - median(X)|
        for i in 0..SC {
            buffer[i] = abs_f32(amplitudes[i] - median_x);
        }

        // 3. Find the median of the absolute deviations
        Self::shell_sort(&mut buffer);
        Self::extract_median(&buffer)
    }

    /// High-performance, zero-allocation Shell Sort (O(N log^2 N)).
    #[inline]
    fn shell_sort<const SC: usize>(arr: &mut [f32; SC]) {
        let n = SC;
        let mut gap = n / 2;
        while gap > 0 {
            for i in gap..n {
                let temp = arr[i];
                let mut j = i;
                while j >= gap && arr[j - gap] > temp {
                    arr[j] = arr[j - gap];
                    j -= gap;
                }
                arr[j] = temp;
            }
            gap /= 2;
        }
    }

    /// Extracts the median from a sorted array.
    #[inline]
    fn extract_median<const SC: usize>(sorted_arr: &[f32; SC]) -> f32 {
        if SC % 2 == 0 {
            (sorted_arr[SC / 2 - 1] + sorted_arr[SC / 2]) / 2.0
        } else {
            sorted_arr[SC / 2]
        }
    }
}

// ============================================================================
// Math Utilities (#![no_std] intrinsic wrappers & bounds)
// ============================================================================

#[inline(always)]
fn clamp_01(val: f32) -> f32 {
    if val < 0.0 {
        0.0
    } else if val > 1.0 {
        1.0
    } else {
        val
    }
}

#[inline(always)]
fn abs_f32(val: f32) -> f32 {
    if val < 0.0 {
        -val
    } else {
        val
    }
}

#[inline(always)]
fn sqrtf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    { x.sqrt() }
    #[cfg(not(feature = "std"))]
    { libm::sqrtf(x) } // Removed `unsafe`
}
