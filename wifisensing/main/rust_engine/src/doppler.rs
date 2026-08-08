//! ============================================================================
//! Module 10: Doppler & Velocity Physics (`doppler.rs`)
//! ============================================================================
//! 
//! Physical Model:
//! A moving object in the RF multipath field causes a frequency shift f_d:
//! 
//!     f_d = (2 * v * f_c) / c = (2 * v) / λ
//! 
//! Where:
//! - v: Radial velocity of the target (m/s)
//! - λ: Wavelength of the Wi-Fi carrier (~0.0525 meters at 5.7 GHz, ~0.1249 meters at 2.4 GHz)
//! - f_d: Doppler frequency shift (Hz)
//! 
//! Rearranging for Velocity:
//! 
//!     v = (λ * f_d) / 2
//! 
//! Pipeline Constraints:
//! 1. Uses Short-Time Fourier Transform (STFT) over temporal subcarrier phase/complex series.
//! 2. Performs peak detection on the Doppler spectrum to locate dominant movement velocity.
//! 3. Strictly estimates velocity and spectral spread. 
//! 4. NEVER performs classification, filtering, or phase unwrapping.
//! ============================================================================

use crate::FloatExt;

/// Number of temporal frames used for the STFT Doppler window (must be a power of 2 for FFT)
pub const DOPPLER_WINDOW_SIZE: usize = 32;

/// Maximum number of subcarriers processed
pub const MAX_SUBCARRIERS: usize = 384;

/// Standard Wi-Fi wavelengths in meters
pub const WAVELENGTH_2_4GHZ: f32 = 0.1249_f32; // 2.412 - 2.484 GHz
pub const WAVELENGTH_5GHZ: f32   = 0.0525_f32; // 5.180 - 5.825 GHz

/// Complex number structure for standard Radix-2 FFT computations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    #[inline(always)]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub fn norm_sq(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

/// Physical output containing velocity estimation derived from Doppler processing.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DopplerVelocityEstimate {
    /// Dominant target radial velocity in meters per second (m/s)
    pub dominant_velocity_mps: f32,
    /// Absolute dominant Doppler frequency shift in Hz
    pub dominant_doppler_hz: f32,
    /// Velocity dispersion / spectral bandwidth (proxy for movement complexity)
    pub spectral_spread_mps: f32,
    /// Peak Doppler power-to-noise ratio
    pub peak_power: f32,
    /// Direction indicator: +1.0 for approaching, -1.0 for receding, 0.0 for static
    pub direction: f32,
}

impl Default for DopplerVelocityEstimate {
    fn default() -> Self {
        Self {
            dominant_velocity_mps: 0.0_f32,
            dominant_doppler_hz: 0.0_f32,
            spectral_spread_mps: 0.0_f32,
            peak_power: 0.0_f32,
            direction: 0.0_f32,
        }
    }
}

// ============================================================================
// Doppler Processing Engine
// ============================================================================

pub struct DopplerEngine {
    wavelength: f32,
}

impl DopplerEngine {
    /// Constructs a Doppler processing engine with a target RF wavelength.
    pub fn new(wavelength_meters: f32) -> Self {
        Self {
            wavelength: wavelength_meters,
        }
    }

    /// Evaluates target velocity across the temporal sliding window of CSI frames.
    pub fn estimate_velocity(
        &self,
        temporal_iq_matrix: &[[Complex32; MAX_SUBCARRIERS]; DOPPLER_WINDOW_SIZE],
        sampling_rate_hz: f32,
        active_subcarrier_count: usize,
    ) -> DopplerVelocityEstimate {
        let valid_subcarriers = active_subcarrier_count.min(MAX_SUBCARRIERS);
        if valid_subcarriers == 0 || sampling_rate_hz <= 0.0_f32 {
            return DopplerVelocityEstimate::default();
        }

        // Accumulated Power Spectrum across all active subcarriers
        let mut accumulated_doppler_spectrum = [0.0_f32; DOPPLER_WINDOW_SIZE];
        let mut fft_buffer = [Complex32::new(0.0_f32, 0.0_f32); DOPPLER_WINDOW_SIZE];

        // 1. Perform STFT across every subcarrier temporally
        for sc in 0..valid_subcarriers {
            // Apply Hanning Window to suppress spectral leakage before FFT
            for t in 0..DOPPLER_WINDOW_SIZE {
                let hann = 0.5_f32 * (1.0_f32 - (2.0_f32 * core::f32::consts::PI * (t as f32) / ((DOPPLER_WINDOW_SIZE - 1) as f32)).cos());
                let sample = temporal_iq_matrix[t][sc];
                fft_buffer[t] = Complex32::new(sample.re * hann, sample.im * hann);
            }

            // In-place Radix-2 Cooley-Tukey FFT
            Self::in_place_fft(&mut fft_buffer);

            // Accumulate Power Spectral Density (PSD)
            for (k, item) in accumulated_doppler_spectrum.iter_mut().enumerate().take(DOPPLER_WINDOW_SIZE) {
                *item += fft_buffer[k].norm_sq();
            }
        }

        // 2. Normalize accumulated spectrum
        let norm_factor = 1.0_f32 / (valid_subcarriers as f32);
        for item in accumulated_doppler_spectrum.iter_mut().take(DOPPLER_WINDOW_SIZE) {
            *item *= norm_factor;
        }

        // 3. Peak Detection and Doppler Frequency Bin Mapping
        let df = sampling_rate_hz / (DOPPLER_WINDOW_SIZE as f32);
        
        let mut max_power = 0.0_f32;
        let mut peak_bin = 0usize;
        let mut total_power = 0.0_f32;

        // Skip DC offset at index 0 to ignore static reflection power
        for k in 1..DOPPLER_WINDOW_SIZE {
            let p = accumulated_doppler_spectrum[k];
            total_power += p;
            if p > max_power {
                max_power = p;
                peak_bin = k;
            }
        }

        if max_power == 0.0_f32 || total_power == 0.0_f32 {
            return DopplerVelocityEstimate::default();
        }

        // 4. Map FFT Index to Positive/Negative Doppler Bins
        let half_window = DOPPLER_WINDOW_SIZE / 2;
        let signed_doppler_hz = if peak_bin < half_window {
            peak_bin as f32 * df
        } else {
            -((DOPPLER_WINDOW_SIZE - peak_bin) as f32 * df)
        };

        // 5. Physics Mapping: v = (λ * f_d) / 2
        let dominant_velocity_mps = (self.wavelength * signed_doppler_hz) / 2.0_f32;

        // Direction indicator
        let direction = if signed_doppler_hz > 0.05_f32 {
            1.0_f32
        } else if signed_doppler_hz < -0.05_f32 {
            -1.0_f32
        } else {
            0.0_f32
        };

        // 6. Spectral Spread Calculation (Weighted Variance around Peak)
        let mut weighted_var_sum = 0.0_f32;
        for k in 1..DOPPLER_WINDOW_SIZE {
            let bin_hz = if k < half_window {
                k as f32 * df
            } else {
                -((DOPPLER_WINDOW_SIZE - k) as f32 * df)
            };
            let diff = bin_hz - signed_doppler_hz;
            weighted_var_sum += (diff * diff) * (accumulated_doppler_spectrum[k] / total_power);
        }
        let doppler_bandwidth_hz = weighted_var_sum.sqrt();
        let spectral_spread_mps = (self.wavelength * doppler_bandwidth_hz) / 2.0_f32;

        DopplerVelocityEstimate {
            dominant_velocity_mps,
            dominant_doppler_hz: signed_doppler_hz.abs(),
            spectral_spread_mps,
            peak_power: max_power,
            direction,
        }
    }

    // ========================================================================
    // In-Place Radix-2 Cooley-Tukey FFT (Zero-Allocation)
    // ========================================================================
    fn in_place_fft(buf: &mut [Complex32; DOPPLER_WINDOW_SIZE]) {
        let n = DOPPLER_WINDOW_SIZE;
        
        // Bit-reversal permutation
        let mut i = 0usize;
        let mut j = 0usize;
        while i < n {
            if i < j {
                buf.swap(i, j);
            }
            let mut bit = n >> 1;
            while bit & j != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            i += 1;
        }

        // Butterfly calculations
        let mut len = 2usize;
        while len <= n {
            let half_len = len / 2;
            let angle = -2.0_f32 * core::f32::consts::PI / (len as f32);
            let w_step = Complex32::new(angle.cos(), angle.sin());

            let mut i = 0usize;
            while i < n {
                let mut w = Complex32::new(1.0_f32, 0.0_f32);
                for k in 0..half_len {
                    let u = buf[i + k];
                    let v = Complex32::new(
                        buf[i + k + half_len].re * w.re - buf[i + k + half_len].im * w.im,
                        buf[i + k + half_len].re * w.im + buf[i + k + half_len].im * w.re,
                    );

                    buf[i + k] = Complex32::new(u.re + v.re, u.im + v.im);
                    buf[i + k + half_len] = Complex32::new(u.re - v.re, u.im - v.im);

                    let next_w_re = w.re * w_step.re - w.im * w_step.im;
                    let next_w_im = w.re * w_step.im + w.im * w_step.re;
                    w.re = next_w_re;
                    w.im = next_w_im;
                }
                i += len;
            }
            len <<= 1;
        }
    }
}
