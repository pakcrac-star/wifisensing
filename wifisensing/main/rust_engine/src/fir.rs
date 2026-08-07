//! ============================================================================
//! Module 5: Finite Impulse Response (FIR) Mathematics (`fir.rs`)
//! ============================================================================
//! 
//! Mathematical Model:
//! A linear time-invariant (LTI) FIR filter computes the discrete convolution 
//! of an input signal x[n] with a finite impulse response h[k]:
//! 
//!     y[n] = Σ_{k=0}^{N-1} h[k] * x[n-k]
//! 
//! Design Methodologies Supported:
//! 1. Windowed Sinc (Analytical, computed on-device via `FirDesign`)
//! 2. Parks-McClellan / Remez (Equiripple, computed offline, ingested via `new`)
//! 3. Least Squares (Minimum mean square error, computed offline, ingested via `new`)
//! 
//! Filter Topologies:
//! - Low-pass
//! - High-pass
//! - Band-pass
//! - Band-stop
//! 
//! Invariant Guarantees:
//! 1. Mathematical Purity: Exclusively maps x[n] -> y[n] via convolution.
//! 2. Zero Allocation: The delay line x[n-k] is a statically sized ring buffer.
//! 3. Domain Isolation: Has absolutely no concept of Wi-Fi, CSI, or classification.
//! ============================================================================

use core::f32::consts::PI;

const EPSILON: f32 = 1e-6;

/// A universal, zero-allocation FIR convolution engine.
/// `N` is the number of filter taps (coefficients).
#[derive(Debug, Clone)]
pub struct FirFilter<const N: usize> {
    /// The impulse response coefficients h[k]
    coefficients: [f32; N],
    /// The discrete time delay line x[n-k]
    delay_line: [f32; N],
    /// The current insertion head for the circular delay line buffer
    head: usize,
}

impl<const N: usize> FirFilter<N> {
    /// Instantiates the convolution engine with pre-computed coefficients.
    /// Ideal for Parks-McClellan or Least Squares coefficients designed offline.
    pub fn new(coefficients: [f32; N]) -> Self {
        Self {
            coefficients,
            delay_line: [0.0; N],
            head: 0,
        }
    }

    /// Evaluates the discrete convolution equation: y[n] = Σ h[k] * x[n-k]
    #[inline(always)]
    pub fn update(&mut self, x: f32) -> f32 {
        // 1. Insert x[n] into the delay line
        self.delay_line[self.head] = x;

        let mut y = 0.0f32;
        let mut k = 0;

        // 2. Convolve: Multiply and accumulate (MAC)
        while k < N {
            // Calculate historical index: (head - k + N) % N
            let delay_idx = (self.head + N - k) % N;
            y += self.coefficients[k] * self.delay_line[delay_idx];
            k += 1;
        }

        // 3. Advance the ring buffer head
        self.head = (self.head + 1) % N;

        y
    }

    pub fn reset(&mut self) {
        self.delay_line = [0.0; N];
        self.head = 0;
    }
}

// ============================================================================
// Windowed Sinc Filter Design (Analytical Generation)
// ============================================================================

/// Analytical filter topologies for on-device coefficient generation.
/// Cutoff frequencies must be normalized (0.0 to 0.5), where 0.5 is the Nyquist frequency.
pub enum FirTopology {
    LowPass { cutoff: f32 },
    HighPass { cutoff: f32 },
    BandPass { low_cutoff: f32, high_cutoff: f32 },
    BandStop { low_cutoff: f32, high_cutoff: f32 },
}

pub struct FirDesign;

impl FirDesign {
    /// Dynamically computes FIR coefficients using the Windowed Sinc method.
    /// Applies a Blackman window to truncate the infinite sinc sequence, providing
    /// excellent stop-band attenuation.
    /// 
    /// # Constraints
    /// `N` must be odd to maintain a symmetrical Type-I linear phase filter.
    pub fn generate_windowed_sinc<const N: usize>(topology: FirTopology) -> [f32; N] {
        debug_assert!(N % 2 == 1, "FIR length must be odd");
        let mut h = [0.0f32; N];
        
        match topology {
            FirTopology::LowPass { cutoff } => {
                Self::generate_lowpass(&mut h, cutoff);
            }
            FirTopology::HighPass { cutoff } => {
                Self::generate_lowpass(&mut h, cutoff);
                Self::spectral_inversion(&mut h);
            }
            FirTopology::BandPass { low_cutoff, high_cutoff } => {
                let mut h_wide = [0.0f32; N];
                let mut h_narrow = [0.0f32; N];
                
                // f2 (high_cutoff) forms the wider low-pass base
                Self::generate_lowpass(&mut h_wide, high_cutoff);
                // f1 (low_cutoff) forms the narrower low-pass block
                Self::generate_lowpass(&mut h_narrow, low_cutoff);
                
                // Bandpass = LowPass(high) - LowPass(low)
                let mut i = 0;
                while i < N {
                    h[i] = h_wide[i] - h_narrow[i];
                    i += 1;
                }
            }
            FirTopology::BandStop { low_cutoff, high_cutoff } => {
                let mut h_low = [0.0f32; N];
                let mut h_high = [0.0f32; N];
                
                Self::generate_lowpass(&mut h_low, low_cutoff);
                Self::generate_lowpass(&mut h_high, high_cutoff);
                Self::spectral_inversion(&mut h_high); // Convert high_cutoff to high-pass
                
                // Bandstop = LowPass(low) + HighPass(high)
                let mut i = 0;
                while i < N {
                    h[i] = h_low[i] + h_high[i];
                    i += 1;
                }
            }
        }

        h
    }

    /// Generates a foundational Low-Pass windowed sinc response.
    fn generate_lowpass(h: &mut [f32], cutoff_normalized: f32) {
        let n_len = h.len();
        let m = n_len - 1;
        let center = (m as f32) / 2.0;
        let omega_c = 2.0 * PI * cutoff_normalized;

        let mut sum = 0.0f32;
        let mut i = 0;

        while i < n_len {
            let n_val = (i as f32) - center;
            
            // 1. Compute infinite ideal Sinc response with proper π normalization
            let sinc = if i == n_len / 2 {
                omega_c / PI
            } else {
                sinf(omega_c * n_val) / (PI * n_val)
            };

            // 2. Apply Blackman Window to truncate response
            // w[n] = 0.42 - 0.5 * cos(2πn/M) + 0.08 * cos(4πn/M)
            let ratio = (i as f32) / (m as f32);
            let window = 0.42 
                       - 0.50 * cosf(2.0 * PI * ratio) 
                       + 0.08 * cosf(4.0 * PI * ratio);

            h[i] = sinc * window;
            sum += h[i];
            i += 1;
        }

        // 3. Normalize to ensure unity DC gain (0 dB at 0 Hz)
        let safe_sum = if sum < EPSILON { EPSILON } else { sum };
        let inv_sum = 1.0 / safe_sum;
        i = 0;
        while i < n_len {
            h[i] *= inv_sum;
            i += 1;
        }
    }

    /// Transforms a Low-Pass filter into a High-Pass filter via Spectral Inversion.
    /// h_hp[n] = δ[n - M/2] - h_lp[n]
    fn spectral_inversion(h: &mut [f32]) {
        let n_len = h.len();
        let center = n_len / 2;
        
        let mut i = 0;
        while i < n_len {
            h[i] = -h[i];
            if i == center {
                h[i] += 1.0; // Add delta function at symmetry point
            }
            i += 1;
        }
    }
}

// ============================================================================
// Math Utilities (#![no_std] intrinsic wrappers)
// ============================================================================

#[inline(always)]
fn sinf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    { x.sin() }
    #[cfg(not(feature = "std"))]
    { libm::sinf(x) }
}

#[inline(always)]
fn cosf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    { x.cos() }
    #[cfg(not(feature = "std"))]
    { libm::cosf(x) }
}
