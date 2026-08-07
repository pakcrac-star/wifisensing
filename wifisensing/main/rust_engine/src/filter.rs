//! ============================================================================
//! Module 4: Digital Signal Processing & Filters (`filters.rs`)
//! ============================================================================

use core::f32::consts::PI;

//const EPSILON: f32 = 1e-6;

// ============================================================================
// 1. Exponential Moving Average (EMA)
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct EmaFilter {
    alpha: f32,
    y_prev: f32,
    initialized: bool,
}

impl EmaFilter {
    /// Creates a new EMA filter. `alpha` must be in the range (0.0, 1.0].
    pub fn new(alpha: f32) -> Self {
        debug_assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0.0, 1.0]");
        Self {
            alpha,
            y_prev: 0.0,
            initialized: false,
        }
    }

    /// Applies the EMA difference equation: y_n = α * x_n + (1 - α) * y_{n-1}
    #[must_use]
    #[inline(always)]
    pub fn update(&mut self, x: f32) -> f32 {
        if !self.initialized {
            self.y_prev = x;
            self.initialized = true;
            return x;
        }
        self.y_prev = self.alpha * x + (1.0 - self.alpha) * self.y_prev;
        self.y_prev
    }

    pub fn current(&self) -> f32 {
        self.y_prev
    }

    pub fn reset(&mut self) {
        self.initialized = false;
        self.y_prev = 0.0;
    }
}

impl Default for EmaFilter {
    fn default() -> Self {
        Self::new(0.1)
    }
}

// ============================================================================
// 2. Simple Moving Average (SMA)
// ============================================================================

#[derive(Debug, Clone)]
pub struct SmaFilter<const N: usize> {
    buffer: [f32; N],
    index: usize,
    count: usize,
    sum: f32,
}

impl<const N: usize> SmaFilter<N> {
    pub fn new() -> Self {
        debug_assert!(N > 0, "SMA filter length N must be greater than 0");
        Self {
            buffer: [0.0; N],
            index: 0,
            count: 0,
            sum: 0.0,
        }
    }

    #[must_use]
    #[inline(always)]
    pub fn update(&mut self, x: f32) -> f32 {
        self.sum -= self.buffer[self.index];
        self.buffer[self.index] = x;
        self.sum += x;

        self.index = (self.index + 1) % N;
        if self.count < N {
            self.count += 1;
        }

        self.sum / (self.count as f32)
    }

    pub fn current(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / (self.count as f32)
        }
    }

    pub fn reset(&mut self) {
        self.buffer = [0.0; N];
        self.index = 0;
        self.count = 0;
        self.sum = 0.0;
    }
}

impl<const N: usize> Default for SmaFilter<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 3. Median Filter (Order Statistics)
// ============================================================================

#[derive(Debug, Clone)]
pub struct MedianFilter<const N: usize> {
    buffer: [f32; N],
    index: usize,
    count: usize,
    median: f32,
}

impl<const N: usize> MedianFilter<N> {
    pub fn new() -> Self {
        debug_assert!(N > 0, "Median filter length N must be greater than 0");
        Self {
            buffer: [0.0; N],
            index: 0,
            count: 0,
            median: 0.0,
        }
    }

    #[must_use]
    #[inline(always)]
    pub fn update(&mut self, x: f32) -> f32 {
        self.buffer[self.index] = x;
        self.index = (self.index + 1) % N;
        if self.count < N {
            self.count += 1;
        }

        let mut sorted = [0.0f32; N];
        sorted[..self.count].copy_from_slice(&self.buffer[..self.count]);

        // Idiomatic zero-allocation slice sorting from core library
        sorted[..self.count].sort_unstable_by(|a, b| {
            a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
        });

        self.median = if self.count % 2 == 0 {
            let mid = self.count / 2;
            (sorted[mid - 1] + sorted[mid]) * 0.5
        } else {
            sorted[self.count / 2]
        };

        self.median
    }

    pub fn current(&self) -> f32 {
        self.median
    }

    pub fn reset(&mut self) {
        self.buffer = [0.0; N];
        self.index = 0;
        self.count = 0;
        self.median = 0.0;
    }
}

impl<const N: usize> Default for MedianFilter<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 4. Butterworth / General Biquad Filter (Direct Form 1)
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct BiquadFilter {
    b0: f32, b1: f32, b2: f32,
    a1: f32, a2: f32,
    x1: f32, x2: f32,
    y1: f32, y2: f32,
}

impl BiquadFilter {
    pub fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0, b1, b2, a1, a2,
            x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0,
        }
    }

    /// Low-pass filter configuration (cutoff in (0, 0.5)).
    pub fn lowpass(cutoff: f32, q: f32) -> Self {
        let omega = 2.0 * PI * cutoff;
        let alpha = sinf(omega) / (2.0 * q);
        let cos_omega = cosf(omega);

        let b0 = (1.0 - cos_omega) * 0.5;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) * 0.5;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// High-pass filter configuration.
    pub fn highpass(cutoff: f32, q: f32) -> Self {
        let omega = 2.0 * PI * cutoff;
        let alpha = sinf(omega) / (2.0 * q);
        let cos_omega = cosf(omega);

        let b0 = (1.0 + cos_omega) * 0.5;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) * 0.5;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Band-pass filter configuration (constant 0 dB peak gain).
    pub fn bandpass(center: f32, q: f32) -> Self {
        let omega = 2.0 * PI * center;
        let alpha = sinf(omega) / (2.0 * q);
        let cos_omega = cosf(omega);

        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Notch filter configuration (e.g., 50/60 Hz powerline filtering).
    pub fn notch(center: f32, q: f32) -> Self {
        let omega = 2.0 * PI * center;
        let alpha = sinf(omega) / (2.0 * q);
        let cos_omega = cosf(omega);

        let b0 = 1.0;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Peaking EQ filter configuration.
    pub fn peaking(center: f32, q: f32, gain_db: f32) -> Self {
        let omega = 2.0 * PI * center;
        let alpha = sinf(omega) / (2.0 * q);
        let cos_omega = cosf(omega);
        let a_gain = powf(10.0, gain_db / 40.0);

        let b0 = 1.0 + alpha * a_gain;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0 - alpha * a_gain;
        let a0 = 1.0 + alpha / a_gain;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha / a_gain;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Low Shelf filter configuration.
    pub fn low_shelf(cutoff: f32, q: f32, gain_db: f32) -> Self {
        let omega = 2.0 * PI * cutoff;
        let alpha = sinf(omega) / (2.0 * q);
        let cos_omega = cosf(omega);
        let a_gain = powf(10.0, gain_db / 40.0);
        let two_sqrt_a_alpha = 2.0 * sqrtf(a_gain) * alpha;

        let b0 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_omega + two_sqrt_a_alpha);
        let b1 = 2.0 * a_gain * ((a_gain - 1.0) - (a_gain + 1.0) * cos_omega);
        let b2 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_omega - two_sqrt_a_alpha);
        let a0 = (a_gain + 1.0) + (a_gain - 1.0) * cos_omega + two_sqrt_a_alpha;
        let a1 = -2.0 * ((a_gain - 1.0) + (a_gain + 1.0) * cos_omega);
        let a2 = (a_gain + 1.0) + (a_gain - 1.0) * cos_omega - two_sqrt_a_alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// High Shelf filter configuration.
    pub fn high_shelf(cutoff: f32, q: f32, gain_db: f32) -> Self {
        let omega = 2.0 * PI * cutoff;
        let alpha = sinf(omega) / (2.0 * q);
        let cos_omega = cosf(omega);
        let a_gain = powf(10.0, gain_db / 40.0);
        let two_sqrt_a_alpha = 2.0 * sqrtf(a_gain) * alpha;

        let b0 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_omega + two_sqrt_a_alpha);
        let b1 = -2.0 * a_gain * ((a_gain - 1.0) + (a_gain + 1.0) * cos_omega);
        let b2 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_omega - two_sqrt_a_alpha);
        let a0 = (a_gain - 1.0) + (a_gain + 1.0) * cos_omega + two_sqrt_a_alpha;
        let a1 = 2.0 * ((a_gain - 1.0) - (a_gain + 1.0) * cos_omega);
        let a2 = (a_gain - 1.0) + (a_gain + 1.0) * cos_omega - two_sqrt_a_alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// All Pass filter configuration.
    pub fn allpass(cutoff: f32, q: f32) -> Self {
        let omega = 2.0 * PI * cutoff;
        let alpha = sinf(omega) / (2.0 * q);
        let cos_omega = cosf(omega);

        let b0 = 1.0 - alpha;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0 + alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    #[must_use]
    #[inline(always)]
    pub fn update(&mut self, x: f32) -> f32 {
        let y = (self.b0 * x) + (self.b1 * self.x1) + (self.b2 * self.x2)
              - (self.a1 * self.y1) - (self.a2 * self.y2);

        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;

        y
    }

    pub fn current(&self) -> f32 {
        self.y1
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

impl Default for BiquadFilter {
    fn default() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0, 0.0)
    }
}

// ============================================================================
// 5. 1D Kalman Filter
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct KalmanFilter1D {
    q: f32,
    r: f32,
    x: f32,
    p: f32,
    initialized: bool,
}

impl KalmanFilter1D {
    pub fn new(process_noise: f32, measurement_noise: f32) -> Self {
        debug_assert!(process_noise >= 0.0);
        debug_assert!(measurement_noise >= 0.0);
        Self {
            q: process_noise,
            r: measurement_noise,
            x: 0.0,
            p: 1.0,
            initialized: false,
        }
    }

    #[must_use]
    #[inline(always)]
    pub fn update(&mut self, z: f32) -> f32 {
        if !self.initialized {
            self.x = z;
            self.p = 1.0;
            self.initialized = true;
            return self.x;
        }

        let x_p = self.x;
        let p_p = self.p + self.q;

        let k = p_p / (p_p + self.r);
        self.x = x_p + k * (z - x_p);
        self.p = (1.0 - k) * p_p;

        self.x
    }

    pub fn current(&self) -> f32 {
        self.x
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.p = 1.0;
        self.initialized = false;
    }
}

impl Default for KalmanFilter1D {
    fn default() -> Self {
        Self::new(0.01, 0.1)
    }
}

// ============================================================================
// 6. DC Blocker (High-Pass IIR)
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct DcBlocker {
    r: f32,
    x_prev: f32,
    y_prev: f32,
}

impl DcBlocker {
    pub fn new(r: f32) -> Self {
        debug_assert!(r > 0.0 && r < 1.0, "DC Blocker R must be in (0, 1)");
        Self { r, x_prev: 0.0, y_prev: 0.0 }
    }

    #[must_use]
    #[inline(always)]
    pub fn update(&mut self, x: f32) -> f32 {
        let y = x - self.x_prev + self.r * self.y_prev;
        self.x_prev = x;
        self.y_prev = y;
        y
    }

    pub fn current(&self) -> f32 {
        self.y_prev
    }

    pub fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }
}

impl Default for DcBlocker {
    fn default() -> Self {
        Self::new(0.99)
    }
}

// ============================================================================
// 7. One Euro Filter (Adaptive Smoothing)
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct OneEuroFilter {
    freq: f32,
    mincutoff: f32,
    beta: f32,
    dcutoff: f32,
    x_prev: f32,
    dx_prev: f32,
    initialized: bool,
}

impl OneEuroFilter {
    pub fn new(freq: f32, mincutoff: f32, beta: f32, dcutoff: f32) -> Self {
        Self {
            freq,
            mincutoff,
            beta,
            dcutoff,
            x_prev: 0.0,
            dx_prev: 0.0,
            initialized: false,
        }
    }

    #[must_use]
    #[inline(always)]
    pub fn update(&mut self, x: f32) -> f32 {
        if !self.initialized {
            self.x_prev = x;
            self.dx_prev = 0.0;
            self.initialized = true;
            return x;
        }

        let dx = (x - self.x_prev) * self.freq;
        let alpha_d = self.smoothing_factor(self.dcutoff);
        let dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev;

        let cutoff = self.mincutoff + self.beta * dx_hat.abs();
        let alpha = self.smoothing_factor(cutoff);
        let x_hat = alpha * x + (1.0 - alpha) * self.x_prev;

        self.x_prev = x_hat;
        self.dx_prev = dx_hat;
        x_hat
    }

    #[inline(always)]
    fn smoothing_factor(&self, cutoff: f32) -> f32 {
        let tau = 1.0 / (2.0 * PI * cutoff);
        1.0 / (1.0 + tau * self.freq)
    }

    pub fn current(&self) -> f32 {
        self.x_prev
    }

    pub fn reset(&mut self) {
        self.initialized = false;
        self.x_prev = 0.0;
        self.dx_prev = 0.0;
    }
}

impl Default for OneEuroFilter {
    fn default() -> Self {
        Self::new(50.0, 1.0, 0.007, 1.0)
    }
}

// ============================================================================
// 8. Alpha-Beta Tracker
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct AlphaBetaTracker {
    alpha: f32,
    beta: f32,
    dt: f32,
    x: f32,
    v: f32,
    initialized: bool,
}

impl AlphaBetaTracker {
    pub fn new(alpha: f32, beta: f32, dt: f32) -> Self {
        Self { alpha, beta, dt, x: 0.0, v: 0.0, initialized: false }
    }

    #[must_use]
    #[inline(always)]
    pub fn update(&mut self, z: f32) -> f32 {
        if !self.initialized {
            self.x = z;
            self.v = 0.0;
            self.initialized = true;
            return self.x;
        }

        let x_p = self.x + self.v * self.dt;
        let v_p = self.v;
        let residual = z - x_p;

        self.x = x_p + self.alpha * residual;
        self.v = v_p + (self.beta / self.dt) * residual;

        self.x
    }

    pub fn current(&self) -> f32 {
        self.x
    }

    pub fn velocity(&self) -> f32 {
        self.v
    }

    pub fn reset(&mut self) {
        self.initialized = false;
        self.x = 0.0;
        self.v = 0.0;
    }
}

impl Default for AlphaBetaTracker {
    fn default() -> Self {
        Self::new(0.2, 0.1, 0.02)
    }
}

// ============================================================================
// 9. Running Variance & Standard Deviation Filter
// ============================================================================

#[derive(Debug, Clone)]
pub struct RunningVarianceFilter<const N: usize> {
    buffer: [f32; N],
    index: usize,
    count: usize,
    sum: f32,
    sum_sq: f32,
}

impl<const N: usize> RunningVarianceFilter<N> {
    pub fn new() -> Self {
        debug_assert!(N > 1, "Variance window N must be > 1");
        Self { buffer: [0.0; N], index: 0, count: 0, sum: 0.0, sum_sq: 0.0 }
    }

    #[must_use]
    #[inline(always)]
    pub fn update(&mut self, x: f32) -> f32 {
        if self.count == N {
            let old = self.buffer[self.index];
            self.sum -= old;
            self.sum_sq -= old * old;
        } else {
            self.count += 1;
        }
        self.buffer[self.index] = x;
        self.sum += x;
        self.sum_sq += x * x;
        self.index = (self.index + 1) % N;

        self.variance()
    }

    pub fn variance(&self) -> f32 {
        if self.count <= 1 {
            return 0.0;
        }
        let n = self.count as f32;
        let mean = self.sum / n;
        let var = (self.sum_sq - self.sum * mean) / (n - 1.0);
        var.max(0.0)
    }

    pub fn standard_deviation(&self) -> f32 {
        sqrtf(self.variance())
    }

    pub fn current(&self) -> f32 {
        if self.count == 0 { 0.0 } else { self.buffer[(self.index + N - 1) % N] }
    }

    pub fn reset(&mut self) {
        self.buffer = [0.0; N];
        self.index = 0;
        self.count = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }
}

impl<const N: usize> Default for RunningVarianceFilter<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 10. Peak Hold Filter
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct PeakHoldFilter {
    peak: f32,
    decay_rate: f32,
}

impl PeakHoldFilter {
    pub fn new(decay_rate: f32) -> Self {
        debug_assert!(decay_rate > 0.0 && decay_rate <= 1.0, "decay_rate must be in (0, 1]");
        Self { peak: 0.0, decay_rate }
    }

    #[must_use]
    #[inline(always)]
    pub fn update(&mut self, x: f32) -> f32 {
        self.peak *= self.decay_rate;
        if x > self.peak {
            self.peak = x;
        }
        self.peak
    }

    pub fn current(&self) -> f32 {
        self.peak
    }

    pub fn reset(&mut self) {
        self.peak = 0.0;
    }
}

impl Default for PeakHoldFilter {
    fn default() -> Self {
        Self::new(0.99)
    }
}

// ============================================================================
// 11. Root Mean Square (RMS) Filter
// ============================================================================

#[derive(Debug, Clone)]
pub struct RmsFilter<const N: usize> {
    buffer: [f32; N],
    index: usize,
    count: usize,
    sum_sq: f32,
}

impl<const N: usize> RmsFilter<N> {
    pub fn new() -> Self {
        debug_assert!(N > 0, "RMS filter N must be > 0");
        Self { buffer: [0.0; N], index: 0, count: 0, sum_sq: 0.0 }
    }

    #[must_use]
    #[inline(always)]
    pub fn update(&mut self, x: f32) -> f32 {
        if self.count == N {
            let old = self.buffer[self.index];
            self.sum_sq -= old * old;
        } else {
            self.count += 1;
        }
        self.buffer[self.index] = x;
        self.sum_sq += x * x;
        self.index = (self.index + 1) % N;

        let mean_sq = self.sum_sq / (self.count as f32);
        sqrtf(mean_sq.max(0.0))
    }

    pub fn current(&self) -> f32 {
        if self.count == 0 { 0.0 } else { self.buffer[(self.index + N - 1) % N] }
    }

    pub fn reset(&mut self) {
        self.buffer = [0.0; N];
        self.index = 0;
        self.count = 0;
        self.sum_sq = 0.0;
    }
}

impl<const N: usize> Default for RmsFilter<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 12. Sliding Min / Max Filter
// ============================================================================

#[derive(Debug, Clone)]
pub struct SlidingMinMaxFilter<const N: usize> {
    buffer: [f32; N],
    index: usize,
    count: usize,
}

impl<const N: usize> SlidingMinMaxFilter<N> {
    pub fn new() -> Self {
        debug_assert!(N > 0, "Window N must be > 0");
        Self { buffer: [0.0; N], index: 0, count: 0 }
    }

    #[must_use]
    #[inline(always)]
    pub fn update(&mut self, x: f32) -> (f32, f32) {
        self.buffer[self.index] = x;
        self.index = (self.index + 1) % N;
        if self.count < N {
            self.count += 1;
        }

        let active = &self.buffer[..self.count];
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        for &val in active {
            if val < min_val { min_val = val; }
            if val > max_val { max_val = val; }
        }

        (min_val, max_val)
    }

    pub fn reset(&mut self) {
        self.buffer = [0.0; N];
        self.index = 0;
        self.count = 0;
    }
}

impl<const N: usize> Default for SlidingMinMaxFilter<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Math Utilities (ESP-IDF / Portable #![no_std] wrappers)
// ============================================================================

#[inline(always)]
fn sinf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    { x.sin() }
    #[cfg(not(feature = "std"))]
    {
        extern "C" { fn sinf(x: f32) -> f32; }
        unsafe { sinf(x) }
    }
}

#[inline(always)]
fn cosf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    { x.cos() }
    #[cfg(not(feature = "std"))]
    {
        extern "C" { fn cosf(x: f32) -> f32; }
        unsafe { cosf(x) }
    }
}

#[inline(always)]
fn sqrtf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    { x.sqrt() }
    #[cfg(not(feature = "std"))]
    {
        extern "C" { fn sqrtf(x: f32) -> f32; }
        unsafe { sqrtf(x) }
    }
}

#[inline(always)]
fn powf(base: f32, exp: f32) -> f32 {
    #[cfg(feature = "std")]
    { base.powf(exp) }
    #[cfg(not(feature = "std"))]
    {
        extern "C" { fn powf(base: f32, exp: f32) -> f32; }
        unsafe { powf(base, exp) }
    }
}
