
//! ============================================================================
//! Module 14: Behavioral Physics & Long-Term Adaptation (`signature_handler.rs`)
//! ============================================================================
//!
//! Purpose
//! -------
//! Maintains long-term behavioral baselines and compares current physical
//! observations against historical patterns. This module does NOT perform
//! activity classification. Instead, it produces quantitative similarity
//! metrics that higher-level fusion and classification modules may consume.
//!
//! Mathematical Model
//! ------------------
//! 1. Exponential Moving Average (EMA)
//!      μ[t] = (1-β)μ[t-1] + βx[t]
//!
//! 2. Circadian Adaptation
//!      baseline(hour) ← slowly adapted activity expectation
//!
//! 3. Behavioral Signature Matching
//!      • Sleep
//!      • Walking
//!      • Fall anomaly
//!
//! Invariants
//! ----------
//! • Never accesses CSI samples.
//! • Never performs classification.
//! • Pure historical behavioral analysis.
//! • Fully compatible with #![no_std].
//! ============================================================================



/// Number of hours represented by the circadian profile.
pub const CIRCADIAN_HOURS: usize = 24;

/// Long-term EMA adaptation coefficient.
const BASELINE_ADAPTATION_RATE: f32 = 0.0005;

/// Circadian profile learning rate.
const CIRCADIAN_UPDATE_RATE: f32 = 0.01;

/// Default baseline values.
const DEFAULT_BASELINE_ENERGY: f32 = 10.0;
const DEFAULT_BASELINE_VELOCITY: f32 = 0.0;
const DEFAULT_BASELINE_VITALS: f32 = 1.2;

/// Default circadian expectations.
const NIGHT_ACTIVITY: f32 = 0.3;
const DAY_ACTIVITY: f32 = 1.2;

/// Physiological limits.
const MIN_VITAL_FREQ: f32 = 0.5;
const MAX_VITAL_FREQ: f32 = 3.0;

/// Fall detection normalization.
const FALL_THRESHOLD: f32 = 4.0;
const FALL_NORMALIZATION: f32 = 6.0;

/// Numerical stability.
const EPSILON: f32 = 1e-6;

// ============================================================================
// Signature Scores
// ============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignatureScores {
    pub sleep_signature_match: f32,
    pub walking_signature_match: f32,
    pub fall_anomaly_score: f32,
    pub baseline_deviation: f32,
    pub circadian_baseline_scalar: f32,
}

impl Default for SignatureScores {
    fn default() -> Self {
        Self {
            sleep_signature_match: 0.0,
            walking_signature_match: 0.0,
            fall_anomaly_score: 0.0,
            baseline_deviation: 0.0,
            circadian_baseline_scalar: 1.0,
        }
    }
}

impl SignatureScores {
    #[inline]
    pub fn is_sleeping(&self) -> bool {
        self.sleep_signature_match > 0.80
    }

    #[inline]
    pub fn possible_fall(&self) -> bool {
        self.fall_anomaly_score > 0.80
    }
}

// ============================================================================
// Historical Snapshot
// ============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HistoricalSnapshot {
    pub mean_energy: f32,
    pub energy_variance: f32,
    pub mean_velocity: f32,
    pub velocity_variance: f32,
    pub mean_vitals_hz: f32,
    pub peak_transient_energy: f32,
}

// ============================================================================
// Personal Baseline
// ============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PersonalBaseline {
    pub avg_energy: f32,
    pub avg_velocity: f32,
    pub avg_vitals: f32,
    circadian_profile: [f32; CIRCADIAN_HOURS],
    pub initialized: bool, // Changed to pub so struct can be read cleanly if exported
}

impl Default for PersonalBaseline {
    fn default() -> Self {
        Self::new()
    }
}

impl PersonalBaseline {
    pub fn new() -> Self {
        let mut profile = [DAY_ACTIVITY; CIRCADIAN_HOURS];

        // Zero-cost iterator mapping, much safer and faster than `while` loops
        for (hour, val) in profile.iter_mut().enumerate() {
            if hour >= 22 || hour <= 6 {
                *val = NIGHT_ACTIVITY;
            }
        }

        Self {
            avg_energy: DEFAULT_BASELINE_ENERGY,
            avg_velocity: DEFAULT_BASELINE_VELOCITY,
            avg_vitals: DEFAULT_BASELINE_VITALS,
            circadian_profile: profile,
            initialized: false,
        }
    }

    pub fn update(&mut self, snapshot: &HistoricalSnapshot, current_hour: u8) {
        if !self.initialized {
            self.avg_energy = snapshot.mean_energy;
            self.avg_velocity = snapshot.mean_velocity;
            self.avg_vitals = snapshot.mean_vitals_hz;
            self.initialized = true;
            return;
        }

        self.avg_energy = (1.0 - BASELINE_ADAPTATION_RATE) * self.avg_energy
            + BASELINE_ADAPTATION_RATE * snapshot.mean_energy;

        self.avg_velocity = (1.0 - BASELINE_ADAPTATION_RATE) * self.avg_velocity
            + BASELINE_ADAPTATION_RATE * snapshot.mean_velocity;

        self.avg_vitals = (1.0 - BASELINE_ADAPTATION_RATE) * self.avg_vitals
            + BASELINE_ADAPTATION_RATE * snapshot.mean_vitals_hz;

        let index = (current_hour as usize) % CIRCADIAN_HOURS;

        self.circadian_profile[index] = (1.0 - CIRCADIAN_UPDATE_RATE)
            * self.circadian_profile[index]
            + CIRCADIAN_UPDATE_RATE * (snapshot.mean_energy / (self.avg_energy + EPSILON));
    }

    #[inline]
    pub fn circadian_scalar(&self, current_hour: u8) -> f32 {
        self.circadian_profile[(current_hour as usize) % CIRCADIAN_HOURS]
    }
}

// ============================================================================
// Signature Handler
// ============================================================================

pub struct SignatureHandler {
    baseline: PersonalBaseline,
}

impl Default for SignatureHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl SignatureHandler {
    pub fn new() -> Self {
        Self {
            baseline: PersonalBaseline::new(),
        }
    }

    pub fn evaluate_signatures(
        &mut self,
        snapshot: &HistoricalSnapshot,
        current_hour: u8,
    ) -> SignatureScores {
        self.baseline.update(snapshot, current_hour);

        let circadian_scalar = self.baseline.circadian_scalar(current_hour);
        let expected_energy = self.baseline.avg_energy * circadian_scalar;
        let expected_energy_safe = expected_energy + EPSILON;

        // ------------------------------------------------------------
        // Sleep Signature
        // ------------------------------------------------------------
        let energy_sleep_raw = 1.0 - (snapshot.mean_energy / (self.baseline.avg_energy * 0.5 + EPSILON));
        let energy_sleep_fit = safe_clamp(energy_sleep_raw, 0.0, 1.0);

        let velocity_sleep_raw = 1.0 - snapshot.mean_velocity;
        let velocity_sleep_fit = safe_clamp(velocity_sleep_raw, 0.0, 1.0);

        let vitals_fit = if snapshot.mean_vitals_hz >= MIN_VITAL_FREQ
            && snapshot.mean_vitals_hz <= MAX_VITAL_FREQ
        {
            1.0
        } else {
            0.2
        };

        let sleep_score = energy_sleep_fit * 0.4 + velocity_sleep_fit * 0.4 + vitals_fit * 0.2;

        // ------------------------------------------------------------
        // Walking Signature
        // ------------------------------------------------------------
        let velocity_ratio = if snapshot.mean_velocity > 0.1 {
            safe_clamp(snapshot.mean_velocity / (snapshot.velocity_variance + 0.1), 0.0, 1.0)
        } else {
            0.0
        };

        let energy_walk_raw = snapshot.mean_energy / (self.baseline.avg_energy * 2.0 + EPSILON);
        let energy_walk_fit = safe_clamp(energy_walk_raw, 0.0, 1.0);

        let walking_score = velocity_ratio * 0.6 + energy_walk_fit * 0.4;

        // ------------------------------------------------------------
        // Fall Signature
        // ------------------------------------------------------------
        let peak_ratio = snapshot.peak_transient_energy / expected_energy_safe;

        let fall_score = if peak_ratio > FALL_THRESHOLD {
            safe_clamp((peak_ratio - FALL_THRESHOLD) / FALL_NORMALIZATION, 0.0, 1.0)
        } else {
            0.0
        };

        // ------------------------------------------------------------
        // Baseline Deviation
        // ------------------------------------------------------------
        let deviation = abs(snapshot.mean_energy - expected_energy) / expected_energy_safe;

        SignatureScores {
            sleep_signature_match: safe_clamp(sleep_score, 0.0, 1.0),
            walking_signature_match: safe_clamp(walking_score, 0.0, 1.0),
            fall_anomaly_score: fall_score,
            baseline_deviation: deviation,
            circadian_baseline_scalar: circadian_scalar,
        }
    }

    #[inline]
    pub fn baseline(&self) -> &PersonalBaseline {
        &self.baseline
    }
}

// ============================================================================
// Math Utilities
// ============================================================================

/// Panic-safe clamp for embedded environments.
/// Standard `f32::clamp` will panic if given a NaN. This will safely fall back to `min`.
#[inline]
fn safe_clamp(val: f32, min: f32, max: f32) -> f32 {
    if val < min || val.is_nan() {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

/// Fast absolute value via bitmasking.
/// Single-cycle instruction on Xtensa/RISC-V, much faster than `libm::fabsf`.
#[inline(always)]
fn abs(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}
