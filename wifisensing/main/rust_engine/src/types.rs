//! ============================================================================
//! Module: Shared Data Types (`types.rs`)
//! ============================================================================
//!
//! Purpose
//! -------
//! Shared data definitions for the Wi-Fi CSI sensing pipeline.
//! This module contains ONLY data structures. No algorithms, filters, DSP, 
//! ML, or mutable global state.
//!
//! Design Guarantees
//! -----------------
//! • Fully compatible with `#![no_std]`.
//! • Manual `Default` implementations for large fixed-size arrays.
//! ============================================================================

/// Number of CSI subcarriers (standard 20MHz channel).
pub const NUM_SUBCARRIERS: usize = 64;

/// ESP32 hardware buffer limit for CSI byte data.
pub const CSI_MAX_SAMPLES: usize = 384;

/// Number of neural-network input features.
pub const MODEL_FEATURES: usize = 16;


// ============================================================================
// Basic Math Types
// ============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ComplexF32 {
    pub re: f32,
    pub im: f32,
}


// ============================================================================
// Hardware FFI Bindings (C -> Rust)
// ============================================================================
// These structs must exactly match the memory layout of `csi_frame_t` 
// and `metadata_frame_t` in your `wifi_csi.c` file.

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CsiFrame {
    pub timestamp_us: u64,
    pub mac_address: [u8; 6],
    pub rssi: i8,
    pub noise_floor: i8,
    pub channel: u8,
    pub bandwidth: u8,
    pub phy_mode: u8,
    pub csi_length: u16,
    pub csi_data: [u8; CSI_MAX_SAMPLES],
}

impl Default for CsiFrame {
    fn default() -> Self {
        Self {
            timestamp_us: 0,
            mac_address: [0; 6],
            rssi: 0,
            noise_floor: 0,
            channel: 0,
            bandwidth: 0,
            phy_mode: 0,
            csi_length: 0,
            csi_data: [0; CSI_MAX_SAMPLES],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MetadataFrame {
    pub timestamp_us: u64,
    pub bssid: [u8; 6],
    pub ssid: [u8; 33],
    pub rssi: i8,
    pub channel: u8,
    pub security_type: u8,
    pub phy_mode: u8,
    pub bandwidth: u8,
    pub noise: i8,
    pub beacon_interval: u16,
}

impl Default for MetadataFrame {
    fn default() -> Self {
        Self {
            timestamp_us: 0,
            bssid: [0; 6],
            ssid: [0; 33],
            rssi: 0,
            channel: 0,
            security_type: 0,
            phy_mode: 0,
            bandwidth: 0,
            noise: 0,
            beacon_interval: 0,
        }
    }
}

extern "C" {
    /// Pops a hardware CSI frame from the C ringbuffer. Blocks up to timeout_ms.
    pub fn wifi_csi_pop_frame(out_frame: *mut CsiFrame, timeout_ms: u32) -> bool;
    
    /// Pops a metadata frame from the C queue. Blocks up to timeout_ms.
    pub fn wifi_csi_pop_metadata(out_meta: *mut MetadataFrame, timeout_ms: u32) -> bool;
}


// ============================================================================
// Signal Processing Types (Rust Pipeline)
// ============================================================================

/// Parsed CSI subcarriers extracted from the raw hardware byte array.
/// This is the first step in the DSP pipeline (handled by `acquisition.rs`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ParsedCsiFrame {
    pub timestamp_us: u64,
    pub subcarriers: [ComplexF32; NUM_SUBCARRIERS],
    pub rssi: i8,
    pub noise_floor: i8,
}

impl Default for ParsedCsiFrame {
    fn default() -> Self {
        Self {
            timestamp_us: 0,
            subcarriers: [ComplexF32 { re: 0.0, im: 0.0 }; NUM_SUBCARRIERS],
            rssi: 0,
            noise_floor: 0,
        }
    }
}

/// Calibrated electromagnetic channel after phase sanitization (CFO/SFO correction)
/// and amplitude filtering. Handled by `phase.rs` and `amplitude.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CalibratedChannel {
    pub timestamp_us: u64,
    pub true_amplitude: [f32; NUM_SUBCARRIERS],
    pub true_phase: [f32; NUM_SUBCARRIERS],
}

impl Default for CalibratedChannel {
    fn default() -> Self {
        Self {
            timestamp_us: 0,
            true_amplitude: [0.0; NUM_SUBCARRIERS],
            true_phase: [0.0; NUM_SUBCARRIERS],
        }
    }
}


// ============================================================================
// Feature Extraction Output
// ============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ChannelQuality {
    pub signal_to_noise_ratio_db: f32,
    pub rssi_variance: f32,
    pub burst_packet_loss: u16,
    pub subcarrier_anomalies: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct KinematicFeatures {
    pub peak_doppler_shift_hz: f32,
    pub estimated_velocity_m_s: f32,
    pub velocity_variance: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct BiometricFeatures {
    pub breathing_rate_hz: f32,
    pub heartbeat_rate_hz: f32,
    pub chest_displacement_mm: f32,
    pub signal_confidence: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct InformationMetrics {
    pub spatial_entropy: f32,
    pub temporal_entropy: f32,
    pub spectral_entropy: f32,
}


// ============================================================================
// Sensor Fusion & AI
// ============================================================================

/// Complete physical representation of the observed environment before AI mapping.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct UnifiedScene {
    pub timestamp_us: u64,
    pub channel_quality: ChannelQuality,
    pub kinematics: KinematicFeatures,
    pub biometrics: BiometricFeatures,
    pub entropy: InformationMetrics,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TemporalSignature {
    pub baseline_entropy: f32,
    pub baseline_breathing_hz: f32,
    pub circadian_variance: f32,
}

/// Feature vector presented to the TinyML/classification model.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InferenceTensor {
    pub features: [f32; MODEL_FEATURES],
    pub feature_count: u8,
}

impl Default for InferenceTensor {
    fn default() -> Self {
        Self {
            features: [0.0; MODEL_FEATURES],
            feature_count: MODEL_FEATURES as u8,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhysicalState {
    #[default]
    StaticEmpty = 0,
    StaticOccupied = 1,
    Walking = 2,
    Falling = 3,
    Sleeping = 4,
    HardwareFault = 255,
}

/// Final output returned from the Rust sensing engine.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FinalPrediction {
    pub timestamp_us: u64,
    pub confidence_score: f32,
    pub velocity_m_s: f32,
    pub breathing_rate_hz: f32,
    pub heartbeat_rate_hz: f32,
    pub state: PhysicalState,
}
