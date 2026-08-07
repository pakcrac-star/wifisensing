//! ============================================================================
//! Wi-Fi CSI Sensing Engine (`lib.rs`)
//! ============================================================================

#![cfg_attr(not(feature = "std"), no_std)]

// ============================================================================
// Float Extensions for `no_std` using `libm`
// ============================================================================

pub trait FloatExt {
    fn sqrt(self) -> Self;
    fn log10(self) -> Self;
    fn cos(self) -> Self;
    fn sin(self) -> Self;
}

impl FloatExt for f32 {
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn log10(self) -> Self {
        libm::log10f(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cosf(self)
    }
    #[inline]
    fn sin(self) -> Self {
        libm::sinf(self)
    }
}

// ============================================================================
// Module Declarations
// ============================================================================

pub mod acquisition;
pub mod amplitude;
pub mod classification;
pub mod doppler;
pub mod entropy;
pub mod filter;
pub mod fir;
pub mod fusion;
pub mod interference;
pub mod model;
pub mod phase;
pub mod recovery;
pub mod ringbuffer;
pub mod signature_handler;
pub mod types;
pub mod vitals;

// ============================================================================
// Re-exports
// ============================================================================

pub use types::{
    BiometricFeatures, CalibratedChannel, ChannelQuality, ComplexF32,
    FinalPrediction, InferenceTensor, InformationMetrics, KinematicFeatures,
    PhysicalState, ParsedCsiFrame, CsiFrame, MetadataFrame, UnifiedScene, NUM_SUBCARRIERS, MODEL_FEATURES,
};

pub use recovery::NumAnalyzer;
pub use ringbuffer::RingBuffer;
pub use signature_handler::{SignatureHandler, SignatureScores, HistoricalSnapshot, PersonalBaseline};
pub use vitals::{BiquadFilter, extract_vitals, find_autocorrelation_peak};

// ============================================================================
// Panic Handler for `no_std`
// ============================================================================

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

// ============================================================================
// C-FFI Interfaces (Resolving Linker Errors & Matching main.c)
// ============================================================================

/// Called by C `app_main` to initialize the Rust DSP engine.
#[no_mangle]
pub extern "C" fn rust_engine_init() {
    // Initialize static buffers, filters, or tracking structures here
}

/// Called by C `data_router_task` when a high-speed CSI frame arrives.
#[no_mangle]
pub extern "C" fn rust_engine_push_csi(frame_ptr: *const CsiFrame) {
    if frame_ptr.is_null() {
        return;
    }
    
    // Safety: The pointer comes directly from main.c matching `CsiFrame` layout in types.rs
    let _raw_frame = unsafe { *frame_ptr };
    
    // Hook into your pipeline processing here if desired
}

/// Called by C `data_router_task` when slow-path metadata arrives.
#[no_mangle]
pub extern "C" fn rust_engine_push_metadata(meta_ptr: *const MetadataFrame) {
    if meta_ptr.is_null() {
        return;
    }
    
    // Safety: The pointer comes directly from main.c matching `MetadataFrame` layout in types.rs
    let _metadata = unsafe { *meta_ptr };
    
    // Hook into your metadata processing here if desired
}
