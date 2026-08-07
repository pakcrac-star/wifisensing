// wifisensing/main/data_transfer.rs
// 
// Comprehensive ESP32 ↔ Laptop data transport layer
// - UDP/TCP CSI frame ingestion from firmware
// - Multi-node synchronization (node_id, sequence_number)
// - Timestamp alignment (TDM slot scheduling)
// - Vital signs extraction (breathing, heart rate)
// - Multi-node fusion (N×(N-1) multistatic links)
// - Error recovery + dropped frame detection
// - Home Assistant MQTT bridge
//
// Built for RuView-compatible WiFi CSI sensing

use std::collections::{HashMap, VecDeque};
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use tokio::task::JoinHandle;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// =====================================================
// CONSTANTS & CONFIGURATION
// =====================================================

const CSI_SUBCARRIERS: usize = 128;
const FEATURE_VECTOR_SIZE: usize = 64;
const WINDOW_SIZE: usize = 8;
const CSI_BUFFER_SIZE: usize = 1024;
const MAX_NODES: usize = 6;
const UDP_BUFFER_SIZE: usize = 2048;
const FRAME_TIMEOUT_MS: u64 = 1000;

// Vital signs filter design (100 Hz sample rate)
const BREATHING_FREQ_MIN: f64 = 0.1;  // 6 BPM
const BREATHING_FREQ_MAX: f64 = 0.5;  // 30 BPM
const HEART_FREQ_MIN: f64 = 0.8;      // 48 BPM
const HEART_FREQ_MAX: f64 = 2.0;      // 120 BPM
const SAMPLE_RATE_HZ: f64 = 100.0;

// TDM slot scheduling (ADR-029)
const TDM_SLOT_DURATION_MS: u32 = 5;
const TDM_TOTAL_CYCLE_MS: u32 = TDM_SLOT_DURATION_MS * MAX_NODES as u32;

// =====================================================
// ERROR TYPES
// =====================================================

#[derive(Error, Debug)]
pub enum DataTransferError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    
    #[error("Frame validation failed: {0}")]
    FrameValidationFailed(String),
    
    #[error("Timestamp out of sync: expected {expected}, got {actual}")]
    TimestampSync { expected: u32, actual: u32 },
    
    #[error("Node {node_id} offline: no frames for {duration_ms}ms")]
    NodeOffline { node_id: u8, duration_ms: u64 },
    
    #[error("Fusion error: {0}")]
    FusionError(String),
    
    #[error("Vital signs extraction failed: {0}")]
    VitalSignsError(String),
}

pub type Result<T> = std::result::Result<T, DataTransferError>;

// =====================================================
// DATA STRUCTURES (WIRE FORMAT)
// =====================================================

/// Raw CSI frame from ESP32 firmware (matches main/model.h CSIFrame)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawCSIFrame {
    pub rssi: i8,
    pub channel: u8,
    pub node_id: u8,           // NEW: identifies which ESP32 sent this
    pub timestamp_ms: u32,     // milliseconds since ESP32 boot
    pub sequence_number: u32,  // frame counter for detecting gaps
    pub amplitude: Vec<i16>,   // [CSI_SUBCARRIERS]
    pub phase: Vec<i16>,       // [CSI_SUBCARRIERS]
}

/// Processed CSI frame with extracted features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedCSIFrame {
    pub node_id: u8,
    pub timestamp_ms: u32,
    pub sequence_number: u32,
    pub rssi: i8,
    pub channel: u8,
    pub amplitude_mean: f64,
    pub amplitude_var: f64,
    pub phase_mean: f64,
    pub phase_stability: f64,
    pub motion_energy: f64,
    pub entropy: f64,
    pub band_low: f64,
    pub band_high: f64,
    pub snr_estimate: f64,
    pub feature_vector: Vec<f64>,  // [FEATURE_VECTOR_SIZE]
    pub server_receive_time_ms: u64,  // Wall-clock time received
}

/// Vital signs extracted from bandpass-filtered phase data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VitalSigns {
    pub breathing_bpm: f64,
    pub breathing_confidence: f64,
    pub heart_bpm: f64,
    pub heart_confidence: f64,
    pub timestamp_ms: u64,
}

/// ML inference result from edge model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub motion_detected: bool,
    pub person_present: bool,
    pub confidence: f64,
    pub motion_score: f64,
    pub vitals: VitalSigns,
    pub timestamp_ms: u64,
}

/// Multi-node CSI fusion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionResult {
    pub fusion_timestamp_ms: u64,
    pub active_node_count: usize,
    pub total_links: usize,  // N×(N-1) directed links
    pub combined_confidence: f64,
    pub person_present: bool,
    pub motion_detected: bool,
    pub motion_score: f64,
    pub vitals: VitalSigns,
    pub per_node_results: Vec<PerNodeResult>,
    pub spatial_entropy: f64,  // Measure of signal diversity
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerNodeResult {
    pub node_id: u8,
    pub person_present: bool,
    pub confidence: f64,
    pub motion_score: f64,
}

// =====================================================
// BANDPASS FILTER (IIR, Butterworth-inspired)
// =====================================================

/// Simple 5-tap IIR bandpass filter
/// Designed for low-latency edge computation
pub struct BandpassFilter {
    // Normalized IIR coefficients (bilinear transform)
    b: [f64; 3],  // numerator (feed-forward)
    a: [f64; 3],  // denominator (feedback)
    
    // State: previous samples
    x_hist: VecDeque<f64>,
    y_hist: VecDeque<f64>,
}

impl BandpassFilter {
    /// Create a bandpass filter with given center frequency band
    /// 
    /// # Arguments
    /// * `f_min` - Lower cutoff frequency (Hz)
    /// * `f_max` - Upper cutoff frequency (Hz)
    /// * `sample_rate` - Sample rate (Hz)
    pub fn new(f_min: f64, f_max: f64, sample_rate: f64) -> Self {
        let omega_min = 2.0 * std::f64::consts::PI * f_min / sample_rate;
        let omega_max = 2.0 * std::f64::consts::PI * f_max / sample_rate;
        let omega_center = (omega_min + omega_max) / 2.0;
        let bandwidth = omega_max - omega_min;
        
        // Bilinear transformation coefficients
        // Simplified Butterworth response
        let a0 = 1.0 + (bandwidth / 2.0);
        let b0 = (bandwidth / 2.0);
        
        let b = [b0 / a0, 0.0, -b0 / a0];
        let a = [1.0, -2.0 * omega_center.cos() / a0, (1.0 - bandwidth / 2.0) / a0];
        
        Self {
            b,
            a,
            x_hist: VecDeque::with_capacity(3),
            y_hist: VecDeque::with_capacity(3),
        }
    }
    
    /// Apply filter to a single sample
    pub fn apply_sample(&mut self, x: f64) -> f64 {
        self.x_hist.push_back(x);
        if self.x_hist.len() > 3 {
            self.x_hist.pop_front();
        }
        
        let mut y = 0.0;
        
        // Feed-forward
        for i in 0..self.x_hist.len().min(3) {
            if let Some(&x_val) = self.x_hist.get(i) {
                y += self.b[i] * x_val;
            }
        }
        
        // Feedback
        for i in 0..self.y_hist.len().min(3) {
            if let Some(&y_val) = self.y_hist.get(i) {
                y -= self.a[i + 1] * y_val;
            }
        }
        
        y /= self.a[0];
        
        self.y_hist.push_back(y);
        if self.y_hist.len() > 3 {
            self.y_hist.pop_front();
        }
        
        y
    }
    
    /// Apply filter to entire signal
    pub fn apply(&mut self, signal: &[f64]) -> Vec<f64> {
        signal.iter().map(|&x| self.apply_sample(x)).collect()
    }
    
    /// Reset filter state
    pub fn reset(&mut self) {
        self.x_hist.clear();
        self.y_hist.clear();
    }
}

// =====================================================
// VITAL SIGNS EXTRACTOR
// =====================================================

pub struct VitalSignsExtractor {
    breathing_filter: BandpassFilter,
    heart_filter: BandpassFilter,
    
    // Temporal history for valid estimates
    breathing_history: VecDeque<f64>,
    heart_history: VecDeque<f64>,
    
    // Configuration
    sample_rate: f64,
    min_confidence_threshold: f64,
    min_valid_bpm: f64,
    max_valid_bpm: f64,
}

impl VitalSignsExtractor {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            breathing_filter: BandpassFilter::new(
                BREATHING_FREQ_MIN,
                BREATHING_FREQ_MAX,
                sample_rate,
            ),
            heart_filter: BandpassFilter::new(
                HEART_FREQ_MIN,
                HEART_FREQ_MAX,
                sample_rate,
            ),
            breathing_history: VecDeque::with_capacity(300),  // 3s @ 100Hz
            heart_history: VecDeque::with_capacity(300),
            sample_rate,
            min_confidence_threshold: 0.3,
            min_valid_bpm: 6.0,
            max_valid_bpm: 180.0,
        }
    }
    
    /// Extract vital signs from unwrapped phase signal
    /// 
    /// # Arguments
    /// * `phase_signal` - Unwrapped phase (rad) vs time
    /// 
    /// # Returns
    /// Breathing + heart rate estimates with confidence scores
    pub fn extract(&mut self, phase_signal: &[f64]) -> VitalSigns {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        // Apply bandpass filters
        let breathing_band = self.breathing_filter.apply(phase_signal);
        let heart_band = self.heart_filter.apply(phase_signal);
        
        // Extract BPM via zero-crossing detection
        let breathing_bpm = self.extract_bpm_from_band(&breathing_band);
        let heart_bpm = self.extract_bpm_from_band(&heart_band);
        
        // Compute confidence scores
        let breathing_confidence = self.compute_confidence(&breathing_band, breathing_bpm);
        let heart_confidence = self.compute_confidence(&heart_band, heart_bpm);
        
        // Validate ranges
        let breathing_bpm = if breathing_bpm >= self.min_valid_bpm
            && breathing_bpm <= self.max_valid_bpm
            && breathing_confidence > self.min_confidence_threshold
        {
            breathing_bpm
        } else {
            0.0
        };
        
        let heart_bpm = if heart_bpm >= self.min_valid_bpm
            && heart_bpm <= self.max_valid_bpm
            && heart_confidence > self.min_confidence_threshold
        {
            heart_bpm
        } else {
            0.0
        };
        
        // Update history
        if breathing_bpm > 0.0 {
            self.breathing_history.push_back(breathing_bpm);
            if self.breathing_history.len() > 300 {
                self.breathing_history.pop_front();
            }
        }
        
        if heart_bpm > 0.0 {
            self.heart_history.push_back(heart_bpm);
            if self.heart_history.len() > 300 {
                self.heart_history.pop_front();
            }
        }
        
        VitalSigns {
            breathing_bpm,
            breathing_confidence,
            heart_bpm,
            heart_confidence,
            timestamp_ms: now,
        }
    }
    
    /// Extract BPM from a bandpass-filtered signal via zero-crossing
    fn extract_bpm_from_band(&self, signal: &[f64]) -> f64 {
        if signal.len() < 2 {
            return 0.0;
        }
        
        // Count zero crossings
        let mut zero_crossings = 0;
        for i in 1..signal.len() {
            if (signal[i - 1] < 0.0) != (signal[i] < 0.0) {
                zero_crossings += 1;
            }
        }
        
        // Convert zero-crossings to BPM
        // BPM = (zero_crossings / 2) / duration_sec * 60
        // Each cycle = 2 zero-crossings
        let duration_sec = signal.len() as f64 / self.sample_rate;
        let cycles = zero_crossings as f64 / 2.0;
        let freq_hz = cycles / duration_sec;
        freq_hz * 60.0  // Convert to BPM
    }
    
    /// Compute confidence score based on signal quality
    fn compute_confidence(&self, signal: &[f64], bpm: f64) -> f64 {
        if signal.is_empty() || bpm == 0.0 {
            return 0.0;
        }
        
        // Compute signal-to-noise ratio (SNR)
        // SNR = RMS(signal) / RMS(noise)
        let signal_mean = signal.iter().sum::<f64>() / signal.len() as f64;
        let signal_variance = signal
            .iter()
            .map(|x| (x - signal_mean).powi(2))
            .sum::<f64>()
            / signal.len() as f64;
        
        let snr = if signal_variance > 0.0 {
            signal_variance.sqrt() / 1e-6.max(signal_variance.abs().sqrt() * 0.1)
        } else {
            0.0
        };
        
        // Confidence = sigmoid(SNR) clamped to [0, 1]
        let confidence = 1.0 / (1.0 + (-snr / 5.0).exp());
        confidence.clamp(0.0, 1.0)
    }
    
    /// Reset filter state (e.g., when switching environments)
    pub fn reset(&mut self) {
        self.breathing_filter.reset();
        self.heart_filter.reset();
        self.breathing_history.clear();
        self.heart_history.clear();
    }
}

// =====================================================
// MULTI-NODE SYNCHRONIZER (TDM ALIGNMENT)
// =====================================================

pub struct MultiNodeSynchronizer {
    // Per-node circular buffers
    node_buffers: HashMap<u8, VecDeque<ProcessedCSIFrame>>,
    
    // Timestamp offset tracking
    timestamp_offsets: HashMap<u8, i32>,
    
    // Last seen sequence number per node
    last_sequence: HashMap<u8, u32>,
    
    // Dropped frame count per node
    dropped_frames: HashMap<u8, u32>,
}

impl MultiNodeSynchronizer {
    pub fn new() -> Self {
        Self {
            node_buffers: HashMap::new(),
            timestamp_offsets: HashMap::new(),
            last_sequence: HashMap::new(),
            dropped_frames: HashMap::new(),
        }
    }
    
    /// Register a new node
    pub fn register_node(&mut self, node_id: u8) {
        self.node_buffers.insert(node_id, VecDeque::with_capacity(CSI_BUFFER_SIZE));
        self.timestamp_offsets.insert(node_id, 0);
        self.last_sequence.insert(node_id, 0);
        self.dropped_frames.insert(node_id, 0);
    }
    
    /// Add a frame from a node
    pub fn push_frame(&mut self, frame: ProcessedCSIFrame) -> Result<()> {
        let node_id = frame.node_id;
        
        // Register if unknown
        if !self.node_buffers.contains_key(&node_id) {
            self.register_node(node_id);
        }
        
        // Detect dropped frames
        if let Some(&last_seq) = self.last_sequence.get(&node_id) {
            let expected_seq = last_seq + 1;
            let actual_seq = frame.sequence_number;
            
            if actual_seq != expected_seq && last_seq > 0 {
                let dropped = actual_seq.saturating_sub(expected_seq);
                if let Some(counter) = self.dropped_frames.get_mut(&node_id) {
                    *counter += dropped;
                }
            }
        }
        
        self.last_sequence.insert(node_id, frame.sequence_number);
        
        // Push to buffer
        if let Some(buffer) = self.node_buffers.get_mut(&node_id) {
            buffer.push_back(frame);
            
            // Keep buffer bounded
            while buffer.len() > CSI_BUFFER_SIZE {
                buffer.pop_front();
            }
        }
        
        Ok(())
    }
    
    /// Get synchronized frames from all active nodes
    /// Aligns timestamps across TDM slots
    pub fn get_synchronized_frames(&mut self) -> Result<Vec<ProcessedCSIFrame>> {
        // Find the reference timestamp (newest common time across all nodes)
        let mut min_timestamp = u32::MAX;
        
        for buffer in self.node_buffers.values() {
            if !buffer.is_empty() {
                if let Some(frame) = buffer.back() {
                    min_timestamp = min_timestamp.min(frame.timestamp_ms);
                }
            }
        }
        
        if min_timestamp == u32::MAX {
            return Err(DataTransferError::FusionError("No frames available".to_string()));
        }
        
        // Collect frames within alignment window
        let alignment_window_ms = TDM_TOTAL_CYCLE_MS as u32;
        let mut aligned_frames = Vec::new();
        
        for (node_id, buffer) in self.node_buffers.iter_mut() {
            // Find frames within the window
            while let Some(frame) = buffer.front() {
                if frame.timestamp_ms >= min_timestamp
                    && frame.timestamp_ms < min_timestamp + alignment_window_ms
                {
                    aligned_frames.push(buffer.pop_front().unwrap());
                    break;
                } else if frame.timestamp_ms < min_timestamp {
                    // Discard old frames
                    buffer.pop_front();
                } else {
                    break;
                }
            }
        }
        
        Ok(aligned_frames)
    }
    
    /// Get health status of all nodes
    pub fn get_node_health(&self) -> HashMap<u8, NodeHealth> {
        let mut health = HashMap::new();
        
        for (node_id, buffer) in self.node_buffers.iter() {
            let dropped = self.dropped_frames.get(node_id).copied().unwrap_or(0);
            let is_active = !buffer.is_empty();
            
            health.insert(
                *node_id,
                NodeHealth {
                    node_id: *node_id,
                    is_active,
                    buffer_depth: buffer.len(),
                    dropped_frames: dropped,
                },
            );
        }
        
        health
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealth {
    pub node_id: u8,
    pub is_active: bool,
    pub buffer_depth: usize,
    pub dropped_frames: u32,
}

// =====================================================
// MULTI-NODE FUSION ENGINE
// =====================================================

pub struct FusionEngine {
    synchronizer: Arc<Mutex<MultiNodeSynchronizer>>,
    vital_extractors: HashMap<u8, VitalSignsExtractor>,
}

impl FusionEngine {
    pub fn new() -> Self {
        Self {
            synchronizer: Arc::new(Mutex::new(MultiNodeSynchronizer::new())),
            vital_extractors: HashMap::new(),
        }
    }
    
    /// Process a raw CSI frame from the network
    pub fn process_raw_frame(&mut self, raw_frame: RawCSIFrame) -> Result<()> {
        let node_id = raw_frame.node_id;
        
        // Convert to processed frame
        let processed = self.raw_to_processed(raw_frame)?;
        
        // Add to synchronizer
        let mut sync = self.synchronizer.lock().unwrap();
        sync.push_frame(processed)?;
        
        Ok(())
    }
    
    /// Fuse synchronized frames from multiple nodes
    pub fn fuse_frames(&mut self) -> Result<FusionResult> {
        let mut sync = self.synchronizer.lock().unwrap();
        let frames = sync.get_synchronized_frames()?;
        
        if frames.is_empty() {
            return Err(DataTransferError::FusionError("No aligned frames".to_string()));
        }
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        // Extract per-node results
        let mut per_node_results = Vec::new();
        let mut combined_confidence = 0.0;
        let mut motion_detected = false;
        let mut motion_score = 0.0;
        let mut combined_vitals = VitalSigns::default();
        
        for frame in &frames {
            let node_id = frame.node_id;
            
            // Initialize vital extractor for this node if needed
            if !self.vital_extractors.contains_key(&node_id) {
                self.vital_extractors.insert(node_id, VitalSignsExtractor::new(SAMPLE_RATE_HZ));
            }
            
            // Extract vitals from phase
            let vitals = self.vital_extractors
                .get_mut(&node_id)
                .unwrap()
                .extract(&frame.phase.iter().map(|&x| x as f64).collect::<Vec<_>>());
            
            // Derive presence + motion from features
            let person_present = frame.snr_estimate > 1.2;
            let confidence = self.sigmoid(frame.motion_energy / 10.0);
            
            per_node_results.push(PerNodeResult {
                node_id,
                person_present,
                confidence,
                motion_score: frame.motion_energy,
            });
            
            combined_confidence += confidence;
            motion_detected = motion_detected || (frame.motion_energy > 5.0);
            motion_score = motion_score.max(frame.motion_energy);
            
            // Accumulate vitals
            combined_vitals.breathing_bpm += vitals.breathing_bpm;
            combined_vitals.heart_bpm += vitals.heart_bpm;
            combined_vitals.breathing_confidence += vitals.breathing_confidence;
            combined_vitals.heart_confidence += vitals.heart_confidence;
        }
        
        let n = frames.len() as f64;
        
        // Average vitals across nodes
        combined_vitals.breathing_bpm /= n;
        combined_vitals.heart_bpm /= n;
        combined_vitals.breathing_confidence /= n;
        combined_vitals.heart_confidence /= n;
        combined_vitals.timestamp_ms = now;
        
        // Compute spatial entropy (measure of diversity across nodes)
        let spatial_entropy = self.compute_spatial_entropy(&frames);
        
        Ok(FusionResult {
            fusion_timestamp_ms: now,
            active_node_count: frames.len(),
            total_links: frames.len() * (frames.len() - 1),
            combined_confidence: combined_confidence / n,
            person_present: motion_detected || combined_confidence > 0.5,
            motion_detected,
            motion_score: motion_score / 10.0,  // Normalize
            vitals: combined_vitals,
            per_node_results,
            spatial_entropy,
        })
    }
    
    /// Convert raw frame to processed frame
    fn raw_to_processed(&self, raw: RawCSIFrame) -> Result<ProcessedCSIFrame> {
        // Extract amplitude + phase statistics
        let amplitude: Vec<f64> = raw.amplitude.iter().map(|&x| x as f64).collect();
        let phase: Vec<f64> = raw.phase.iter().map(|&x| x as f64).collect();
        
        // Compute mean and variance
        let amp_mean = amplitude.iter().sum::<f64>() / amplitude.len() as f64;
        let amp_var = amplitude
            .iter()
            .map(|x| (x - amp_mean).powi(2))
            .sum::<f64>()
            / amplitude.len() as f64;
        
        let phase_mean = phase.iter().sum::<f64>() / phase.len() as f64;
        let phase_var = phase
            .iter()
            .map(|x| (x - phase_mean).powi(2))
            .sum::<f64>()
            / phase.len() as f64;
        
        // Compute band energies (low: 0-64, high: 64-128 subcarriers)
        let band_low = amplitude[0..64]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            / 64.0;
        let band_high = amplitude[64..128]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            / 64.0;
        
        let motion_energy = band_high / (band_low + 1e-6);
        
        // Compute entropy (Shannon entropy proxy)
        let snr = amp_var / (phase_var + 1e-6);
        let entropy = -amp_mean.abs() * (amp_mean.abs() + 1e-6).ln();
        
        Ok(ProcessedCSIFrame {
            node_id: raw.node_id,
            timestamp_ms: raw.timestamp_ms,
            sequence_number: raw.sequence_number,
            rssi: raw.rssi,
            channel: raw.channel,
            amplitude_mean: amp_mean,
            amplitude_var: amp_var,
            phase_mean: phase_mean,
            phase_stability: 1.0 / (phase_var + 1e-6),
            motion_energy,
            entropy,
            band_low,
            band_high,
            snr_estimate: snr,
            feature_vector: vec![0.0; FEATURE_VECTOR_SIZE],  // Placeholder
            server_receive_time_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        })
    }
    
    /// Compute spatial entropy (measure of signal diversity)
    fn compute_spatial_entropy(&self, frames: &[ProcessedCSIFrame]) -> f64 {
        if frames.len() < 2 {
            return 0.0;
        }
        
        // Entropy based on variance of confidence scores
        let confidences: Vec<f64> = frames
            .iter()
            .map(|f| self.sigmoid(f.motion_energy / 10.0))
            .collect();
        
        let mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
        let variance = confidences
            .iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f64>()
            / confidences.len() as f64;
        
        variance.sqrt()
    }
    
    /// Sigmoid activation function
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

// =====================================================
// UDP LISTENER (FIRMWARE INTEGRATION)
// =====================================================

pub struct CSIUdpListener {
    socket: UdpSocket,
    fusion_engine: Arc<Mutex<FusionEngine>>,
}

impl CSIUdpListener {
    pub fn new(listen_addr: &str) -> Result<Self> {
        let socket = UdpSocket::bind(listen_addr)?;
        socket.set_read_timeout(Some(Duration::from_secs(5)))?;
        
        Ok(Self {
            socket,
            fusion_engine: Arc::new(Mutex::new(FusionEngine::new())),
        })
    }
    
    /// Start listening for CSI frames (blocks)
    pub async fn listen(self: Arc<Self>) -> Result<()> {
        let mut buf = [0u8; UDP_BUFFER_SIZE];
        
        loop {
            match self.socket.recv_from(&mut buf) {
                Ok((size, src_addr)) => {
                    // Parse incoming frame
                    if let Ok(raw_frame) = serde_json::from_slice::<RawCSIFrame>(&buf[..size]) {
                        // Process frame
                        if let Ok(mut engine) = self.fusion_engine.lock() {
                            let _ = engine.process_raw_frame(raw_frame);
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // Timeout - continue
                    continue;
                }
                Err(e) => {
                    eprintln!("UDP recv error: {}", e);
                }
            }
        }
    }
    
    /// Non-blocking fusion (call periodically)
    pub fn try_fuse(&self) -> Result<Option<FusionResult>> {
        if let Ok(mut engine) = self.fusion_engine.lock() {
            match engine.fuse_frames() {
                Ok(result) => Ok(Some(result)),
                Err(DataTransferError::FusionError(_)) => Ok(None),  // No frames ready
                Err(e) => Err(e),
            }
        } else {
            Ok(None)
        }
    }
}

// =====================================================
// MQTT BRIDGE (HOME ASSISTANT)
// =====================================================

#[cfg(feature = "mqtt")]
pub struct MqttBridge {
    client: paho_mqtt::Client,
    base_topic: String,
}

#[cfg(feature = "mqtt")]
impl MqttBridge {
    pub fn new(broker_addr: &str, base_topic: &str) -> Result<Self> {
        let client_opts = paho_mqtt::CreateOptionsBuilder::new()
            .server_uris(&[broker_addr])
            .client_id("ruview-esp32")
            .finalize();
        
        let client = paho_mqtt::Client::new(client_opts)?;
        client.connect(paho_mqtt::ConnectOptions::new())?;
        
        Ok(Self {
            client,
            base_topic: base_topic.to_string(),
        })
    }
    
    /// Publish Home Assistant discovery payload
    pub fn publish_discovery(&self) -> Result<()> {
        let presence_config = serde_json::json!({
            "name": "RuView Presence",
            "unique_id": "ruview_presence",
            "device_class": "occupancy",
            "state_topic": format!("{}/presence/state", self.base_topic),
            "payload_on": "true",
            "payload_off": "false",
        });
        
        self.client.publish(paho_mqtt::Message::new(
            format!("homeassistant/binary_sensor/ruview_presence/config"),
            presence_config.to_string(),
            0,
        ))?;
        
        Ok(())
    }
    
    /// Publish fusion result to MQTT
    pub fn publish_result(&self, result: &FusionResult) -> Result<()> {
        let payload = serde_json::json!({
            "person_present": result.person_present,
            "motion_detected": result.motion_detected,
            "motion_score": result.motion_score,
            "confidence": result.combined_confidence,
            "breathing_bpm": result.vitals.breathing_bpm,
            "heart_bpm": result.vitals.heart_bpm,
            "active_nodes": result.active_node_count,
            "timestamp": result.fusion_timestamp_ms,
        });
        
        self.client.publish(paho_mqtt::Message::new(
            format!("{}/state", self.base_topic),
            payload.to_string(),
            0,
        ))?;
        
        Ok(())
    }
}

// =====================================================
// BUFFER MANAGER (CIRCULAR RING BUFFER)
// =====================================================

pub struct FrameBuffer {
    frames: VecDeque<ProcessedCSIFrame>,
    capacity: usize,
}

impl FrameBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
    
    pub fn push(&mut self, frame: ProcessedCSIFrame) {
        self.frames.push_back(frame);
        while self.frames.len() > self.capacity {
            self.frames.pop_front();
        }
    }
    
    pub fn pop(&mut self) -> Option<ProcessedCSIFrame> {
        self.frames.pop_front()
    }
    
    pub fn len(&self) -> usize {
        self.frames.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
    
    pub fn get_latest(&self) -> Option<&ProcessedCSIFrame> {
        self.frames.back()
    }
}

// =====================================================
// STATISTICS & HEALTH MONITORING
// =====================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TransferStats {
    pub total_frames_received: u64,
    pub total_frames_dropped: u64,
    pub fusion_cycles: u64,
    pub avg_latency_ms: f64,
    pub bandwidth_kbps: f64,
    pub error_count: u64,
}

pub struct StatsTracker {
    stats: Arc<Mutex<TransferStats>>,
    start_time: SystemTime,
}

impl StatsTracker {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(TransferStats::default())),
            start_time: SystemTime::now(),
        }
    }
    
    pub fn record_frame(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_frames_received += 1;
        }
    }
    
    pub fn record_dropped(&self, count: u32) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_frames_dropped += count as u64;
        }
    }
    
    pub fn record_fusion(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.fusion_cycles += 1;
        }
    }
    
    pub fn get_stats(&self) -> TransferStats {
        self.stats.lock().unwrap().clone()
    }
}

// =====================================================
// TESTS
// =====================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bandpass_filter() {
        let mut filter = BandpassFilter::new(0.1, 0.5, 100.0);
        let signal = vec![0.1; 100];
        let filtered = filter.apply(&signal);
        assert_eq!(filtered.len(), 100);
    }
    
    #[test]
    fn test_vital_signs_extraction() {
        let mut extractor = VitalSignsExtractor::new(100.0);
        let signal = vec![0.0; 1500];  // 15s @ 100Hz
        let vitals = extractor.extract(&signal);
        assert!(vitals.timestamp_ms > 0);
    }
    
    #[test]
    fn test_synchronizer_registration() {
        let mut sync = MultiNodeSynchronizer::new();
        sync.register_node(1);
        assert!(sync.node_buffers.contains_key(&1));
    }
    
    #[test]
    fn test_fusion_engine_creation() {
        let engine = FusionEngine::new();
        assert!(engine.vital_extractors.is_empty());
    }
}
