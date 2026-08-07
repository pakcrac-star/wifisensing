
//! ============================================================================
//! Module 12: Multi-Node Sensor Fusion (`fusion.rs`)
//! ============================================================================
//! 
//! Mathematical Model:
//! Combines distributed RF observations from N nodes (1 ≤ N ≤ 6) into a single 
//! unified physical representation of the monitored volume.
//! 
//! 1. Dynamic Quality-Weighted Fusion:
//!    Primary fusion weight for node i is proportional to its PHY link quality q_i:
//!        w_i = q_i / Σ_{j=1}^N q_j
//! 
//! 2. Spatial Outlier Rejection & Node Consensus (M-Estimator / Gaussian Kernel):
//!    For N ≥ 2, nodes exhibiting severe multipath corruption are downweighted 
//!    based on their residual distance from the weighted spatial mean:
//!        a_i = exp( - (x_i - x_mean)² / (2 * σ_consensus²) )
//!        w_i_final = q_i * a_i
//! 
//! 3. Bayesian Presence & Spatial Confidence:
//!    Overall spatial confidence scales with node agreement and active node count N:
//!        C_presence = ConsensusScore * (1 - exp( - λ_nodes * N )) * Mean_Quality
//! 
//! 4. Temporal Consistency (1D Kalman Tracking):
//!    Filters spatial estimations across time to preserve momentum and prevent 
//!    instantaneous state dropouts.
//! 
//! Invariant Guarantees:
//! 1. Strictly produces `UnifiedScene` and nothing more.
//! 2. Zero Allocation: Memory scales statically up to `MAX_NODES = 6`.
//! 3. Elastic Self-Adjustment: Seamlessly handles 1 to 6 nodes dropping in/out.
//! 4. NO Classification: Never assigns semantic labels (e.g. "Walking").
//! ============================================================================

/// Maximum physical sensing nodes supported by the fusion architecture
pub const MAX_NODES: usize = 6;

const EPSILON: f32 = 1e-6;

// Pre-computed squared sigmas to avoid `#![no_std]` const float math errors
const CONSENSUS_SIGMA_VELOCITY_SQ: f32 = 1.0;    // (1.0 m/s)^2
const CONSENSUS_SIGMA_ENERGY_SQ: f32   = 400.0;  // (20.0 Joules proxy)^2
const CONSENSUS_SIGMA_ENTROPY_SQ: f32  = 0.0625; // (0.25 Shannon scale)^2

const PROCESS_NOISE_ENERGY: f32 = 0.5;
const PROCESS_NOISE_VELOCITY: f32 = 0.1;
const PROCESS_NOISE_VITALS: f32 = 0.01;
const PROCESS_NOISE_ENTROPY: f32 = 0.02;

const MEASUREMENT_NOISE_ENERGY: f32 = 2.0;
const MEASUREMENT_NOISE_VELOCITY: f32 = 0.5;
const MEASUREMENT_NOISE_VITALS: f32 = 0.2;
const MEASUREMENT_NOISE_ENTROPY: f32 = 0.1;

const EMPTY_SCENE: UnifiedScene = UnifiedScene {
    kinetic_energy: 0.0,
    velocity_mps: 0.0,
    micro_displacement_hz: 0.0,
    spatial_entropy: 0.0,
    presence_confidence: 0.0,
    active_node_count: 0,
};

/// Physical observation reported by a single ESP32 sensing node.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeObservation {
    /// Hardware identifier of the receiving node (1 .. MAX_NODES)
    pub node_id: u8,
    /// PHY link quality score from `interference.rs` [0.0 = corrupt, 1.0 = pristine]
    pub quality: f32,
    /// Estimated target velocity from `doppler.rs` (m/s)
    pub doppler_velocity_mps: f32,
    /// Dominant micro-vibration frequency from `vitals.rs` (Hz)
    pub vitals_frequency_hz: f32,
    /// Spatial multipath chaos from `entropy.rs` [0.0 - 1.0]
    pub spatial_entropy: f32,
    /// Total channel power fluctuation proxy (Joules equivalent)
    pub kinetic_energy: f32,
}

impl NodeObservation {
    pub const fn zero() -> Self {
        Self {
            node_id: 0,
            quality: 0.0,
            doppler_velocity_mps: 0.0,
            vitals_frequency_hz: 0.0,
            spatial_entropy: 0.0,
            kinetic_energy: 0.0,
        }
    }
}

impl Default for NodeObservation {
    fn default() -> Self {
        Self::zero()
    }
}

/// Abstract representation of the physical space (Consumed by `classification.rs`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnifiedScene {
    /// Fused kinetic energy proxy across active nodes
    pub kinetic_energy: f32,
    /// Fused radial velocity in meters per second
    pub velocity_mps: f32,
    /// Fused micro-displacement frequency in Hz
    pub micro_displacement_hz: f32,
    /// Fused spatial Shannon entropy
    pub spatial_entropy: f32,
    /// Bayesian presence confidence metric [0.0 - 1.0]
    pub presence_confidence: f32,
    /// Number of valid nodes used in generating this scene
    pub active_node_count: u8,
}

impl UnifiedScene {
    pub const fn zero() -> Self {
        EMPTY_SCENE
    }
}

impl Default for UnifiedScene {
    fn default() -> Self {
        Self::zero()
    }
}

// ============================================================================
// 1D Kalman State Tracker for Temporal Consistency
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct StateTracker {
    state: f32,
    covariance: f32,
    process_noise: f32,
    measurement_noise: f32,
}

impl StateTracker {
    const fn new(q: f32, r: f32) -> Self {
        Self {
            state: 0.0,
            covariance: 1.0,
            process_noise: q,
            measurement_noise: r,
        }
    }

    #[inline(always)]
    fn update(&mut self, measurement: f32) -> f32 {
        let predicted_covariance = self.covariance + self.process_noise;
        let gain = predicted_covariance / (predicted_covariance + self.measurement_noise);
        self.state += gain * (measurement - self.state);
        self.covariance = (1.0 - gain) * predicted_covariance;
        self.state
    }
}

// ============================================================================
// Multi-Node Fusion Engine
// ============================================================================

pub struct FusionEngine {
    // Temporal Kalman trackers for output parameters
    tracker_energy: StateTracker,
    tracker_velocity: StateTracker,
    tracker_vitals: StateTracker,
    tracker_entropy: StateTracker,
}

impl Default for FusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionEngine {
    /// Initializes the fusion engine with physical temporal noise dynamics.
    pub fn new() -> Self {
        Self {
            tracker_energy: StateTracker::new(PROCESS_NOISE_ENERGY, MEASUREMENT_NOISE_ENERGY),
            tracker_velocity: StateTracker::new(PROCESS_NOISE_VELOCITY, MEASUREMENT_NOISE_VELOCITY),
            tracker_vitals: StateTracker::new(PROCESS_NOISE_VITALS, MEASUREMENT_NOISE_VITALS),
            tracker_entropy: StateTracker::new(PROCESS_NOISE_ENTROPY, MEASUREMENT_NOISE_ENTROPY),
        }
    }

    /// Primary interface: Combines 1 to MAX_NODES spatial observations into a `UnifiedScene`.
    /// 
    /// # Elastic Self-Adjustment
    /// Automatically handles any array slice from size 1 up to `MAX_NODES`.
    pub fn fuse_nodes(&mut self, observations: &[NodeObservation]) -> UnifiedScene {
        let count = if observations.len() > MAX_NODES { MAX_NODES } else { observations.len() };
        if count == 0 {
            return EMPTY_SCENE;
        }

        // 1. Filter out invalid/corrupted nodes (Quality <= 0)
        let mut valid_obs = [NodeObservation::zero(); MAX_NODES];
        let mut valid_count = 0usize;

        let mut i = 0;
        while i < count {
            if observations[i].quality > EPSILON {
                valid_obs[valid_count] = observations[i];
                valid_count += 1;
            }
            i += 1;
        }

        if valid_count == 0 {
            return EMPTY_SCENE;
        }

        // 2. Compute First-Pass Quality-Weighted Accumulations
        let mut sum_quality = 0.0f32;
        let mut weighted_energy_sum = 0.0f32;
        let mut weighted_velocity_sum = 0.0f32;
        let mut weighted_entropy_sum = 0.0f32;

        i = 0;
        while i < valid_count {
            let q = valid_obs[i].quality;
            sum_quality += q;
            weighted_energy_sum += valid_obs[i].kinetic_energy * q;
            weighted_velocity_sum += valid_obs[i].doppler_velocity_mps * q;
            weighted_entropy_sum += valid_obs[i].spatial_entropy * q;
            i += 1;
        }

        // Safe inverse mapping replacing `.max()`
        let safe_quality = if sum_quality < EPSILON { EPSILON } else { sum_quality };
        let inv_quality = 1.0 / safe_quality;
        
        let mean_energy = weighted_energy_sum * inv_quality;
        let mean_velocity = weighted_velocity_sum * inv_quality;
        let mean_entropy = weighted_entropy_sum * inv_quality;

        // 3. Perform Spatial Consensus & Outlier Downweighting (for N >= 2)
        let mut final_weights = [0.0f32; MAX_NODES];
        let mut total_final_weight = 0.0f32;
        let mut consensus_sum = 0.0f32;

        i = 0;
        while i < valid_count {
            let obs = &valid_obs[i];

            let consensus_weight = if valid_count >= 2 {
                // Compute spatial residuals from weighted mean
                let diff_v = obs.doppler_velocity_mps - mean_velocity;
                let diff_e = obs.kinetic_energy - mean_energy;
                let diff_h = obs.spatial_entropy - mean_entropy;

                // Gaussian Kernel Consensus Weight
                let dist_sq = (diff_v * diff_v) / CONSENSUS_SIGMA_VELOCITY_SQ
                            + (diff_e * diff_e) / CONSENSUS_SIGMA_ENERGY_SQ
                            + (diff_h * diff_h) / CONSENSUS_SIGMA_ENTROPY_SQ;

                gaussian(dist_sq)
            } else {
                1.0 // Single node: agreement is trivial
            };

            consensus_sum += consensus_weight;
            let final_weight = obs.quality * consensus_weight;
            final_weights[i] = final_weight;
            total_final_weight += final_weight;
            i += 1;
        }

        // Safe inverse mapping replacing `.max()`
        let safe_total_weight = if total_final_weight < EPSILON { EPSILON } else { total_final_weight };
        let inv_weight = 1.0 / safe_total_weight;

        // 4. Compute Final Consensus-Fused Parameters
        let mut fused_energy = 0.0f32;
        let mut fused_velocity = 0.0f32;
        let mut fused_vitals = 0.0f32;
        let mut fused_entropy = 0.0f32;

        i = 0;
        while i < valid_count {
            let w = final_weights[i] * inv_weight;
            fused_energy += valid_obs[i].kinetic_energy * w;
            fused_velocity += valid_obs[i].doppler_velocity_mps * w;
            fused_vitals += valid_obs[i].vitals_frequency_hz * w;
            fused_entropy += valid_obs[i].spatial_entropy * w;
            i += 1;
        }

        // 5. Evaluate Bayesian Presence Confidence
        let spatial_consensus_score = consensus_sum / (valid_count as f32);
        let average_quality = sum_quality / (valid_count as f32);
        
        // Multi-node spatial diversity gain: C_nodes = 1 - exp(-0.6 * N)
        let node_diversity_factor = 1.0 - expf(-0.6 * (valid_count as f32));
        
        // Safe bounding replacing `.clamp()`
        let raw_presence_confidence = clamp_01(
            0.45 * spatial_consensus_score + 
            0.35 * average_quality + 
            0.20 * node_diversity_factor
        );

        // 6. Apply Temporal Kalman Filtering (Momentum / Consistency)
        let smoothed_energy = self.tracker_energy.update(fused_energy);
        let smoothed_velocity = self.tracker_velocity.update(fused_velocity);
        let smoothed_vitals = self.tracker_vitals.update(fused_vitals);
        let smoothed_entropy = self.tracker_entropy.update(fused_entropy);

        UnifiedScene {
            kinetic_energy: smoothed_energy,
            velocity_mps: smoothed_velocity,
            micro_displacement_hz: smoothed_vitals,
            spatial_entropy: smoothed_entropy,
            presence_confidence: raw_presence_confidence,
            active_node_count: valid_count as u8,
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
fn gaussian(distance_sq: f32) -> f32 {
    clamp_01(expf(-0.5 * distance_sq))
}

#[inline(always)]
fn expf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    { x.exp() }
    #[cfg(not(feature = "std"))]
    { libm::expf(x) }
}
