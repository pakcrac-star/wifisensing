//! ============================================================================
//! Module 13: Machine Learning & Classification (`classification.rs`)
//! ============================================================================
//! 
//! Mathematical Model:
//! Translates physical scene metrics into discrete semantic states using statistical 
//! learning and temporal smoothing.
//! 
//! Hidden Markov Model (HMM) Temporal Smoothing:
//! The true human state X_t is hidden. We observe E_t (UnifiedScene).
//! Belief updating follows Bayesian inference:
//! 
//!     P(X_t | E_{1:t}) ∝ P(E_t | X_t) * Σ [ P(X_t | X_{t-1}) * P(X_{t-1} | E_{1:t-1}) ]
//! 
//! Where:
//! - P(E_t | X_t): Emission probability (from Decision Tree / TinyML output)
//! - P(X_t | X_{t-1}): Transition matrix (kinematic constraints)
//! 
//! Invariant Guarantees:
//! 1. CSI Isolation: Never touches phase, amplitude, or hardware subcarriers.
//! 2. Feature Boundary: Only accepts `UnifiedScene` (macroscopic physics).
//! 3. Zero Allocation: All matrices and belief vectors are fixed-size stack arrays.
//! 4. ML Gateway: Prepares normalized feature vectors for `model.rs` (TinyML/TFLite).
//! ============================================================================

/// Number of discrete activity classes the system tracks.
pub const NUM_CLASSES: usize = 5;

/// Discrete human activity states.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityState {
    EmptySpace = 0,
    Stationary = 1,     // Sitting/Lying (Vital signs dominant)
    Walking = 2,        // Moderate Doppler velocity
    Running = 3,        // High Doppler velocity, high entropy
    Falling = 4,        // Extreme transient velocity, rapid spatial variance
}

/// Abstract representation of the physical space, produced by `fusion.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct UnifiedScene {
    pub kinetic_energy: f32,
    pub velocity_mps: f32,
    pub micro_displacement_hz: f32,
    pub spatial_entropy: f32,
    pub presence_confidence: f32,
}

/// Normalized feature vector ready for C-FFI export to TinyML/Neural Networks.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ModelFeatureVector {
    pub features: [f32; 8],
    pub feature_count: u8,
}

/// Emission probabilities outputted by the base classifier (Tree/NN) for the current frame.
type EmissionProbabilities = [f32; NUM_CLASSES];

// ============================================================================
// Classification Engine & Temporal Smoothing
// ============================================================================

pub struct ClassificationEngine {
    /// P(X_{t-1} | E_{1:t-1}): The previous state belief distribution.
    state_beliefs: [f32; NUM_CLASSES],
    
    /// P(X_t | X_{t-1}): Transition matrix enforcing physical kinematic constraints.
    transition_matrix: [[f32; NUM_CLASSES]; NUM_CLASSES],
}

impl ClassificationEngine {
    /// Primary interface: Takes a physical scene, evaluates classical ML rules,
    /// applies HMM smoothing, and returns the highest-probability state.
    pub fn predict_state(&mut self, scene: &UnifiedScene) -> ActivityState {
        // 1. Generate Raw Emissions (from internal Decision Tree or RF)
        let emissions = self.heuristic_decision_tree(scene);

        // 2. Bayesian Belief Update (HMM Forward Pass)
        self.apply_temporal_smoothing(emissions);

        // 3. Argmax to find the most probable state
        self.get_dominant_state()
    }

    /// Prepares a normalized, memory-safe tensor for an external Neural Network (e.g., TFLite in C).
    pub fn prepare_nn_features(&self, scene: &UnifiedScene) -> ModelFeatureVector {
        ModelFeatureVector {
            features: [
                scene.kinetic_energy,
                scene.velocity_mps,
                scene.micro_displacement_hz,
                scene.spatial_entropy,
                scene.presence_confidence,
                // Include temporal context: previous top beliefs
                self.state_beliefs[ActivityState::Walking as usize],
                self.state_beliefs[ActivityState::Stationary as usize],
                self.state_beliefs[ActivityState::EmptySpace as usize],
            ],
            feature_count: 8,
        }
    }

    // ========================================================================
    // Internal ML / Math Methods
    // ========================================================================

    /// An embedded rule-based Decision Tree serving as the base classifier.
    fn heuristic_decision_tree(&self, scene: &UnifiedScene) -> EmissionProbabilities {
        let mut prob = [0.01_f32; NUM_CLASSES]; // Base Laplace smoothing
        
        // Strict f32 literals prevent f64 software-float promotion and type mismatch errors
        if scene.presence_confidence < 0.2_f32 {
            prob[ActivityState::EmptySpace as usize] = 0.96_f32;
        } else if scene.velocity_mps > 2.5_f32 && scene.kinetic_energy > 50.0_f32 {
            prob[ActivityState::Running as usize] = 0.90_f32;
        } else if scene.velocity_mps > 0.5_f32 {
            prob[ActivityState::Walking as usize] = 0.85_f32;
        } else if scene.kinetic_energy > 80.0_f32 && scene.spatial_entropy > 0.8_f32 {
            prob[ActivityState::Falling as usize] = 0.90_f32; 
        } else {
            prob[ActivityState::Stationary as usize] = 0.95_f32;
        }

        prob
    }

    /// Performs the Hidden Markov Model belief update.
    fn apply_temporal_smoothing(&mut self, emissions: EmissionProbabilities) {
        let mut new_beliefs = [0.0_f32; NUM_CLASSES];
        let mut sum = 0.0_f32;

        // Σ [ P(X_t | X_{t-1}) * P(X_{t-1}) ]
        for current_state in 0..NUM_CLASSES {
            let mut prior_sum = 0.0_f32;
            for prev_state in 0..NUM_CLASSES {
                prior_sum += self.transition_matrix[prev_state][current_state] * self.state_beliefs[prev_state];
            }
            
            // Multiply by Emission P(E_t | X_t)
            new_beliefs[current_state] = emissions[current_state] * prior_sum;
            sum += new_beliefs[current_state];
        }

        // Normalize distribution so it sums to 1.0
        if sum > 0.0_f32 {
            for i in 0..NUM_CLASSES {
                self.state_beliefs[i] = new_beliefs[i] / sum;
            }
        }
    }

    /// Extracts the highest probability state from the belief vector.
    fn get_dominant_state(&self) -> ActivityState {
        let mut max_p = -1.0_f32;
        let mut best_state = ActivityState::EmptySpace;

        let states = [
            ActivityState::EmptySpace,
            ActivityState::Stationary,
            ActivityState::Walking,
            ActivityState::Running,
            ActivityState::Falling,
        ];

        // Idiomatic iterator loop removes the need for array bounds checking
        for (i, &probability) in self.state_beliefs.iter().enumerate() {
            if probability > max_p {
                max_p = probability;
                best_state = states[i];
            }
        }

        best_state
    }
}

impl Default for ClassificationEngine {
    /// Initializes the engine with a default transition matrix based on human kinematics.
    fn default() -> Self {
        Self {
            // Initial belief: 100% Empty Space (Strict f32)
            state_beliefs: [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32],
            
            // T[prev][next]: Rows sum to 1.0. 
            transition_matrix: [
                // Next: Empty, Stat,  Walk,  Run,   Fall
                [0.90_f32, 0.10_f32, 0.00_f32, 0.00_f32, 0.00_f32], // Prev: EmptySpace
                [0.05_f32, 0.85_f32, 0.10_f32, 0.00_f32, 0.00_f32], // Prev: Stationary
                [0.00_f32, 0.10_f32, 0.80_f32, 0.09_f32, 0.01_f32], // Prev: Walking
                [0.00_f32, 0.00_f32, 0.15_f32, 0.80_f32, 0.05_f32], // Prev: Running
                [0.05_f32, 0.80_f32, 0.00_f32, 0.00_f32, 0.15_f32], // Prev: Falling
            ],
        }
    }
}
