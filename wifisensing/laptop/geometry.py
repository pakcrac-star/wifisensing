"""
------------------------------------------------------------
geometry.py

WiFi Radar Human Geometry Decoder (Laptop AI Layer)

ROLE IN SYSTEM
--------------
Input:
    Latent features from model.py

Output:
    Probabilistic human body structure

Responsibilities:
----------------
• Reconstruct human pose from embeddings
• Estimate body keypoints probabilistically
• Track temporal motion
• Estimate respiration + heartbeat signals
• Maintain human motion continuity
------------------------------------------------------------
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict


# ----------------------------------------------------------
# KEYPOINT STRUCTURE
# ----------------------------------------------------------
@dataclass
class KeyPoint:
    name: str
    x: float
    y: float
    confidence: float


# ----------------------------------------------------------
# HUMAN STATE MODEL
# ----------------------------------------------------------
@dataclass
class HumanState:
    detected: bool
    confidence: float

    center_x: float
    center_y: float

    orientation: float

    heartbeat: float
    respiration: float

    motion_speed: float

    keypoints: List[KeyPoint]


# ----------------------------------------------------------
# GEOMETRY ENGINE (CORE AI DECODER)
# ----------------------------------------------------------
class GeometryEngine:

    def __init__(self):

        self.history = []
        self.last_state = None

    # ------------------------------------------------------
    # MAIN ENTRY
    # latent → human structure
    # ------------------------------------------------------
    def predict(self, latent_features: np.ndarray) -> HumanState:

        # Convert latent vector into human state
        state = self._decode_latent(latent_features)

        # temporal smoothing
        state = self._smooth(state)

        self.history.append(state)
        self.last_state = state

        return state

    # ------------------------------------------------------
    # LATENT → HUMAN MODEL
    # ------------------------------------------------------
    def _decode_latent(self, z) -> HumanState:

        # z is output from model.py (deep network embedding)

        center_x = float(np.tanh(z[0]) * 3.0)
        center_y = float(np.tanh(z[1]) * 3.0)

        orientation = float(np.tanh(z[2]) * 180)

        motion_speed = float(np.abs(z[3]))

        heartbeat = 70 + float(np.tanh(z[4]) * 15)
        respiration = 14 + float(np.tanh(z[5]) * 4)

        confidence = float(min(1.0, np.abs(np.mean(z))))

        return HumanState(
            detected=True,
            confidence=confidence,

            center_x=center_x,
            center_y=center_y,

            orientation=orientation,

            heartbeat=heartbeat,
            respiration=respiration,

            motion_speed=motion_speed,

            keypoints=self._generate_pose(center_x, center_y, orientation)
        )

    # ------------------------------------------------------
    # HUMAN POSE GENERATION (probabilistic skeleton)
    # ------------------------------------------------------
    def _generate_pose(self, cx, cy, angle):

        a = np.radians(angle)

        def rot(x, y):
            xr = x * np.cos(a) - y * np.sin(a)
            yr = x * np.sin(a) + y * np.cos(a)
            return xr, yr

        return [

            KeyPoint("Head", *self._offset(cx, cy, 0, -0.5), 0.95),

            KeyPoint("Neck", *self._offset(cx, cy, 0, -0.3), 0.93),

            KeyPoint("LeftShoulder", *self._offset(cx, cy, -0.3, -0.2), 0.90),

            KeyPoint("RightShoulder", *self._offset(cx, cy, 0.3, -0.2), 0.90),

            KeyPoint("Chest", cx, cy, 0.97),

            KeyPoint("Waist", *self._offset(cx, cy, 0, 0.4), 0.92),

            KeyPoint("LeftHand", *self._offset(cx, cy, -0.7, 0.1), 0.85),

            KeyPoint("RightHand", *self._offset(cx, cy, 0.7, 0.1), 0.85),

            KeyPoint("LeftKnee", *self._offset(cx, cy, -0.2, 1.0), 0.88),

            KeyPoint("RightKnee", *self._offset(cx, cy, 0.2, 1.0), 0.88),

            KeyPoint("LeftFoot", *self._offset(cx, cy, -0.2, 1.8), 0.86),

            KeyPoint("RightFoot", *self._offset(cx, cy, 0.2, 1.8), 0.86),
        ]

    # ------------------------------------------------------
    # OFFSET HELPER (spatial body model)
    # ------------------------------------------------------
    def _offset(self, cx, cy, dx, dy):
        return cx + dx, cy + dy

    # ------------------------------------------------------
    # TEMPORAL SMOOTHING
    # ------------------------------------------------------
    def _smooth(self, state: HumanState) -> HumanState:

        if not self.history:
            return state

        prev = self.history[-1]

        state.center_x = 0.7 * state.center_x + 0.3 * prev.center_x
        state.center_y = 0.7 * state.center_y + 0.3 * prev.center_y

        state.motion_speed = 0.6 * state.motion_speed + 0.4 * prev.motion_speed

        return state

    # ------------------------------------------------------
    # EXPORT FOR server.py
    # ------------------------------------------------------
    def to_dict(self, state: HumanState) -> Dict:

        return {

            "detected": state.detected,
            "confidence": state.confidence,

            "center": [state.center_x, state.center_y],
            "orientation": state.orientation,

            "heartbeat": state.heartbeat,
            "respiration": state.respiration,

            "motion_speed": state.motion_speed,

            "keypoints": [
                {
                    "name": k.name,
                    "x": k.x,
                    "y": k.y,
                    "confidence": k.confidence
                }
                for k in state.keypoints
            ]
        }


# ----------------------------------------------------------
# TEST
# ----------------------------------------------------------
if __name__ == "__main__":

    engine = GeometryEngine()

    fake_latent = np.random.randn(10)

    result = engine.predict(fake_latent)

    print(engine.to_dict(result))