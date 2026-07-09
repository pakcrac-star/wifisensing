"""
------------------------------------------------------------
server.py

WiFi Radar Streaming Server (Laptop)

ROLE
----
• Receive CSI feature packets from ESP32-S3
• Parse + validate incoming JSON stream
• Maintain real-time buffer
• Forward features to model.py
• Receive human state from geometry pipeline
• Output final human reconstruction

DATA FLOW
---------
ESP32 → server.py → model.py → geometry.py → output
------------------------------------------------------------
"""

import json
import socket
import numpy as np

from model import WiFiRadarModel


HOST = "0.0.0.0"
PORT = 9000


# ----------------------------------------------------------
# SERVER CORE
# ----------------------------------------------------------
class WiFiRadarServer:

    def __init__(self):

        self.model = WiFiRadarModel()
        self.model.load()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.buffer = []

    # ------------------------------------------------------
    # START SERVER
    # ------------------------------------------------------
    def start(self):

        self.socket.bind((HOST, PORT))
        self.socket.listen(1)

        print("\n===================================")
        print(" WiFi Radar AI Server Running")
        print(f" Listening on {HOST}:{PORT}")
        print("===================================\n")

        while True:

            client, addr = self.socket.accept()

            print(f"[CONNECTED] ESP32 → {addr}")

            self.handle_client(client)

    # ------------------------------------------------------
    # HANDLE CLIENT STREAM
    # ------------------------------------------------------
    def handle_client(self, client):

        file = client.makefile("r")  # IMPORTANT: line-based streaming

        while True:

            try:

                line = file.readline()

                if not line:
                    break

                packet = self.decode(line)

                if packet is None:
                    continue

                # forward to AI model
                result = self.model.update(packet)

                # output final geometry
                if result is not None:

                    self.publish(result)

            except Exception as e:

                print("[ERROR]", e)
                break

        client.close()
        print("[DISCONNECTED] ESP32")

    # ------------------------------------------------------
    # DECODE JSON PACKET
    # ------------------------------------------------------
    def decode(self, data):

        try:

            packet = json.loads(data)

            # -----------------------------
            # REQUIRED FIELD: CSI FEATURES
            # -----------------------------
            features = packet.get("features", None)

            if features is None:
                return None

            # convert to numpy vector
            feature_vector = np.array([
                features.get("amplitude_mean", 0),
                features.get("phase_stability", 0),
                features.get("motion_energy", 0),
                features.get("entropy", 0),
                features.get("band_low", 0),
                features.get("band_high", 0),
                features.get("snr_estimate", 0)
            ], dtype=np.float32)

            return feature_vector

        except Exception as e:

            print("[DECODE ERROR]", e)

            return None

    # ------------------------------------------------------
    # OUTPUT HUMAN STATE
    # ------------------------------------------------------
    def publish(self, human_state):

        result = self.model.geometry.to_dict(human_state)

        print("\n--- HUMAN STATE ---")
        print(json.dumps(result, indent=4))


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":

    server = WiFiRadarServer()
    server.start()