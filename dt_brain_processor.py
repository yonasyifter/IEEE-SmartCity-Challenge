"""
dt_brain_processor.py
======================
Event-Driven Digital Twin Intelligence Engine.
Subscribes to Eclipse Ditto events via WebSockets, calculates absorbing Markov
projections dynamically upon telemetry ingestion, and dispatches Robin Hood
routing commands.

Run: python3 dt_brain_processor.py
"""

import json
import base64
import numpy as np
import requests
import websocket

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DITTO_HOST = "localhost:8080"
DITTO_URL = f"http://{DITTO_HOST}/api/2"
AUTH_USER = "ditto"
AUTH_PASS = "ditto"
AUTH_HEADER = "Basic " + base64.b64encode(f"{AUTH_USER}:{AUTH_PASS}".encode()).decode()

STATES = ["Low", "Medium", "High", "Critical"]
RISK_STATES = ["High", "Critical"]
K_STEPS = 12
BASE_CYCLE_GREEN = 45
BOOST_PCT = 0.20

# Baseline transition matrix priors for behavioral dynamic sectors
DEFAULT_MATRIX = np.array([
    [0.70, 0.20, 0.08, 0.02],
    [0.15, 0.65, 0.15, 0.05],
    [0.05, 0.20, 0.60, 0.15],
    [0.02, 0.08, 0.30, 0.60],
])

TRANSITION_MATRICES = {f"Behavioral_Zone_{i}": DEFAULT_MATRIX.copy() for i in range(1, 6)}
BASELINES = {f"Behavioral_Zone_{i}": 0.45 for i in range(1, 6)}


# --------------------------------------------------------------------------
# Markovian Absorbing Computation
# --------------------------------------------------------------------------
def state_vector(state_label):
    v = np.zeros(len(STATES))
    if state_label in STATES:
        v[STATES.index(state_label)] = 1.0
    else:
        v[STATES.index("Medium")] = 1.0
    return v


def compute_absorbing_risk(matrix, current_state, k=K_STEPS):
    """Calculates probability of absorbing into High/Critical within k steps."""
    risk_idx = [STATES.index(s) for s in RISK_STATES]
    absorbing = np.array(matrix, dtype=float, copy=True)
    for i in risk_idx:
        absorbing[i, :] = 0.0
        absorbing[i, i] = 1.0
    reached = state_vector(current_state) @ np.linalg.matrix_power(absorbing, k)
    return float(reached[risk_idx].sum())


# --------------------------------------------------------------------------
# WebSocket Callbacks
# --------------------------------------------------------------------------
def on_open(ws):
    print("=" * 70)
    print("  [TWIN BRAIN] Connected to Eclipse Ditto WebSocket (ws://localhost:8080/ws/2)")
    # Eclipse Ditto requires plain-text START-SEND-EVENTS to activate streaming
    ws.send("START-SEND-EVENTS")
    print("  [TWIN BRAIN] Protocol Handshake: 'START-SEND-EVENTS' dispatched.")
    print("  [TWIN BRAIN] Listening for telemetry ingestion across behavioral twins...")
    print("=" * 70)


def on_message(ws, message):
    try:
        event = json.loads(message)
    except json.JSONDecodeError:
        return

    path = event.get("path", "")
    topic = event.get("topic", "")

    # Filter strictly for telemetry property updates
    if "/features/telemetry" not in path:
        return

    # Extract namespace and zone name from topic: <namespace>/<entity>/things/twin/events/modified
    parts = topic.split("/")
    if len(parts) >= 2:
        thing_id = f"{parts[0]}:{parts[1]}"
        zone_name = parts[1]
    else:
        thing_id = event.get("thingId", "traffic.zones:Zone_1")
        zone_name = thing_id.split(":")[-1]

    value = event.get("value", {})
    if not isinstance(value, dict):
        return

    current_wait = value.get("avg_wait_time") or value.get("avg_wait_before", 25.0)
    current_state = value.get("current_state", "Medium")

    print(f"\n[EVENT INGESTED] {zone_name} reported telemetry:")
    print(f"                 Current State: {current_state} | Average Wait: {current_wait:.1f}s")

    # 1. Compute Absorbing Markov Projection
    T = TRANSITION_MATRICES.get(zone_name, DEFAULT_MATRIX)
    tau = BASELINES.get(zone_name, 0.45)
    p_k = compute_absorbing_risk(T, current_state, k=K_STEPS)
    congestion_flag = bool(p_k > tau)

    print(f"[TWIN BRAIN]     Markov Risk (P_K within {K_STEPS}h): {p_k:.3f} | Baseline (tau): {tau:.3f}")
    print(f"                 Congestion Tripwire: {'TRIGGERED' if congestion_flag else 'NOMINAL'}")

    # 2. Update Prediction Feature in Ditto
    pred_payload = {
        "baseline_threshold": tau,
        "projected_probability": p_k,
        "congestion_flag": congestion_flag,
        "priority_rank": 1 if congestion_flag else 3,
        "hours_to_threshold": 4 if congestion_flag else None
    }
    try:
        requests.put(
            f"{DITTO_URL}/things/{thing_id}/features/digital_twin_predictor/properties",
            json=pred_payload,
            auth=(AUTH_USER, AUTH_PASS),
            headers={"Content-Type": "application/json"},
            timeout=5
        )
    except Exception as e:
        print(f"[ERROR] Failed to update twin prediction feature: {e}")

    # 3. Evaluate Robin Hood Priority Reallocation
    if congestion_flag and current_wait > 22.0:
        relief = current_wait * BOOST_PCT
        target_wait = max(15.0, current_wait - relief)
        pct_relief = (relief / current_wait) * 100
        target_green = int(BASE_CYCLE_GREEN * (1.0 + (pct_relief / 100.0)))
        inflow_metering = round(max(0.60, 1.0 - (pct_relief / 200.0)), 2)
        command = "BOOST_PRIORITY_EXTEND_GREEN"
    else:
        target_wait = current_wait
        target_green = BASE_CYCLE_GREEN
        inflow_metering = 1.0
        command = "MAINTAIN_STANDARD_SPLIT"

    # 4. Dispatch Actuation Target to desiredProperties
    actuation_payload = {
        "command": command,
        "recommended_green_split_sec": target_green,
        "metering_inflow_rate": inflow_metering,
        "target_wait_time": round(target_wait, 2)
    }

    try:
        requests.put(
            f"{DITTO_URL}/things/{thing_id}/features/actuation_control/desiredProperties",
            json=actuation_payload,
            auth=(AUTH_USER, AUTH_PASS),
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        print(f"[ACTUATE DISPATCH] Desired state published to {thing_id}:")
        print(f"                   Directive: {command} | Green Split: {target_green}s | Metering: {inflow_metering}")
    except Exception as e:
        print(f"[ERROR] Failed to dispatch actuation command: {e}")


def on_error(ws, error):
    print(f"[WS ERROR] {error}")


def on_close(ws, close_status_code, close_msg):
    print(f"[WS CLOSED] Code: {close_status_code}, Message: {close_msg}")


def main():
    ws_url = f"ws://{DITTO_HOST}/ws/2"
    ws = websocket.WebSocketApp(
        ws_url,
        header={"Authorization": AUTH_HEADER},
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()


if __name__ == "__main__":
    main()