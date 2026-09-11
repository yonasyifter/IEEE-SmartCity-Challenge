"""
edge_traffic_controller.py
==========================
Simulates roadside intersection hardware executing Eclipse Ditto commands.
Acts as the physical Cyber-Physical System (CPS) layer:
  1. Polls desiredProperties from Eclipse Ditto (Virtual -> Physical).
  2. Detects target setpoint changes (green split, metering inflow rate).
  3. Simulates controller relay switching latency.
  4. Acknowledges confirmed hardware state to properties (Action-Back Verification).

Run: python3 edge_traffic_controller.py
"""

import time
import requests

DITTO_URL = "http://localhost:8080/api/2"
AUTH = ("ditto", "ditto")
HEADERS = {"Content-Type": "application/json"}
HTTP_TIMEOUT = 10
POLL_INTERVAL_SEC = 2
ZONES = [f"traffic.zones:Zone_{i}" for i in range(1, 6)]

last_applied_state = {}


def run_edge_controller():
    print("=" * 70)
    print("  [EDGE AGENT] Roadside Traffic Controller Simulator Active")
    print(f"  Target Twin Registry : {DITTO_URL}")
    print(f"  Monitored Sectors    : {', '.join(ZONES)}")
    print("=" * 70)

    while True:
        for thing_id in ZONES:
            desired_url = f"{DITTO_URL}/things/{thing_id}/features/actuation_control/desiredProperties"

            try:
                # 1. Fetch desired actuation targets set by dt_brain_processor
                res = requests.get(desired_url, auth=AUTH, headers=HEADERS, timeout=HTTP_TIMEOUT)

                if res.status_code == 200 and res.json():
                    desired = res.json()
                    command = desired.get("command", "MAINTAIN_STANDARD_SPLIT")
                    target_split = desired.get("recommended_green_split_sec", 45)
                    metering_rate = desired.get("metering_inflow_rate", 1.0)
                    target_wait = desired.get("target_wait_time")

                    # Deduplication check
                    prev_state = last_applied_state.get(thing_id, {})
                    is_new_command = (
                        prev_state.get("split") != target_split or
                        prev_state.get("metering") != metering_rate or
                        prev_state.get("command") != command
                    )

                    if is_new_command:
                        print(f"\n[ACTUATING] << Hardware received directive for {thing_id}")
                        print(f"            Directive   : {command}")
                        print(f"            Green Split : {target_split}s (Baseline: 45s)")
                        print(f"            Inflow Rate : {metering_rate * 100:.0f}%")
                        if target_wait is not None:
                            print(f"            Target Wait : {target_wait:.1f}s")

                        # 2. Simulate hardware relay switching and cycle sync delay
                        time.sleep(0.5)

                        # 3. Action Back: Push verified physical state to properties
                        ack_url = f"{DITTO_URL}/things/{thing_id}/features/actuation_control/properties"
                        ack_payload = {
                            "current_green_split_sec": target_split,
                            "metering_inflow_rate": metering_rate,
                            "status": "APPLIED_ON_HARDWARE",
                            "last_command_executed": command,
                            "hardware_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                        }

                        ack_res = requests.put(ack_url, json=ack_payload, auth=AUTH, headers=HEADERS, timeout=HTTP_TIMEOUT)

                        if ack_res.status_code in (200, 204):
                            print(f"[ACK]       >> Confirmed APPLIED_ON_HARDWARE to {thing_id} (HTTP {ack_res.status_code})")
                            last_applied_state[thing_id] = {
                                "split": target_split,
                                "metering": metering_rate,
                                "command": command
                            }
                        else:
                            print(f"[ERROR]     Failed to send ACK for {thing_id} (HTTP {ack_res.status_code})")

            except requests.exceptions.RequestException:
                pass

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    try:
        run_edge_controller()
    except KeyboardInterrupt:
        print("\n[EDGE AGENT] Terminated by user.")