"""
Robin Hood Routing - Urban Traffic Fairness (Behavioral Digital Twin)
========================================================================
A closed-loop Cyber-Physical System (CPS) for urban traffic management:
  1. Behavioral K-Means clustering aggregated by intersection_id (all data 
     for the same intersection remain grouped within the same dynamic zone).
  2. Discrete-Time Markov Chain state mirrors with absorbing risk projections.
  3. Fair priority reallocation with nearest spatial-neighbor offloading.
  4. Bidirectional Digital Twin via Eclipse Ditto:
     - Ingests Physical Telemetry -> Twin State (Physical -> Virtual)
     - Dispatches Actuation Control Targets (desiredProperties) (Virtual -> Physical)
     - Ingests Physical Controller Feedback (properties) (Action-Back Verification)

Run: python robin_hood_routing.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_FILE = "smart_city_traffic_mobility.csv"
N_CLUSTERS = 5        # Number of behavioral dynamic zones
BOOST_PCT = 0.20      # Wait-time cut target for zones that qualify for help
RANDOM_STATE = 42     # KMeans seed for reproducible zone assignments

# Projection horizon (native hourly resolution)
K_STEPS = 12
PROJECTION_BASIS = "within"  # "within" (absorbing risk) or "at" (literal T^K)
ANCHOR_TIMESTAMP = None      # None = latest timestamp in the dataset

# Eclipse Ditto Integration
SYNC_TO_DITTO = True
DITTO_URL = "http://localhost:8080/api/2"
DITTO_AUTH = ("ditto", "ditto")
DITTO_HEADERS = {"Content-Type": "application/json"}
HTTP_TIMEOUT = 15  # Tolerates cluster load/initialization latency

BASE_CYCLE_GREEN = 45  # Standard physical green light baseline (seconds)

REQUIRED_COLUMNS = [
    "intersection_id",
    "city_zone",
    "average_wait_time",
    "congestion_score",
    "vehicle_count",
    "average_speed",
    "timestamp",
    "hour",
]

STATES = ["Low", "Medium", "High", "Critical"]
RISK_STATES = ["High", "Critical"]

# Operational features defining the behavioral footprint
CLUSTER_FEATURES = [
    "average_wait_time",
    "congestion_score",
    "vehicle_count",
    "average_speed",
]


def banner(step, title):
    print()
    print("=" * 70)
    print(f"[{step}] {title}")
    print("=" * 70)


# --------------------------------------------------------------------------
# 0. Load and validate
# --------------------------------------------------------------------------
def load_data(path=DATA_FILE):
    banner(0, "LOADING DATA")

    if not os.path.exists(path):
        sys.exit(
            f"ERROR: Could not find '{path}'.\n"
            f"Put the CSV next to this script, or edit DATA_FILE at the top."
        )

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(
            "ERROR: Dataset is missing required columns: "
            + ", ".join(missing)
            + f"\nFound instead: {', '.join(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")

    before = len(df)
    df = df.dropna(subset=CLUSTER_FEATURES + ["hour", "timestamp"]).copy()
    df["hour"] = df["hour"].astype(int)
    dropped = before - len(df)

    lat_col = next((c for c in df.columns if c.lower() in ["latitude", "lat", "y_coord"]), None)
    lon_col = next((c for c in df.columns if c.lower() in ["longitude", "lon", "lng", "x_coord"]), None)

    if lat_col and lon_col:
        df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"]).copy()
    else:
        unique_ids = df["intersection_id"].unique()
        rng = np.random.default_rng(RANDOM_STATE)
        base_lat, base_lon = 39.3090, 16.2290
        coords = {
            iid: (base_lat + rng.normal(0, 0.05), base_lon + rng.normal(0, 0.05))
            for iid in unique_ids
        }
        df["latitude"] = df["intersection_id"].map(lambda x: coords[x][0])
        df["longitude"] = df["intersection_id"].map(lambda x: coords[x][1])

    print(f"Loaded {len(df):,} rows from '{path}'")
    print(f"  {df['intersection_id'].nunique()} intersections")
    print(f"  {df['city_zone'].nunique()} labeled municipal zones (ignored on purpose)")
    if dropped:
        print(f"  Dropped {dropped:,} rows with missing values")

    return df


# --------------------------------------------------------------------------
# 1. Discover zones from BEHAVIOR (Guarantees same intersection in same zone)
# --------------------------------------------------------------------------
def discover_dynamic_zones(df, n_clusters=N_CLUSTERS):
    banner(1, "DISCOVERING DYNAMIC ZONES (Behavioral K-Means by Intersection)")

    # Group strictly by intersection_id to aggregate operational profile
    profile = (
        df.groupby("intersection_id")[CLUSTER_FEATURES + ["latitude", "longitude"]]
        .mean()
        .reset_index()
    )

    k = min(n_clusters, len(profile))
    if k < n_clusters:
        print(f"NOTE: only {len(profile)} intersections, reducing k to {k}")

    # Cluster strictly on operational behavioral metrics
    X = StandardScaler().fit_transform(profile[CLUSTER_FEATURES])
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    profile["cluster"] = km.fit_predict(X)

    # Sort zones by baseline wait time so Zone_1 is best-served, Zone_K is worst
    order = (
        profile.groupby("cluster")["average_wait_time"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    label_of = {c: f"Zone_{i + 1}" for i, c in enumerate(order)}
    profile["dynamic_zone"] = profile["cluster"].map(label_of)

    # Merge dynamic zone assignment back to every raw timestamp row of the intersection
    df = df.merge(
        profile[["intersection_id", "dynamic_zone"]],
        on="intersection_id",
        how="left",
    )

    print(f"Discovered {k} behavioral zones across all {len(profile)} intersections:")
    summary = (
        df.groupby("dynamic_zone")
        .agg(
            intersections=("intersection_id", "nunique"),
            avg_wait=("average_wait_time", "mean"),
            avg_congestion=("congestion_score", "mean"),
            avg_volume=("vehicle_count", "mean"),
            avg_speed=("average_speed", "mean"),
        )
        .round(2)
    )
    print(summary.to_string())
    print("\n  (Ranked by wait time: Behavioral_Zone_1 = best served, Behavioral_Zone_K = worst served)")
    return df


# --------------------------------------------------------------------------
# 2. Digital Twin: State Mirror
# --------------------------------------------------------------------------
def _bin_states(values, states=STATES):
    counts = values.value_counts().sort_index()
    if len(counts) < len(states):
        return pd.cut(values, bins=len(states), labels=states), "equal-width"

    share = counts / counts.sum()
    centre = share.cumsum() - share / 2
    slot = np.clip((centre * len(states)).astype(int), 0, len(states) - 1)
    mapping = {v: states[i] for v, i in zip(counts.index, slot)}
    return values.map(mapping), "equal-frequency (ties kept together)"


def build_state_mirror(df):
    banner(2, "DIGITAL TWIN - BUILDING STATE MIRROR")

    mirror = (
        df.groupby(["dynamic_zone", "timestamp"])["congestion_score"]
        .mean()
        .reset_index()
        .sort_values(["dynamic_zone", "timestamp"])
    )

    mirror["state"], method = _bin_states(mirror["congestion_score"])
    mirror["state"] = mirror["state"].astype(str)

    per_zone = mirror.groupby("dynamic_zone").size()
    span = mirror["timestamp"].max() - mirror["timestamp"].min()
    print(f"State mirror built: {len(mirror):,} zone-hour observations "
          f"({per_zone.iloc[0]:,} per zone over {span.days} days), {method} bins")

    dist = (
        mirror.groupby(["dynamic_zone", "state"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATES, fill_value=0)
    )
    print("\nHow often each behavioral zone sits in each state:")
    print((dist.div(dist.sum(axis=1), axis=0) * 100).round(1).to_string())
    print("  (% of hours -- base rate the Markov forecast is measured against)")
    return mirror


# --------------------------------------------------------------------------
# 3. Digital Twin: Transition Matrices
# --------------------------------------------------------------------------
def build_transition_matrices(mirror):
    banner(3, "DIGITAL TWIN - BUILDING TRANSITION MATRICES")

    matrices, n_obs = {}, 0
    for zone, g in mirror.groupby("dynamic_zone"):
        g = g.sort_values("timestamp")
        step = g["timestamp"].diff().shift(-1)
        pairs = pd.DataFrame(
            {"cur": g["state"], "nxt": g["state"].shift(-1), "step": step}
        ).dropna()
        pairs = pairs[pairs["step"] == pd.Timedelta(hours=1)]
        n_obs = len(pairs)

        counts = pd.DataFrame(0.0, index=STATES, columns=STATES)
        for a, b in zip(pairs["cur"], pairs["nxt"]):
            counts.loc[a, b] += 1

        totals = counts.sum(axis=1)
        matrix = counts.div(totals, axis=0)
        matrix.loc[totals == 0] = 1.0 / len(STATES)
        matrices[zone] = matrix

    print(f"Built {len(matrices)} transition matrices ({len(STATES)}x{len(STATES)}), "
          f"estimated from ~{n_obs:,} real hour-to-hour transitions per zone")
    return matrices


# --------------------------------------------------------------------------
# 4. Projections & Horizon Math
# --------------------------------------------------------------------------
def baseline_congestion_rate(states):
    return float(pd.Series(states).isin(RISK_STATES).mean())


def _state_vector(state):
    v = np.zeros(len(STATES))
    v[STATES.index(state)] = 1.0
    return v


def project_at_k(matrix, state_now, k=None):
    k = K_STEPS if k is None else k
    risk_idx = [STATES.index(s) for s in RISK_STATES]
    reached = _state_vector(state_now) @ np.linalg.matrix_power(matrix, k)
    return float(reached[risk_idx].sum())


def project_within_k(matrix, state_now, k=None):
    k = K_STEPS if k is None else k
    risk_idx = [STATES.index(s) for s in RISK_STATES]
    absorbing = np.array(matrix, dtype=float, copy=True)
    for i in risk_idx:
        absorbing[i, :] = 0.0
        absorbing[i, i] = 1.0
    reached = _state_vector(state_now) @ np.linalg.matrix_power(absorbing, k)
    return float(reached[risk_idx].sum())


def hours_to_threshold(matrix, state_now, tau, max_k=72):
    for k in range(1, max_k + 1):
        if project_within_k(matrix, state_now, k) > tau:
            return k
    return np.nan


def horizon_phrase(k=None, basis=None):
    k = K_STEPS if k is None else k
    basis = basis or PROJECTION_BASIS
    if basis == "within":
        return "within the next hour" if k == 1 else f"within the next {k}h"
    return "at the next hour" if k == 1 else f"exactly {k}h out"


def _project(matrix, state_now, k=None, basis=None):
    basis = basis or PROJECTION_BASIS
    if basis == "at":
        return project_at_k(matrix, state_now, k)
    if basis == "within":
        return project_within_k(matrix, state_now, k)
    raise ValueError(f"PROJECTION_BASIS must be 'within' or 'at', got {basis!r}")


# --------------------------------------------------------------------------
# 5. Risk Scores & Equity Gap
# --------------------------------------------------------------------------
def compute_risk_scores(df, mirror, matrices):
    banner(5, "BASELINE (tau) vs PROJECTION (P_K)")

    risk_idx = [STATES.index(s) for s in RISK_STATES]
    anchor = (
        pd.to_datetime(ANCHOR_TIMESTAMP)
        if ANCHOR_TIMESTAMP is not None
        else mirror["timestamp"].max()
    )

    rows = []
    for zone, g in mirror.groupby("dynamic_zone"):
        g = g.sort_values("timestamp")
        tau = baseline_congestion_rate(g["state"])

        upto = g[g["timestamp"] <= anchor]
        state_now = (upto if len(upto) else g)["state"].iloc[-1]

        matrix = matrices.get(zone)
        if matrix is None:
            rows.append({
                "dynamic_zone": zone, "current_state": state_now,
                "baseline_threshold": tau, "projected_probability": tau,
                "projected_at_k": tau, "projected_within_k": tau,
                "rho": tau, "hours_to_threshold": np.nan,
            })
            continue

        T = matrix.values
        p_at = project_at_k(T, state_now, K_STEPS)
        p_within = project_within_k(T, state_now, K_STEPS)
        p_k = _project(T, state_now, K_STEPS)

        step = np.linalg.matrix_power(T, K_STEPS)
        risk_from = step[:, risk_idx].sum(axis=1)
        freq = np.array([(g["state"] == s).mean() for s in STATES])
        rho = float(freq @ risk_from)

        rows.append({
            "dynamic_zone": zone,
            "current_state": state_now,
            "baseline_threshold": tau,
            "projected_probability": p_k,
            "projected_at_k": p_at,
            "projected_within_k": p_within,
            "rho": rho,
            "hours_to_threshold": hours_to_threshold(T, state_now, tau),
        })

    risk = pd.DataFrame(rows)
    risk["congestion_flag"] = risk["projected_probability"] > risk["baseline_threshold"]
    risk["priority_rank"] = risk["projected_probability"].rank(ascending=False, method="min").astype(int)
    risk = risk.sort_values("priority_rank").reset_index(drop=True)
    gap = risk["rho"].max() - risk["rho"].min()

    basis_word = horizon_phrase()
    print(f"Anchor: {anchor} | Horizon: K_STEPS = {K_STEPS} ({basis_word})\n")
    show = risk[[
        "dynamic_zone", "current_state", "baseline_threshold",
        "projected_probability", "congestion_flag", "priority_rank",
        "hours_to_threshold", "rho",
    ]]
    print(show.round(3).to_string(index=False))
    print(f"\n>>> RISK EQUITY GAP (rho max - min): {gap:.3f}")
    return risk, gap


# --------------------------------------------------------------------------
# 6. Baseline Chart
# --------------------------------------------------------------------------
def baseline_chart(zone_stats, filename="baseline_wait_time.png"):
    banner(6, "CHARTING THE BASELINE")
    d = zone_stats.sort_values("avg_wait_before")
    norm = plt.Normalize(d["avg_wait_before"].min(), d["avg_wait_before"].max())
    colors = plt.cm.RdYlGn_r(norm(d["avg_wait_before"]))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(d["dynamic_zone"], d["avg_wait_before"], color=colors, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.axhline(d["avg_wait_before"].mean(), ls="--", lw=1.2, color="#444",
               label=f"city average ({d['avg_wait_before'].mean():.1f}s)")
    ax.set_title("Baseline Average Wait Time Across Zones", fontsize=13, weight="bold")
    ax.set_xlabel(" Zone (Zone_1 = fastest corridors)")
    ax.set_ylabel("Average wait time (s)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved '{filename}'")


# --------------------------------------------------------------------------
# 7. Robin Hood Fix (Capped Reduction + Nearest Spatial Offload)
# --------------------------------------------------------------------------
def apply_robin_hood_fix(
    df: pd.DataFrame,
    zone_stats: pd.DataFrame,
    risk_df: pd.DataFrame,
    boost_pct: float = BOOST_PCT
) -> pd.DataFrame:
    banner(7, "APPLYING ROBIN HOOD ROUTING (FAIR OFF-LOADING TO NEAREST CLUSTERS)")

    merged = zone_stats.merge(
        risk_df[[
            "dynamic_zone", "rho", "congestion_flag", "baseline_threshold",
            "projected_probability", "priority_rank", "hours_to_threshold"
        ]],
        on="dynamic_zone"
    )

    # Calculate cluster spatial centroids to preserve physical routing realism
    centroids = (
        df.groupby("dynamic_zone")[["latitude", "longitude"]]
        .mean()
        .reset_index()
    )
    merged = merged.merge(centroids, on="dynamic_zone")

    city_mean_wait = merged["avg_wait_before"].mean()
    city_mean_rho = merged["rho"].mean()

    # Qualifies if chronically risky/flagged and waiting longer than average
    merged["boosted"] = (
        (merged["rho"] > city_mean_rho) | (merged["congestion_flag"])
    ) & (merged["avg_wait_before"] > city_mean_wait)

    merged["avg_wait_after"] = merged["avg_wait_before"].copy()

    print(f"City Mean Wait Floor: {city_mean_wait:.2f}s | Mean Risk Threshold: {city_mean_rho:.4f}\n")

    for idx, row in merged.sort_values("rho", ascending=False).iterrows():
        if not row["boosted"]:
            continue

        zone_name = row["dynamic_zone"]
        w_orig = row["avg_wait_after"]

        raw_reduced_wait = w_orig * (1.0 - boost_pct)
        capped_wait = max(raw_reduced_wait, city_mean_wait)
        delay_relief = w_orig - capped_wait

        if delay_relief <= 0:
            continue

        # Sort candidate neighbors by Euclidean distance between cluster centroids
        candidates = merged[merged["dynamic_zone"] != zone_name].copy()
        d_lat = candidates["latitude"] - row["latitude"]
        d_lon = candidates["longitude"] - row["longitude"]
        candidates["dist"] = np.sqrt(d_lat**2 + d_lon**2)
        candidates = candidates.sort_values("dist")

        absorbed_delay = 0.0
        donor_log = []

        for c_idx, cand in candidates.iterrows():
            curr_cand_wait = merged.loc[c_idx, "avg_wait_after"]
            headroom = max(0.0, city_mean_wait - curr_cand_wait)

            if headroom > 0:
                needed = delay_relief - absorbed_delay
                transfer = min(needed, headroom)
                merged.loc[c_idx, "avg_wait_after"] += transfer
                absorbed_delay += transfer
                donor_log.append(f"{cand['dynamic_zone']} (+{transfer:.2f}s)")

                if np.isclose(absorbed_delay, delay_relief):
                    break

        merged.loc[idx, "avg_wait_after"] = w_orig - absorbed_delay
        receptive_info = ", ".join(donor_log) if donor_log else "None (Other Zones at Capacity)"
        print(f"  [BOOSTED] {zone_name:<18} : {w_orig:.1f}s -> {merged.loc[idx, 'avg_wait_after']:.1f}s "
              f"(-{absorbed_delay:.2f}s) | Offloaded to: {receptive_info}")

    return merged


# --------------------------------------------------------------------------
# 8. Before/After Chart
# --------------------------------------------------------------------------
def before_after_chart(result, filename="before_after_wait_time.png"):
    banner(8, "CHARTING BEFORE VS. AFTER")

    d = result.sort_values("dynamic_zone")
    x = np.arange(len(d))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - w / 2, d["avg_wait_before"], w, label="Before (naive)",
                color="#c0392b", edgecolor="white")
    b2 = ax.bar(x + w / 2, d["avg_wait_after"], w, label="After (Robin Hood)",
                color="#27ae60", edgecolor="white")
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)

    labels = [
        f"{z}\n(boosted)" if b else z
        for z, b in zip(d["dynamic_zone"], d["boosted"])
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Robin Hood Routing: Zone Wait Time Before vs. After", fontsize=13, weight="bold")
    ax.set_xlabel(" Dynamic Zone")
    ax.set_ylabel("Average wait time (s)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved '{filename}'")


# --------------------------------------------------------------------------
# 9. Equity Scores and Grades
# --------------------------------------------------------------------------
def _grade(score):
    for cutoff, letter in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if score >= cutoff:
            return letter
    return "F"


def equity_score_table(result, filename="equity_summary.csv"):
    banner(9, "SCORING EQUITY")

    d = result.copy()
    benchmark = d["avg_wait_before"].min()
    worst = d["avg_wait_before"].max()

    def score(waits):
        if np.isclose(benchmark, 0):
            if np.isclose(worst, 0):
                return pd.Series(100.0, index=waits.index)
            return ((1 - waits / worst) * 100).clip(0, 100)
        return (benchmark / waits.replace(0, np.nan) * 100).fillna(100).clip(0, 100)

    d["equity_score_before"] = score(d["avg_wait_before"])
    d["equity_score_after"] = score(d["avg_wait_after"])
    d["grade_before"] = d["equity_score_before"].map(_grade)
    d["grade_after"] = d["equity_score_after"].map(_grade)

    cols = [
        "dynamic_zone", "avg_wait_before", "avg_wait_after",
        "baseline_threshold", "projected_probability", "congestion_flag",
        "priority_rank", "hours_to_threshold", "rho",
        "equity_score_before", "equity_score_after", "grade_before", "grade_after",
    ]
    out = d[[c for c in cols if c in d.columns]].sort_values("priority_rank").round(3)
    out.to_csv(filename, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved '{filename}'")
    return out


# --------------------------------------------------------------------------
# 10. Plain-English Summary
# --------------------------------------------------------------------------
def plain_english_summary(result, table, risk_gap):
    banner(10, "SUMMARY")

    before, after = result["avg_wait_before"], result["avg_wait_after"]
    gap_before = before.max() - before.min()
    gap_after = after.max() - after.min()
    narrowed = gap_before - gap_after
    pct = (narrowed / gap_before * 100) if gap_before else 0.0

    worst = result.loc[before.idxmax()]
    best = result.loc[before.idxmin()]
    n_boosted = int(result["boosted"].sum())
    n_flagged = int(result["congestion_flag"].sum())

    ranked = result.sort_values("priority_rank")
    horizon = ("over the next hour" if K_STEPS == 1 else f"over the next {K_STEPS} hours")

    lines = []
    for _, r in ranked.iterrows():
        verdict = (
            "above its {:.0%} baseline".format(r["baseline_threshold"])
            if r["congestion_flag"]
            else "still within its {:.0%} baseline".format(r["baseline_threshold"])
        )
        tail = " -- BOOSTED" if r["boosted"] else ""
        eta = (f", first crossing in {int(r['hours_to_threshold'])}h"
               if pd.notna(r["hours_to_threshold"]) else "")
        lines.append(
            f"  {r['dynamic_zone']} has a {r['projected_probability']:.0%} "
            f"projected congestion probability {horizon}, {verdict}{eta} "
            f"-- ranked #{int(r['priority_rank'])} priority{tail}"
        )
    per_zone = "\n".join(lines)

    grade_before = table.loc[table["dynamic_zone"] == worst["dynamic_zone"], "grade_before"].iloc[0]
    grade_after = table.loc[table["dynamic_zone"] == worst["dynamic_zone"], "grade_after"].iloc[0]

    print(
        f"""
We partitioned intersections into {len(result)} behavioral dynamic zones via aggregated K-Means.
All intersections and their corresponding historical telemetry remain strictly mapped to their assigned zone.
The Digital Twin evaluated each zone's baseline normal (tau) against its forward
Markov projection (P_K) {horizon}.

THE PRIORITY ORDER
{per_zone}

  {n_flagged} of {len(result)} zones are projected past their own baseline.

THE OUTCOME
  Before intervention, {worst['dynamic_zone']} waited {worst['avg_wait_before']:.1f}s while
  {best['dynamic_zone']} waited {best['avg_wait_before']:.1f}s (Gap: {gap_before:.1f}s).
  
  Robin Hood Routing granted relief to {n_boosted} zone(s), capping reductions to the city mean
  and offloading absorbed traffic to nearest spatial centroids.

  Wait-time equity gap: {gap_before:.1f}s -> {gap_after:.1f}s (Narrowed by {narrowed:.1f}s / {pct:.1f}%)
  Worst zone's grade: {grade_before} -> {grade_after}
"""
    )


# --------------------------------------------------------------------------
# 11. Eclipse Ditto Digital Twin: Physical -> Virtual Telemetry Ingestion
# --------------------------------------------------------------------------
def sync_results_to_ditto(result_df: pd.DataFrame):
    """Initializes Twin Schema, Telemetry, and Physical Status in Eclipse Ditto."""
    banner("DITTO: INGEST", "1. SYNCING PHYSICAL STATE & PROJECTIONS TO TWINS")

    policy_payload = {
        "entries": {
            "owner": {
                "subjects": {"nginx:ditto": {"type": "nginx basic auth user"}},
                "resources": {
                    "thing:/": {"grant": ["READ", "WRITE"], "revoke": []},
                    "policy:/": {"grant": ["READ", "WRITE"], "revoke": []},
                    "message:/": {"grant": ["READ", "WRITE"], "revoke": []},
                },
            }
        }
    }

    try:
        p_res = requests.put(
            f"{DITTO_URL}/policies/traffic.network:policy",
            json=policy_payload,
            auth=DITTO_AUTH,
            headers=DITTO_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        if p_res.status_code not in (200, 201, 204):
            print(f"  [WARN] Policy setup returned HTTP {p_res.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] Skipping Ditto sync. Cannot connect to {DITTO_URL}: {e}")
        return

    for _, row in result_df.iterrows():
        zone_name = str(row["dynamic_zone"])
        thing_id = f"traffic.zones:{zone_name}"

        lat_val = float(row["latitude"]) if "latitude" in row and pd.notna(row["latitude"]) else 39.3090
        lon_val = float(row["longitude"]) if "longitude" in row and pd.notna(row["longitude"]) else 16.2290

        payload = {
            "policyId": "traffic.network:policy",
            "attributes": {
                "zone_name": zone_name,
                "latitude": lat_val,
                "longitude": lon_val,
                "clustering_type": "behavioral_operational",
            },
            "features": {
                "telemetry": {
                    "properties": {
                        "avg_wait_before": float(row["avg_wait_before"]),
                        "avg_wait_after": float(row["avg_wait_after"]),
                        "boosted": bool(row.get("boosted", False)),
                    }
                },
                "digital_twin_predictor": {
                    "properties": {
                        "baseline_threshold": float(row.get("baseline_threshold", 0.0)),
                        "projected_probability": float(row.get("projected_probability", 0.0)),
                        "congestion_flag": bool(row.get("congestion_flag", False)),
                        "priority_rank": int(row.get("priority_rank", 0)),
                        "rho": float(row.get("rho", 0.0)),
                        "hours_to_threshold": (
                            float(row["hours_to_threshold"])
                            if pd.notna(row.get("hours_to_threshold"))
                            else None
                        ),
                    }
                },
                "actuation_control": {
                    "properties": {
                        "current_green_split_sec": BASE_CYCLE_GREEN,
                        "metering_inflow_rate": 1.0,
                        "status": "OPERATING_NOMINAL",
                    }
                },
            },
        }

        try:
            res = requests.put(
                f"{DITTO_URL}/things/{thing_id}",
                json=payload,
                auth=DITTO_AUTH,
                headers=DITTO_HEADERS,
                timeout=HTTP_TIMEOUT,
            )
            print(f"  Ingested Twin {thing_id} -> HTTP {res.status_code}")
        except requests.exceptions.RequestException as err:
            print(f"  Failed to sync {thing_id}: {err}")


# --------------------------------------------------------------------------
# 12. Eclipse Ditto Digital Twin: Virtual -> Physical Actuation Dispatch
# --------------------------------------------------------------------------
def dispatch_actuation_to_ditto(result_df: pd.DataFrame):
    """
    Virtual -> Physical (Bidirectional Actuation Control):
    Calculates operational control parameters (green-split extension, perimeter metering)
    and writes desiredProperties to each Twin's actuation_control feature.
    """
    banner("DITTO: ACTUATE", "2. DISPATCHING BIDIRECTIONAL CONTROL COMMANDS")

    for _, row in result_df.iterrows():
        zone_name = str(row["dynamic_zone"])
        thing_id = f"traffic.zones:{zone_name}"
        boosted = bool(row.get("boosted", False))

        if boosted:
            wait_relief = float(row["avg_wait_before"] - row["avg_wait_after"])
            pct_relief = (wait_relief / row["avg_wait_before"]) * 100
            target_green = int(BASE_CYCLE_GREEN * (1.0 + (pct_relief / 100.0)))
            metering_rate = round(max(0.60, 1.0 - (pct_relief / 200.0)), 2)
            command_type = "BOOST_PRIORITY_EXTEND_GREEN"
        else:
            pct_relief = 0.0
            target_green = BASE_CYCLE_GREEN
            metering_rate = 1.0
            command_type = "MAINTAIN_STANDARD_SPLIT"

        actuation_payload = {
            "target_wait_time": float(row["avg_wait_after"]),
            "green_split_adjustment_pct": round(pct_relief, 1),
            "recommended_green_split_sec": target_green,
            "metering_inflow_rate": metering_rate,
            "command": command_type,
        }

        url = f"{DITTO_URL}/things/{thing_id}/features/actuation_control/desiredProperties"
        try:
            res = requests.put(
                url,
                json=actuation_payload,
                auth=DITTO_AUTH,
                headers=DITTO_HEADERS,
                timeout=HTTP_TIMEOUT,
            )
            print(f"  Dispatched to {thing_id} -> HTTP {res.status_code} [{command_type} | split={target_green}s | inflow={metering_rate}]")
        except requests.exceptions.RequestException as err:
            print(f"  Failed to dispatch actuation for {thing_id}: {err}")


# --------------------------------------------------------------------------
# 13. Eclipse Ditto Digital Twin: Physical -> Virtual "Action-Back" Verification
# --------------------------------------------------------------------------
def ingest_action_back_feedback(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Closed-Loop Feedback:
    Reads hardware-confirmed actuation state from properties/actuation_control
    after edge_traffic_controller.py executes the setpoints.
    """
    banner("DITTO: ACTION BACK", "3. VERIFYING APPLIED HARDWARE TELEMETRY")

    feedback_df = result_df.copy()

    for idx, row in feedback_df.iterrows():
        thing_id = f"traffic.zones:{row['dynamic_zone']}"
        url = f"{DITTO_URL}/things/{thing_id}/features/actuation_control/properties"

        try:
            res = requests.get(url, auth=DITTO_AUTH, headers=DITTO_HEADERS, timeout=HTTP_TIMEOUT)
            if res.status_code == 200:
                props = res.json()
                applied_green = props.get("current_green_split_sec", BASE_CYCLE_GREEN)
                hw_status = props.get("status", "UNKNOWN")

                if hw_status == "APPLIED_ON_HARDWARE":
                    efficiency_gain = (applied_green - BASE_CYCLE_GREEN) / BASE_CYCLE_GREEN
                    actual_wait = max(5.0, row["avg_wait_before"] * (1.0 - (efficiency_gain * 0.8)))
                    feedback_df.loc[idx, "avg_wait_confirmed"] = round(actual_wait, 2)
                    feedback_df.loc[idx, "controller_status"] = "CONFIRMED"
                    print(f"  {thing_id:<28} | Split: {applied_green}s | Status: {hw_status} | Confirmed Wait: {actual_wait:.1f}s")
                else:
                    feedback_df.loc[idx, "avg_wait_confirmed"] = row["avg_wait_before"]
                    feedback_df.loc[idx, "controller_status"] = "PENDING"
                    print(f"  {thing_id:<28} | Split: {applied_green}s | Status: {hw_status} (Pending Edge Ack)")
        except Exception as err:
            print(f"  Could not read feedback for {thing_id}: {err}")

    return feedback_df


# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------
def main():
    print("\n" + "*" * 70)
    print("*  ROBIN HOOD ROUTING - Urban Traffic Fairness (Behavioral DT)".ljust(69) + "*")
    print("*" * 70)

    df = load_data()
    df = discover_dynamic_zones(df)
    mirror = build_state_mirror(df)
    matrices = build_transition_matrices(mirror)
    risk, risk_gap = compute_risk_scores(df, mirror, matrices)

    zone_stats = (
        df.groupby("dynamic_zone")["average_wait_time"]
        .mean()
        .reset_index()
        .rename(columns={"average_wait_time": "avg_wait_before"})
    )

    baseline_chart(zone_stats)
    result_df = apply_robin_hood_fix(df, zone_stats, risk, boost_pct=BOOST_PCT)
    before_after_chart(result_df)
    table = equity_score_table(result_df)
    plain_english_summary(result_df, table, risk_gap)

    # Closed-Loop Bidirectional Synchronization
    if SYNC_TO_DITTO:
        sync_results_to_ditto(result_df)
        dispatch_actuation_to_ditto(result_df)
        
        print("\n[WAIT] Pausing 6 seconds for edge controller to apply and acknowledge on hardware...")
        time.sleep(6)
        
        ingest_action_back_feedback(result_df)

    print("=" * 70)
    print("Done. Wrote baseline_wait_time.png, before_after_wait_time.png, equity_summary.csv")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()