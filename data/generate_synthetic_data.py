"""
generate_synthetic_data.py
---------------------------------------------------------------
Generates synthetic longitudinal voice-acoustics datasets with configurable
difficulty scenarios (Easy, Moderate, Hard, Ambiguous) and explicit,
ground-truth patient-specific change events.

Supports:
- Class overlap & subtle disease boundaries
- Realistic within-patient variability & temporal drift
- Injected recording noise, missingness, and outliers
- Explicit ground-truth change tracking:
  * known_change_event (bool)
  * change_magnitude (float)
  * change_type (str: 'progression_jump', 'acute_deterioration', 'stable')
  * change_start_visit (int)
"""
import os
import argparse
import datetime
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from src.schemas import ScenarioConfig

SCHEMA_COLUMNS = [
    "patient_id", "visit_id", "visit_date", "group",
    "mdvp_fo", "mdvp_fhi", "mdvp_flo",
    "jitter_pct", "jitter_abs", "rap", "ppq", "ddp",
    "shimmer", "shimmer_db", "apq3", "apq5", "apq11", "dda",
    "nhr", "hnr", "rpde", "dfa", "ppe", "spread1", "spread2", "d2",
    "signal_quality", "updrs_like", "label",
    "known_change_event", "change_magnitude", "change_type", "change_start_visit"
]

SCENARIOS: Dict[str, ScenarioConfig] = {
    "easy": ScenarioConfig(
        scenario_name="easy", class_overlap=0.02, noise_level=0.03,
        temporal_variability=0.04, missingness_rate=0.01, outlier_rate=0.01,
        contradiction_rate=0.02, longitudinal_strength=0.35, assessment_noise=0.02, seed=42
    ),
    "moderate": ScenarioConfig(
        scenario_name="moderate", class_overlap=0.15, noise_level=0.10,
        temporal_variability=0.08, missingness_rate=0.04, outlier_rate=0.03,
        contradiction_rate=0.08, longitudinal_strength=0.25, assessment_noise=0.05, seed=42
    ),
    "hard": ScenarioConfig(
        scenario_name="hard", class_overlap=0.30, noise_level=0.20,
        temporal_variability=0.15, missingness_rate=0.08, outlier_rate=0.06,
        contradiction_rate=0.18, longitudinal_strength=0.18, assessment_noise=0.10, seed=42
    ),
    "ambiguous": ScenarioConfig(
        scenario_name="ambiguous", class_overlap=0.45, noise_level=0.25,
        temporal_variability=0.22, missingness_rate=0.12, outlier_rate=0.10,
        contradiction_rate=0.30, longitudinal_strength=0.12, assessment_noise=0.15, seed=42
    ),
}


def _acoustic_base(is_pd: bool, severity: float, config: ScenarioConfig, rng: np.random.Generator) -> Dict[str, float]:
    """Generates an acoustic feature vector based on disease severity and noise config."""
    sev = float(np.clip(severity, 0.0, 1.0))
    noise = config.noise_level

    # Healthy voices vs PD voices with controlled class overlap
    pd_effect = (1.0 - config.class_overlap) * is_pd + config.class_overlap * (1.0 - is_pd)
    effective_sev = pd_effect * sev

    fo_mean = 154.0 - 25.0 * effective_sev
    fo = float(rng.normal(fo_mean, 18.0 * (1.0 + noise)))
    fhi = float(fo + abs(rng.normal(40.0, 15.0)))
    flo = float(fo - abs(rng.normal(30.0, 10.0)))

    jitter_base = 0.003 + 0.012 * effective_sev
    jitter_pct = float(max(0.0005, rng.normal(jitter_base, 0.002 * (1.0 + noise))))
    jitter_abs = float(jitter_pct * max(50.0, fo) * 1e-4)
    rap = float(jitter_pct * rng.uniform(0.42, 0.58))
    ppq = float(jitter_pct * rng.uniform(0.42, 0.58))
    ddp = float(rap * 3.0)

    shimmer_base = 0.020 + 0.065 * effective_sev
    shimmer = float(max(0.005, rng.normal(shimmer_base, 0.010 * (1.0 + noise))))
    shimmer_db = float(shimmer * rng.uniform(8.5, 11.5))
    apq3 = float(shimmer * rng.uniform(0.32, 0.48))
    apq5 = float(shimmer * rng.uniform(0.38, 0.52))
    apq11 = float(shimmer * rng.uniform(0.52, 0.68))
    dda = float(apq3 * 3.0)

    nhr = float(max(0.002, rng.normal(0.015 + 0.055 * effective_sev, 0.010 * (1.0 + noise))))
    hnr = float(max(4.0, rng.normal(24.0 - 10.0 * effective_sev, 3.0 * (1.0 + noise))))

    rpde = float(np.clip(rng.normal(0.40 + 0.25 * effective_sev, 0.06 * (1.0 + noise)), 0.10, 0.95))
    dfa = float(np.clip(rng.normal(0.68 - 0.08 * effective_sev, 0.05 * (1.0 + noise)), 0.35, 0.90))
    ppe = float(np.clip(rng.normal(0.15 + 0.35 * effective_sev, 0.08 * (1.0 + noise)), 0.01, 0.98))
    spread1 = float(rng.normal(-6.5 + 3.0 * effective_sev, 1.0 * (1.0 + noise)))
    spread2 = float(rng.normal(0.20 + 0.16 * effective_sev, 0.05 * (1.0 + noise)))
    d2 = float(rng.normal(2.30 + 0.70 * effective_sev, 0.30 * (1.0 + noise)))

    # Base signal quality
    signal_quality = float(np.clip(rng.normal(0.88 - 0.15 * noise, 0.10), 0.10, 1.0))

    return dict(
        mdvp_fo=fo, mdvp_fhi=fhi, mdvp_flo=flo,
        jitter_pct=jitter_pct, jitter_abs=jitter_abs, rap=rap, ppq=ppq, ddp=ddp,
        shimmer=shimmer, shimmer_db=shimmer_db, apq3=apq3, apq5=apq5, apq11=apq11, dda=dda,
        nhr=nhr, hnr=hnr, rpde=rpde, dfa=dfa, ppe=ppe,
        spread1=spread1, spread2=spread2, d2=d2,
        signal_quality=signal_quality,
    )


def generate(
    n_healthy: int = 30,
    n_pd: int = 30,
    scenario: str = "moderate",
    config: Optional[ScenarioConfig] = None,
    out_path: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic longitudinal data with explicit ground-truth event tracking."""
    cfg = config or SCENARIOS.get(scenario, SCENARIOS["moderate"])
    if seed != cfg.seed:
        cfg.seed = seed
    rng = np.random.default_rng(cfg.seed)

    rows = []
    base_date = datetime.date(2023, 1, 1)

    # 1. Healthy cohort
    for pid_idx in range(1, n_healthy + 1):
        pid = f"H{pid_idx:03d}"
        n_visits = int(rng.integers(4, 9))
        patient_base_f0 = float(rng.normal(160.0, 15.0))
        patient_noise_bias = float(rng.normal(0.0, cfg.temporal_variability))

        for v in range(1, n_visits + 1):
            visit_date = (base_date + datetime.timedelta(days=(v - 1) * 30)).strftime("%Y-%m-%d")
            # Natural fluctuation
            fluct = float(rng.normal(0.0, cfg.temporal_variability))
            feat = _acoustic_base(is_pd=False, severity=max(0.0, fluct), config=cfg, rng=rng)
            feat["mdvp_fo"] += (patient_base_f0 - 154.0)

            # Injected missingness / outliers / low quality
            if rng.random() < cfg.missingness_rate:
                feat["jitter_pct"] = np.nan
            if rng.random() < cfg.outlier_rate:
                feat["shimmer"] *= rng.choice([0.1, 3.5])
            if rng.random() < cfg.noise_level * 0.5:
                feat["signal_quality"] = float(np.clip(rng.normal(0.35, 0.1), 0.05, 0.54))

            feat.update(
                patient_id=pid,
                visit_id=v,
                visit_date=visit_date,
                group="healthy",
                updrs_like=float(max(0.0, rng.normal(2.5, 1.2))),
                label=0,
                known_change_event=0,
                change_magnitude=0.0,
                change_type="stable",
                change_start_visit=-1,
            )
            rows.append(feat)

    # 2. PD cohort
    for pid_idx in range(1, n_pd + 1):
        pid = f"P{pid_idx:03d}"
        n_visits = int(rng.integers(4, 9))
        base_sev = float(rng.uniform(0.20, 0.55))
        progression_rate = float(rng.uniform(0.015, 0.045))
        
        # Determine whether and when a meaningful change event occurs
        has_jump = bool(n_visits >= 4 and rng.random() < 0.70)
        jump_visit = int(rng.integers(3, n_visits + 1)) if has_jump else -1
        jump_magnitude = float(rng.uniform(0.20, 0.40)) if has_jump else 0.0

        patient_base_f0 = float(rng.normal(148.0, 16.0))

        current_sev = base_sev
        for v in range(1, n_visits + 1):
            visit_date = (base_date + datetime.timedelta(days=(v - 1) * 30)).strftime("%Y-%m-%d")
            
            # Linear progression
            current_sev += progression_rate
            
            is_jump_now = False
            change_type = "stable"
            if has_jump and v >= jump_visit:
                current_sev += (jump_magnitude if v == jump_visit else jump_magnitude * 0.9)
                is_jump_now = True
                change_type = "progression_jump"

            eff_sev = float(np.clip(current_sev, 0.0, 1.0))
            feat = _acoustic_base(is_pd=True, severity=eff_sev, config=cfg, rng=rng)
            feat["mdvp_fo"] += (patient_base_f0 - 154.0)

            # Scenario perturbations
            if rng.random() < cfg.missingness_rate:
                feat["hnr"] = np.nan
            if rng.random() < cfg.outlier_rate:
                feat["jitter_abs"] *= 4.0
            if rng.random() < cfg.contradiction_rate:
                # Contradictory signal: e.g. low jitter but high severity
                feat["jitter_pct"] = 0.002
                feat["shimmer"] = 0.015
            if rng.random() < 0.15:
                # Poor signal quality recording
                feat["signal_quality"] = float(np.clip(rng.normal(0.38, 0.09), 0.05, 0.54))

            updrs = float(8.0 + eff_sev * 32.0 + rng.normal(0.0, 2.0))

            feat.update(
                patient_id=pid,
                visit_id=v,
                visit_date=visit_date,
                group="pd",
                updrs_like=updrs,
                label=1,
                known_change_event=1 if is_jump_now else 0,
                change_magnitude=jump_magnitude if is_jump_now else 0.0,
                change_type=change_type,
                change_start_visit=jump_visit if has_jump else -1,
            )
            rows.append(feat)

    df = pd.DataFrame(rows)[SCHEMA_COLUMNS]
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic longitudinal Parkinson's acoustic data.")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="moderate", help="Difficulty scenario")
    parser.add_argument("--n_healthy", type=int, default=30, help="Number of healthy control patients")
    parser.add_argument("--n_pd", type=int, default=30, help="Number of Parkinson's patients")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    out_csv = args.out or os.path.join(os.path.dirname(__file__), "patients_visits.csv")
    generated_df = generate(
        n_healthy=args.n_healthy,
        n_pd=args.n_pd,
        scenario=args.scenario,
        seed=args.seed,
        out_path=out_csv,
    )
    print(f"Generated {len(generated_df)} visits across {generated_df['patient_id'].nunique()} patients [scenario: {args.scenario}] -> {out_csv}")
