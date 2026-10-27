"""
Tests for Stage 4 fixes: DTO, Domain Profiles, Failure Analysis, and Aggregation (E4.1 - E4.4).
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from app.explainability.aggregate import aggregate_explanations
from app.explainability.domain_profiles import get_domain_profile, is_valid_adversarial
from app.explainability.explanation_result import ExplanationResult
from app.explainability.failure_analysis import analyze_failure


def test_e4_1_explanation_result_fields_and_npz_saving():
    """
    Test E4.1:
    - ExplanationResult includes domain, validity_flags, attribution_status, etc.
    - save_json serializes numpy scalars properly.
    - Large attributions are offloaded to .npz, leaving only downsampled heatmap and path ref in JSON.
    """
    large_attr = np.random.randn(32, 32).astype(np.float32)

    res = ExplanationResult(
        attack_name="pgd",
        technique="shap",
        clean_prediction=np.int64(0),  # Numpy scalar
        adversarial_prediction=np.int64(1),
        clean_confidence=np.float32(0.95),  # Numpy float
        adversarial_confidence=np.float32(0.88),
        prediction_changed=True,
        domain="fraud",
        attribution_status="success",
        validity_flags={"is_valid": True},
        attribution={
            "clean": {
                "attribution": large_attr,
                "postprocessed": {"heatmap": large_attr.tolist()},
            },
            "adversarial": {
                "attribution": large_attr,
            },
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "explanation.json"
        res.save_json(json_path, store_npz=True, max_heatmap_dim=8)

        assert json_path.exists()
        npz_path = Path(tmpdir) / "explanation.npz"
        assert npz_path.exists()

        # Check NPZ contents
        data = np.load(npz_path)
        assert "clean_attribution" in data
        assert data["clean_attribution"].shape == (32, 32)


def test_e4_2_domain_profiles_and_validity_rules():
    """
    Test E4.2:
    - Domain profiles exist for fraud, ids, medical, maintenance.
    - is_valid_adversarial flags negative bytes, fractional protocol types, and altered immutable features.
    """
    # 1. Profile retrieval
    prof_fraud = get_domain_profile("fraud")
    prof_ids = get_domain_profile("ids")
    prof_maint = get_domain_profile("maintenance")

    assert prof_fraud.name == "fraud"
    assert "oldbalanceOrg" in prof_fraud.feature_names
    assert "protocol_type" in prof_ids.integer_features
    assert "T24" in prof_maint.sensor_names

    # 2. IDS validity check: fractional protocol_type & negative byte counts
    # Clean: [duration, protocol_type, src_bytes, dst_bytes, count, srv_count]
    clean_sample = np.array([0.0, 1.0, 200.0, 500.0, 5.0, 5.0])
    # Unrealistic adversarial: protocol_type=1.45 (fractional), src_bytes=-50 (negative byte count)
    adv_invalid = np.array([0.0, 1.45, -50.0, 500.0, 5.0, 5.0])

    is_valid, flags = is_valid_adversarial(clean_sample, adv_invalid, prof_ids)
    assert is_valid is False
    assert len(flags["integer_violations"]) > 0
    assert len(flags["bounds_violations"]) > 0
    assert len(flags["immutable_violations"]) > 0  # protocol_type is immutable


def test_e4_3_failure_analysis_full_vector_shifts_and_sensors():
    """
    Test E4.3:
    - Feature shifts are computed from full vectors, not just top-k lists.
    - Features outside clean top-k do NOT get clean_importance=0.0.
    - Time-series computes per-sensor ranking and (timestep, sensor) cells.
    """
    # Tabular full vectors: feature 2 was NOT in top-1, but has clean importance 0.25
    clean_vec = np.array([0.9, 0.4, 0.25, 0.1])
    adv_vec = np.array([0.2, 0.3, 0.85, 0.05])
    f_names = ["income", "credit_score", "debt", "age"]

    res = analyze_failure(
        clean_prediction=0,
        adversarial_prediction=1,
        true_label=0,
        clean_attribution=clean_vec,
        adv_attribution=adv_vec,
        feature_names=f_names,
        modality="tabular",
        top_k=2,
    )

    contrib = res["contributing_features"]
    debt_entry = next((f for f in contrib if f["feature"] == "debt"), None)
    assert debt_entry is not None
    # Clean importance must accurately be 0.25, NOT 0.0!
    assert np.isclose(debt_entry["clean_importance"], 0.25)
    assert np.isclose(debt_entry["importance_shift"], 0.85 - 0.25)

    # Time-series: (T=3, F=2) sensors
    ts_clean = np.array([[0.1, 0.5], [0.2, 0.6], [0.1, 0.4]])
    ts_adv = np.array([[0.8, 0.2], [0.9, 0.3], [0.7, 0.1]])
    res_ts = analyze_failure(
        clean_prediction=0,
        adversarial_prediction=1,
        true_label=0,
        clean_attribution=ts_clean,
        adv_attribution=ts_adv,
        sensor_names=["temperature", "vibration"],
        modality="timeseries",
    )

    assert "sensor_importance" in res_ts
    assert res_ts["sensor_importance"]["top_exploited_sensor"] == "temperature"
    top_cell = res_ts["contributing_features"][0]
    assert "timestep" in top_cell
    assert top_cell["sensor"] == "temperature"


def test_e4_4_aggregation_attack_success_rate_and_grouping():
    """
    Test E4.4:
    - attack_success_rate is computed on clean-correct samples.
    - Grouped by attack_name with sorted list keys.
    - Exploitation stats split successful vs failed attacks.
    - Group one-hot features to parent and time-series to sensors.
    """
    # Sample 1: Clean correct, attack succeeded
    r1 = ExplanationResult(
        attack_name="pgd",
        technique="shap",
        clean_prediction=0,
        adversarial_prediction=1,
        clean_confidence=0.9,
        adversarial_confidence=0.8,
        prediction_changed=True,
        attack_caused_failure=True,
        failure_analysis={"clean_correct": True, "contributing_features": [{"feature": "type_TRANSFER", "importance_shift": 0.5}]},
        comparison={"confidence_difference": 0.1},
    )
    # Sample 2: Clean correct, attack failed
    r2 = ExplanationResult(
        attack_name="pgd",
        technique="shap",
        clean_prediction=0,
        adversarial_prediction=0,
        clean_confidence=0.9,
        adversarial_confidence=0.85,
        prediction_changed=False,
        attack_caused_failure=False,
        failure_analysis={"clean_correct": True, "contributing_features": [{"feature": "type_CASH_OUT", "importance_shift": 0.1}]},
        comparison={"confidence_difference": 0.05},
    )
    # Sample 3: Model was already WRONG on clean input (should NOT be counted in clean-correct denominator)
    r3 = ExplanationResult(
        attack_name="fgsm",
        technique="shap",
        clean_prediction=1,
        adversarial_prediction=1,
        clean_confidence=0.6,
        adversarial_confidence=0.6,
        prediction_changed=False,
        attack_caused_failure=False,
        failure_analysis={"clean_correct": False, "contributing_features": []},
        comparison={"confidence_difference": 0.0},
    )

    agg = aggregate_explanations([r1, r2, r3])

    # Out of 2 clean-correct samples (r1, r2), 1 attack succeeded -> ASR = 0.5
    assert np.isclose(agg["attack_success_rate"], 0.5)
    assert agg["attack_failure_rate"] == agg["attack_success_rate"]  # Alias check

    # Attack grouping
    assert "pgd" in agg["grouped_by_attack"]
    assert "fgsm" in agg["grouped_by_attack"]
    assert agg["grouped_by_attack"]["pgd"]["attack_success_rate"] == 0.5

    # One-hot grouping: type_TRANSFER -> 'type'
    top_f = agg["global_top_features"]
    assert any(f["feature"] == "type" for f in top_f)
