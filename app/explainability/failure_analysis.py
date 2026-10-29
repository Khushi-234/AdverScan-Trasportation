"""
Prediction-change and failure analysis for explainability module in AdverScan.
Identifies feature contributions to adversarial misclassification.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from app.explainability.utils import to_numpy


def _normalize_label_scalar(val: Any) -> Any:
    """Normalize array, tensor, scalar, or one-hot vector into a comparable scalar or set of scalars."""
    if val is None:
        return None
    if hasattr(val, "detach"):
        val = val.detach().cpu().numpy()
    if isinstance(val, np.ndarray):
        if val.ndim == 1 and len(val) > 1 and np.issubdtype(val.dtype, np.number):
            if np.all((val >= 0.0) & (val <= 1.0)) and np.isclose(np.sum(val), 1.0, atol=1e-3):
                return int(np.argmax(val))
        if val.size == 1:
            return val.item()
        return [int(x) if np.issubdtype(type(x), np.integer) else x for x in val.tolist()]
    if hasattr(val, "item") and callable(val.item):
        try:
            return val.item()
        except Exception:
            pass
    return val


def is_prediction_correct(prediction: Any, true_label: Optional[Any]) -> Optional[bool]:
    """
    Check if a prediction matches the ground truth label.

    Handles numpy scalars, 0-d arrays, 1-element arrays, and one-hot encoded labels safely.

    Args:
        prediction: Predicted class or label (scalar, tensor, or array).
        true_label: Ground truth label or container of acceptable labels.

    Returns:
        True if correct, False if incorrect, None if true_label is unavailable.
    """
    if true_label is None:
        return None

    norm_pred = _normalize_label_scalar(prediction)
    norm_true = _normalize_label_scalar(true_label)

    if isinstance(norm_true, (list, tuple, set)):
        return norm_pred in norm_true
    if isinstance(norm_pred, (list, tuple, set)):
        return norm_true in norm_pred

    try:
        return bool(norm_pred == norm_true)
    except Exception:
        try:
            return np.array_equal(np.asarray(norm_pred), np.asarray(norm_true))
        except Exception:
            return False


def _to_clean_value(val: Any) -> Any:
    """Extract clean python primitive (int, float, list, str) from tensor or numpy scalar."""
    if val is None:
        return None
    if hasattr(val, "detach"):
        val = val.detach().cpu().numpy()
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return val.item()
        return val.tolist()
    if hasattr(val, "item") and callable(val.item):
        try:
            return val.item()
        except Exception:
            pass
    return val


def _is_equal(a: Any, b: Any) -> bool:
    """Safely compare two predictions or labels for equality."""
    norm_a = _normalize_label_scalar(a)
    norm_b = _normalize_label_scalar(b)
    if isinstance(norm_a, set) or isinstance(norm_b, set):
        try:
            return set(norm_a) == set(norm_b)
        except Exception:
            return bool(norm_a == norm_b)
    if isinstance(norm_a, (list, tuple)) or isinstance(norm_b, (list, tuple)):
        try:
            return norm_a == norm_b
        except Exception:
            pass
    try:
        return bool(norm_a == norm_b)
    except Exception:
        try:
            return np.array_equal(np.asarray(norm_a), np.asarray(norm_b))
        except Exception:
            return False


def analyze_failure(
    clean_prediction: Any,
    adversarial_prediction: Any,
    true_label: Optional[Any] = None,
    clean_attribution: Optional[Any] = None,
    adv_attribution: Optional[Any] = None,
    clean_top_features: Optional[List[Dict[str, Any]]] = None,
    adv_top_features: Optional[List[Dict[str, Any]]] = None,
    target_class: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    sensor_names: Optional[List[str]] = None,
    modality: str = "tabular",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Perform deterministic analysis on prediction changes, failure modes, and feature contributions.

    Computes importance shifts from the full attribution vectors so un-selected clean features
    are not falsely attributed 0.0 importance. Provides (timestep, sensor) and per-sensor
    rankings for time-series, and grid-cell regions for images.

    Args:
        clean_prediction: Model prediction on clean input.
        adversarial_prediction: Model prediction on adversarial input.
        true_label: Optional ground truth label.
        clean_attribution: Optional clean feature attributions (full array).
        adv_attribution: Optional adversarial feature attributions (full array).
        clean_top_features: Top contributing features in clean input.
        adv_top_features: Top contributing features in adversarial input.
        target_class: Optional target class index for targeted attacks.
        feature_names: Optional feature name list for tabular data.
        sensor_names: Optional sensor names for time-series data.
        modality: Data modality ('tabular', 'timeseries', 'image', 'text').
        top_k: Number of contributing features to report.

    Returns:
        Dictionary containing failure mode, contributing features, shifts, and targeted info.
    """
    prediction_changed = bool(clean_prediction != adversarial_prediction)
    clean_correct = is_prediction_correct(clean_prediction, true_label)
    adversarial_correct = is_prediction_correct(adversarial_prediction, true_label)

    if true_label is None:
        attack_caused_failure = None
        if prediction_changed:
            failure_mode = "prediction_changed_unknown_correctness"
        else:
            failure_mode = "prediction_unchanged_unknown_correctness"
    else:
        attack_caused_failure = clean_correct is True and adversarial_correct is False

        if clean_correct is True and adversarial_correct is False:
            failure_mode = "clean_correct_to_adversarial_incorrect"
        elif clean_correct is False and adversarial_correct is False:
            failure_mode = "clean_incorrect_to_adversarial_incorrect"
        elif clean_correct is True and adversarial_correct is True:
            if prediction_changed:
                failure_mode = "prediction_changed_remains_correct"
            else:
                failure_mode = "clean_correct_to_adversarial_correct"
        else:  # clean_correct is False, adversarial_correct is True
            failure_mode = "clean_incorrect_to_adversarial_correct"

    # Analyze contributing features from FULL attribution vectors when available
    contributing_features: List[Dict[str, Any]] = []
    sensor_importance: Dict[str, Any] = {}

    clean_arr = to_numpy(clean_attribution, dtype=np.float64) if clean_attribution is not None else None
    adv_arr = to_numpy(adv_attribution, dtype=np.float64) if adv_attribution is not None else None

    if clean_arr is not None and adv_arr is not None and clean_arr.shape == adv_arr.shape:
        # 1. Time-series: evaluate (timestep, sensor) cells and per-sensor importance
        if modality.lower() in ("timeseries", "time_series") and adv_arr.ndim in (2, 3):
            ts_adv = adv_arr[0] if adv_arr.ndim == 3 else adv_arr
            ts_clean = clean_arr[0] if clean_arr.ndim == 3 else clean_arr
            t_steps, n_sensors = ts_adv.shape
            s_names = sensor_names or [f"sensor_{i}" for i in range(n_sensors)]

            # Per-sensor aggregate importance
            clean_sensor_imp = np.mean(np.abs(ts_clean), axis=0)
            adv_sensor_imp = np.mean(np.abs(ts_adv), axis=0)
            sensor_shifts = adv_sensor_imp - clean_sensor_imp

            ranked_sensors = np.argsort(adv_sensor_imp)[::-1]
            sensor_rankings = []
            for rank, s_idx in enumerate(ranked_sensors):
                s_name = s_names[s_idx] if s_idx < len(s_names) else f"sensor_{s_idx}"
                sensor_rankings.append({
                    "sensor": s_name,
                    "adversarial_importance": float(adv_sensor_imp[s_idx]),
                    "clean_importance": float(clean_sensor_imp[s_idx]),
                    "importance_shift": float(sensor_shifts[s_idx]),
                    "rank": rank + 1,
                })
            sensor_importance = {
                "sensor_rankings": sensor_rankings,
                "top_exploited_sensor": sensor_rankings[0]["sensor"] if sensor_rankings else None,
            }

            # Top (timestep, sensor) cells
            flat_adv = ts_adv.ravel()
            flat_clean = ts_clean.ravel()
            ranked_cells = np.argsort(np.abs(flat_adv))[::-1][:top_k]
            for rank, cell_idx in enumerate(ranked_cells):
                t = cell_idx // n_sensors
                s = cell_idx % n_sensors
                s_name = s_names[s] if s < len(s_names) else f"sensor_{s}"
                cell_name = f"t{t}_{s_name}"
                c_imp = float(flat_clean[cell_idx])
                a_imp = float(flat_adv[cell_idx])
                contributing_features.append({
                    "feature": cell_name,
                    "timestep": int(t),
                    "sensor": s_name,
                    "adversarial_importance": a_imp,
                    "clean_importance": c_imp,
                    "importance_shift": float(a_imp - c_imp),
                    "rank": rank + 1,
                })

        # 2. Image: group top features into grid patches if spatial dimensions
        elif modality.lower() == "image" and adv_arr.ndim in (3, 4):
            img_adv = np.abs(adv_arr[0] if adv_arr.ndim == 4 else adv_arr)
            img_clean = np.abs(clean_arr[0] if clean_arr.ndim == 4 else clean_arr)
            if img_adv.ndim == 3 and img_adv.shape[0] in (1, 3, 4):
                img_adv = np.mean(img_adv, axis=0)
                img_clean = np.mean(img_clean, axis=0)
            elif img_adv.ndim == 3 and img_adv.shape[-1] in (1, 3, 4):
                img_adv = np.mean(img_adv, axis=-1)
                img_clean = np.mean(img_clean, axis=-1)

            h, w = img_adv.shape
            grid_size = 4
            gh, gw = max(1, h // grid_size), max(1, w // grid_size)
            patches = []
            for r in range(grid_size):
                for c in range(grid_size):
                    r_slice = slice(r * gh, min((r + 1) * gh, h))
                    c_slice = slice(c * gw, min((c + 1) * gw, w))
                    a_imp = float(np.mean(img_adv[r_slice, c_slice]))
                    c_imp = float(np.mean(img_clean[r_slice, c_slice]))
                    patches.append({
                        "feature": f"patch_r{r}_c{c}",
                        "adversarial_importance": a_imp,
                        "clean_importance": c_imp,
                        "importance_shift": float(a_imp - c_imp),
                    })
            patches.sort(key=lambda x: abs(float(x["adversarial_importance"])), reverse=True)
            for rank, p in enumerate(patches[:top_k]):
                p["rank"] = rank + 1
                contributing_features.append(p)

        # 3. Tabular & 1D features
        else:
            flat_adv = adv_arr.ravel()
            flat_clean = clean_arr.ravel()
            f_names = feature_names or [f"f{i}" for i in range(len(flat_adv))]

            ranked_idx = np.argsort(np.abs(flat_adv))[::-1][:top_k]
            for rank, idx in enumerate(ranked_idx):
                name = f_names[idx] if idx < len(f_names) else f"f{idx}"
                c_imp = float(flat_clean[idx])
                a_imp = float(flat_adv[idx])
                contributing_features.append({
                    "feature": name,
                    "adversarial_importance": a_imp,
                    "clean_importance": c_imp,
                    "importance_shift": float(a_imp - c_imp),
                    "rank": rank + 1,
                })
    elif adv_top_features:
        # Fallback to provided top features
        clean_map = {f.get("feature"): f.get("importance", 0.0) for f in (clean_top_features or [])}
        for f in adv_top_features:
            name = f.get("feature")
            a_imp = f.get("importance", 0.0)
            c_imp = clean_map.get(name, 0.0)
            contributing_features.append({
                "feature": name,
                "adversarial_importance": a_imp,
                "clean_importance": c_imp,
                "importance_shift": float(a_imp - c_imp),
                "rank": f.get("rank"),
            })

    # Targeted attack analysis
    targeted_info = None
    if target_class is not None:
        targeted_success = bool(adversarial_prediction == target_class)
        targeted_info = {
            "target_class": target_class,
            "targeted_success": targeted_success,
            "explanation": (
                f"Model was successfully forced to predict target class {target_class}"
                if targeted_success
                else f"Model predicted {adversarial_prediction} instead of target {target_class}"
            ),
        }

    out: Dict[str, Any] = {
        "prediction_changed": prediction_changed,
        "clean_correct": clean_correct,
        "adversarial_correct": adversarial_correct,
        "attack_caused_failure": attack_caused_failure,
        "failure_mode": failure_mode,
        "contributing_features": contributing_features,
    }
    if sensor_importance:
        out["sensor_importance"] = sensor_importance
    if targeted_info is not None:
        out["targeted_attack"] = targeted_info

    return out
