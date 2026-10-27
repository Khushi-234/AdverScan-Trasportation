"""
Cross-Domain Baseline Evaluation Script for AdverScan XAI (M5).

Runs one tiny model per target domain:
1. Financial Fraud Detection (Tabular Tree Ensemble)
2. Network Intrusion Detection (IDS Tabular Classifier)
3. Medical Imaging Diagnostics (PyTorch Convolutional Network)
4. Predictive Maintenance (PyTorch LSTM Temporal Sensor Streams)

Evaluates explanations, checks domain validity constraints, calculates metrics,
and aggregates vulnerability patterns.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from app.explainability.aggregate import aggregate_explanations
from app.explainability.domain_profiles import get_domain_profile
from app.explainability.explainer import XAIExplainer
from app.explainability.modality import Modality


def run_fraud_domain_evaluation(explainer: XAIExplainer):
    """Evaluate Fraud Detection domain (Tabular Random Forest)."""
    print("\n" + "=" * 60)
    print("1. EVALUATING DOMAIN: FINANCIAL FRAUD (TABULAR)")
    print("=" * 60)

    prof = get_domain_profile("fraud")
    rng = np.random.RandomState(42)

    # 7 features: amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type_TRANSFER, type_CASH_OUT
    X_train = rng.uniform(0, 10000, size=(100, 7))
    X_train[:, 5:] = rng.choice([0.0, 1.0], size=(100, 2))
    y_train = rng.choice([0, 1], size=100)

    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)

    clean_sample = np.array([[5000.0, 5000.0, 0.0, 0.0, 5000.0, 1.0, 0.0]])
    adv_sample = np.array([[450.0, 5000.0, 4550.0, 0.0, 450.0, 1.0, 0.0]])

    res = explainer.explain(
        model=model,
        clean_input=clean_sample,
        adversarial_input=adv_sample,
        true_label=1,
        technique="treeshap",
        domain="fraud",
        attack_name="hopskipjump_fraud",
        compute_metrics=True,
    )

    print(f"Domain:                  {res.domain}")
    print(f"Clean Prediction:        Class {res.clean_prediction} (Confidence: {res.clean_confidence:.4f})")
    print(f"Adversarial Prediction:  Class {res.adversarial_prediction} (Confidence: {res.adversarial_confidence:.4f})")
    print(f"Prediction Changed:      {res.prediction_changed}")
    print(f"Valid Adversarial:       {res.validity_flags.get('is_valid', True)}")
    print(f"Technique Used:          {res.technique}")
    top_f = res.failure_analysis.get("contributing_features", [])[:3]
    print(f"Top Exploited Features:  {[f['feature'] for f in top_f]}")
    return res


def run_ids_domain_evaluation(explainer: XAIExplainer):
    """Evaluate Network Intrusion Detection domain (IDS Tabular Decision Tree)."""
    print("\n" + "=" * 60)
    print("2. EVALUATING DOMAIN: NETWORK IDS (TABULAR)")
    print("=" * 60)

    prof = get_domain_profile("ids")
    rng = np.random.RandomState(42)

    # 6 features: duration, protocol_type, src_bytes, dst_bytes, count, srv_count
    X_train = rng.uniform(0, 1000, size=(100, 6))
    X_train[:, 1] = rng.choice([0.0, 1.0, 2.0], size=100)
    y_train = rng.choice([0, 1], size=100)

    model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

    clean_sample = np.array([[0.0, 1.0, 105.0, 146.0, 2.0, 2.0]])
    # Adversarial perturbation
    adv_sample = np.array([[0.0, 1.0, 850.0, 146.0, 50.0, 50.0]])

    res = explainer.explain(
        model=model,
        clean_input=clean_sample,
        adversarial_input=adv_sample,
        true_label=0,
        technique="shap",
        domain="ids",
        attack_name="pgd_tabular_ids",
        compute_metrics=True,
    )

    print(f"Domain:                  {res.domain}")
    print(f"Clean Prediction:        Class {res.clean_prediction} (Confidence: {res.clean_confidence:.4f})")
    print(f"Adversarial Prediction:  Class {res.adversarial_prediction} (Confidence: {res.adversarial_confidence:.4f})")
    print(f"Prediction Changed:      {res.prediction_changed}")
    print(f"Valid Adversarial:       {res.validity_flags.get('is_valid', True)}")
    return res


def run_medical_domain_evaluation(explainer: XAIExplainer):
    """Evaluate Healthcare / Medical Diagnostics domain (PyTorch CNN)."""
    print("\n" + "=" * 60)
    print("3. EVALUATING DOMAIN: HEALTHCARE / MEDICAL (IMAGE)")
    print("=" * 60)

    class MedicalChestXRayCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(8 * 4 * 4, 2)

        def forward(self, x):
            h = self.relu(self.conv1(x))
            h = self.pool(h)
            return self.fc(h.view(x.size(0), -1))

    model = MedicalChestXRayCNN().eval()

    clean_img = torch.rand(1, 1, 16, 16)
    # Add subtle localized perturbation simulating adversarial patch
    adv_img = clean_img.clone()
    adv_img[:, :, 6:10, 6:10] += 0.2
    adv_img = torch.clamp(adv_img, 0.0, 1.0)

    res = explainer.explain(
        model=model,
        clean_input=clean_img,
        adversarial_input=adv_img,
        true_label=0,
        technique="gradcam",
        modality=Modality.IMAGE,
        domain="medical",
        attack_name="cw_l2_medical",
        compute_metrics=True,
    )

    print(f"Domain:                  {res.domain}")
    print(f"Clean Prediction:        Class {res.clean_prediction} (Confidence: {res.clean_confidence:.4f})")
    print(f"Adversarial Prediction:  Class {res.adversarial_prediction} (Confidence: {res.adversarial_confidence:.4f})")
    print(f"Grad-CAM Heatmap Size:   {np.array(res.attribution['clean']['postprocessed']['heatmap']).shape}")
    print(f"Perturbation Alignment:  Cosine Sim = {res.comparison.get('attribution_cosine_similarity', 0.0):.4f}")
    return res


def run_maintenance_domain_evaluation(explainer: XAIExplainer):
    """Evaluate Predictive Maintenance domain (PyTorch LSTM Temporal Sensor Streams)."""
    print("\n" + "=" * 60)
    print("4. EVALUATING DOMAIN: PREDICTIVE MAINTENANCE (TIME-SERIES)")
    print("=" * 60)

    class TurbofanFailureLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=4, hidden_size=8, batch_first=True)
            self.fc = nn.Linear(8, 2)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = TurbofanFailureLSTM().eval()

    # Sequence length 10 timesteps, 4 sensors: T24, P30, Nf, BPR
    clean_ts = np.random.randn(1, 10, 4).astype(np.float32)
    adv_ts = clean_ts.copy()
    adv_ts[:, :, 1] += 0.5  # Attack targeted sensor P30

    res = explainer.explain(
        model=model,
        clean_input=clean_ts,
        adversarial_input=adv_ts,
        true_label=0,
        technique="integrated_gradients",
        modality=Modality.TIMESERIES,
        domain="maintenance",
        sensor_names=["T24", "P30", "Nf", "BPR"],
        attack_name="pgd_timeseries",
        compute_metrics=True,
    )

    print(f"Domain:                  {res.domain}")
    print(f"Clean Prediction:        Class {res.clean_prediction} (Confidence: {res.clean_confidence:.4f})")
    print(f"Adversarial Prediction:  Class {res.adversarial_prediction} (Confidence: {res.adversarial_confidence:.4f})")
    sensor_info = res.failure_analysis.get("sensor_importance", {})
    print(f"Top Exploited Sensor:    {sensor_info.get('top_exploited_sensor')}")
    return res


def main():
    print("=" * 65)
    print("AdverScan — Module 5 (XAI) Cross-Domain Baseline Evaluation")
    print("=" * 65)

    explainer = XAIExplainer()
    results = []

    results.append(run_fraud_domain_evaluation(explainer))
    results.append(run_ids_domain_evaluation(explainer))
    results.append(run_medical_domain_evaluation(explainer))
    results.append(run_maintenance_domain_evaluation(explainer))

    print("\n" + "=" * 65)
    print("DATASET-LEVEL AGGREGATION SUMMARY")
    print("=" * 65)
    agg = aggregate_explanations(results)
    print(f"Total Evaluated Samples: {agg['total_samples']}")
    print(f"Attacks Run:             {agg['attacks']}")
    print(f"Domains Covered:         {agg['modalities']}")
    print(f"Attack Success Rate:     {agg['attack_success_rate'] * 100:.2f}%")
    print(f"Mean Confidence Drop:    {agg['mean_confidence_drop']:.4f}")
    print(f"Per-Attack Breakdown:    {agg['grouped_by_attack']}")
    print("\n[OK] Cross-Domain Baseline Evaluation executed successfully across all domains!")


if __name__ == "__main__":
    main()
