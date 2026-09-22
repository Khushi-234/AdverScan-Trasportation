========================================================================
          ADVERSCAN SECURITY ASSESSMENT REPORT           
========================================================================
  Report ID   : RPT-B13E41E6
  Scan ID     : SCAN-0237E8
  Timestamp   : 2026-09-19 15:19:48
  Risk Level  : LOW
  Vuln. Score : 16.98
========================================================================

1. EXECUTIVE SUMMARY
------------------------------------------------------------------------
  Scan ID            : SCAN-0237E8
  Risk Level         : LOW
  Vulnerability Score: 16.98
  Baseline Accuracy  : 90.00%
  Attacks Evaluated  : fgsm
  XAI Enabled        : True
  Hardening Applied  : True
  Re-Test Conducted  : True
  Pipeline Status    : SUCCESS

2. MODEL INFORMATION
------------------------------------------------------------------------
  - framework: pytorch
  - model_name: GTSRB_ViT_Demo
  - input_shape: (1, 3, 224, 224)
  - output_shape: (1, 44)
  - num_classes: 44
  - task_type: classification
  - device: cuda
  - extra_info: {}

3. DATASET / EVALUATION CONFIGURATION
------------------------------------------------------------------------
  No dataset configuration recorded.

4. BASELINE PERFORMANCE
------------------------------------------------------------------------
  - dataset_name: bazyl/GTSRB
  - model_name: GTSRB_ViT_Demo
  - num_samples: 10.00
  - num_classes: 43.00
  - accuracy: 90.00%
  - precision_macro: 80.00%
  - recall_macro: 80.00%
  - f1_macro: 80.00%
  - precision_weighted: 90.00%
  - recall_weighted: 90.00%
  - f1_weighted: 90.00%
  - average_confidence: 99.90%
  - average_entropy: 1.56%
  - batch_size: 32.00
  - device: cuda
  - timestamp: 2026-09-19 15:19:48
  - extra_metadata: {}

5. ADVERSARIAL ATTACK RESULTS
------------------------------------------------------------------------
| Attack Vector | Parameters | Exec Time | Baseline Acc | Adv Acc | Acc Drop | Attack Success Rate (ASR) |
|---|---|---|---|---|---|---|
| FGSM          | Default    | 0.21s     | 90.00%       | 70.00%  | 20.00%   | **22.22%** |

  ▶ How Attacks Were Performed:
    • FGSM: Configured with [Default parameters].

6. VULNERABILITY ASSESSMENT
------------------------------------------------------------------------
  ▶ Vector: FGSM
    - Attack Success Rate (ASR) : 22.22%
    - Vulnerability Score       : 16.98
    - Risk Level                : LOW
    - Clean vs Adversarial Acc  : 90.00% ➔ 70.00% (Drop: 20.00%)
    - Perturbation Magnitude    : linf_mean=90.20%, l2_mean=174.20, l0_mean=97.73%

7. VULNERABILITY SCORE & RISK LEVEL
------------------------------------------------------------------------
  Overall Vulnerability Score : 16.98
  Risk Level                  : LOW

8. MITRE ATLAS MAPPING
------------------------------------------------------------------------
  ▶ FGSM
    tactic      : AML.TA0000 — ML Attack Staging
    technique   : AML.T0043 — Craft Adversarial Data (FGSM)
    mitigation  : AML.M0003 — Adversarial Input Detection

9. XAI FINDINGS
------------------------------------------------------------------------
  ▶ Technique: fgsm_shap
    - attack_name: fgsm
    - technique: shap
    - clean_confidence: 0.9990301132202148
    - adversarial_confidence: 0.7257112264633179
    - prediction_changed: True
    - attack_caused_failure: False
    - attribution: [Detailed Data Omitted]
    - comparison: [Detailed Data Omitted]
    - failure_analysis: [Detailed Data Omitted]

10. HARDENING
------------------------------------------------------------------------
  - metadata: {'defense_name': 'spatial_smoothing', 'defense_type': 'preprocessing', 'parameters': {'kernel_size': 3, 'sigma': 1.0}, 'execution_time_seconds': 0.008981466293334961, 'timestamp': '2026-09-19 15:19:48', 'extra_metadata': {'defense_family': 'preprocessing'}}
  - success: True
  - metrics_before: {}
  - metrics_after: {}
  - recommendations: ['Applied Gaussian Filter defense (kernel_size=3, sigma=1.00).', "Selector Context: Attack: FGSM | Risk: LOW | Vulnerability Score: 17.0 | Latency Sensitive: False. Generated 5 defense candidates: ['spatial_smoothing', 'gaussian_filter', 'median_filter', 'image_denoising', 'feature_squeezing']. Primary recommendation is 'spatial_smoothing' (suitability score: 398.3). Final defense selection should be based on post-defense empirical re-test results to confirm robustness improvement and clean performance preservation."]
  - hardened_model_class: HardenedModelWrapper
  - has_hardened_inputs: True

11. RE-TEST RESULTS
------------------------------------------------------------------------
  - Hardened Model    : GTSRB_ViT_Demo
  - Dataset / Samples : bazyl/GTSRB (10 samples)
  - Defense Mechanism : Input Preprocessing / Hardening
  - Re-Test Status    : ❌ FAILED — Vulnerabilities Persist

12. BEFORE VS AFTER COMPARISON
------------------------------------------------------------------------
| Attack Vector | Before Defense (Adv Acc) | After Defense (Hardened Acc) | Robustness Gain | Before Risk | After Risk | Defense Status |
|---|---|---|---|---|---|---|
| FGSM          | 70.00%                   | 70.00%                       | +0.00%          | LOW         | LOW        | **❌ FAILED** |

  ▶ Defense Verification & Retest Summary:
    • Verdict: ⚠️ PARTIAL DEFENSE — Some attack vectors showed persistent vulnerability.

13. EXECUTION PERFORMANCE
------------------------------------------------------------------------
  Run Label    : AdverScan [full]
  Started At   : 2026-09-19 15:19:47
  Total Time   : 0.00s
  Overall      : SUCCESS

  MODULE                         STATUS          TIME
  ········································································
  M1 INGESTION                   ✅ SUCCESS      0.32s
  M2 BASELINE                    ✅ SUCCESS      0.26s
  M3 ATTACK ENGINE               ✅ SUCCESS      0.34s
  M5 VULNERABILITY ANALYSIS      ✅ SUCCESS      0.01s
  M6 EXPLAINABILITY              ✅ SUCCESS      0.08s
  M7 HARDENING                   ✅ SUCCESS      0.01s
  M8 RETEST                      ✅ SUCCESS      0.30s

14. RECOMMENDATIONS
------------------------------------------------------------------------
  [01] Model exhibits strong baseline robustness against tested attack suites. Maintain continuous monitoring and conduct periodic adversarial re-testing.
  [02] XAI attribution maps are available. Review highlighted input regions disproportionately targeted by adversarial perturbations to guide robustness patches.
  [03] Adversarial defense was applied. Validate post-hardening accuracy retention and conduct periodic re-tests to ensure defense durability.
  [04] Re-test indicates persistent vulnerability for vector 'fgsm'. Increase adversarial training epochs or broaden epsilon schedules.

15. FINAL SECURITY SUMMARY
------------------------------------------------------------------------
  - risk_level: LOW
  - vulnerability_score: 16.98
  - baseline_accuracy: 90.00%
  - mean_adversarial_accuracy: 70.00%
  - attacks_evaluated: ['fgsm']
  - hardening_applied: True
  - retest_conducted: True
  - total_recommendations: 4
  - primary_recommendation: Model exhibits strong baseline robustness against tested attack suites. Maintain continuous monitoring and conduct periodic adversarial re-testing.

========================================================================
  Generated by AdverScan — 2026-09-19 15:19:48
========================================================================