========================================================================
          ADVERSCAN SECURITY ASSESSMENT REPORT           
========================================================================
  Report ID   : RPT-D99B4A4B
  Scan ID     : SCAN-8529C6
  Timestamp   : 2026-09-21 15:22:20
  Risk Level  : MEDIUM
  Vuln. Score : 45.77
========================================================================

1. EXECUTIVE SUMMARY
------------------------------------------------------------------------
  Scan ID            : SCAN-8529C6
  Risk Level         : MEDIUM
  Vulnerability Score: 45.77
  Baseline Accuracy  : 97.00%
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
  - num_samples: 100.00
  - num_classes: 43.00
  - accuracy: 97.00%
  - precision_macro: 97.40%
  - recall_macro: 97.21%
  - f1_macro: 96.75%
  - precision_weighted: 97.83%
  - recall_weighted: 97.00%
  - f1_weighted: 96.97%
  - average_confidence: 98.98%
  - average_entropy: 6.52%
  - batch_size: 32.00
  - device: cuda
  - timestamp: 2026-09-21 15:22:06
  - extra_metadata: {}

5. ADVERSARIAL ATTACK RESULTS
------------------------------------------------------------------------
| Attack Vector | Parameters | Exec Time | Baseline Acc | Adv Acc | Acc Drop | Attack Success Rate (ASR) |
|---|---|---|---|---|---|---|
| FGSM          | Default    | 4.69s     | 97.00%       | 31.00%  | 66.00%   | **68.04%** |

  ▶ How Attacks Were Performed:
    • FGSM: Configured with [Default parameters].

6. VULNERABILITY ASSESSMENT
------------------------------------------------------------------------
  ▶ Vector: FGSM
    - Attack Success Rate (ASR) : 68.04%
    - Vulnerability Score       : 45.77
    - Risk Level                : MEDIUM
    - Clean vs Adversarial Acc  : 97.00% ➔ 31.00% (Drop: 66.00%)
    - Perturbation Magnitude    : linf_mean=91.55%, l2_mean=219.03, l0_mean=98.41%

7. VULNERABILITY SCORE & RISK LEVEL
------------------------------------------------------------------------
  Overall Vulnerability Score : 45.77
  Risk Level                  : MEDIUM

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
    - clean_confidence: 0.9898384809494019
    - adversarial_confidence: 0.4198632836341858
    - prediction_changed: True
    - attack_caused_failure: False
    - attribution: [Detailed Data Omitted]
    - comparison: [Detailed Data Omitted]
    - failure_analysis: [Detailed Data Omitted]

10. HARDENING
------------------------------------------------------------------------
  - metadata: {'defense_name': 'median_filter', 'defense_type': 'preprocessing', 'parameters': {'kernel_size': 3}, 'execution_time_seconds': 0.09232878684997559, 'timestamp': '2026-09-21 15:22:18', 'extra_metadata': {'defense_family': 'preprocessing'}}
  - success: True
  - status: accepted
  - selected_defense: median_filter
  - num_attempts: 4
  - is_improved: True
  - improvement_sufficient: True
  - thresholds: {'min_vuln_score_improvement': 5.0, 'min_asr_reduction': 0.1, 'max_clean_accuracy_drop': 0.02, 'max_latency_overhead_ms': 30.0, 'respect_latency_sensitive': True, 'severe_clean_drop_threshold': 0.1, 'extra_thresholds': {}}
  - defense_attempts: [{'defense_name': 'spatial_smoothing', 'status': 'insufficient', 'is_improved': True, 'improvement_sufficient': False, 'vuln_score_before': 62.5, 'vuln_score_after': 59.38, 'vuln_score_improvement': 3.12, 'asr_before': 0.625, 'asr_after': 0.5938, 'asr_reduction': 0.0312, 'clean_accuracy_before': 0.97, 'clean_accuracy_after': 0.96, 'clean_accuracy_drop': 0.01, 'clean_accuracy_drop_pct_points': 1.0, 'latency_ms': 15.0, 'latency_overhead_ms': 0.0, 'parameters': {'kernel_size': 3, 'sigma': 1.0}, 'reason': 'Insufficient robustness improvement: Vuln score improvement 3.1 < 5.0 points and ASR reduction 3.1% < 10.0%'}, {'defense_name': 'gaussian_filter', 'status': 'insufficient', 'is_improved': True, 'improvement_sufficient': False, 'vuln_score_before': 62.5, 'vuln_score_after': 59.38, 'vuln_score_improvement': 3.12, 'asr_before': 0.625, 'asr_after': 0.5938, 'asr_reduction': 0.0312, 'clean_accuracy_before': 0.97, 'clean_accuracy_after': 0.96, 'clean_accuracy_drop': 0.01, 'clean_accuracy_drop_pct_points': 1.0, 'latency_ms': 15.0, 'latency_overhead_ms': 0.0, 'parameters': {'kernel_size': 3, 'sigma': 1.0}, 'reason': 'Insufficient robustness improvement: Vuln score improvement 3.1 < 5.0 points and ASR reduction 3.1% < 10.0%'}, {'defense_name': 'feature_squeezing', 'status': 'harmful', 'is_improved': True, 'improvement_sufficient': False, 'vuln_score_before': 62.5, 'vuln_score_after': 37.5, 'vuln_score_improvement': 25.0, 'asr_before': 0.625, 'asr_after': 0.375, 'asr_reduction': 0.25, 'clean_accuracy_before': 0.97, 'clean_accuracy_after': 0.41, 'clean_accuracy_drop': 0.56, 'clean_accuracy_drop_pct_points': 56.0, 'latency_ms': 15.0, 'latency_overhead_ms': 0.0, 'parameters': {'bit_depth': 4}, 'reason': 'Severe clean accuracy drop: 56.0 percentage points exceeds harmful threshold 10.0%'}, {'defense_name': 'median_filter', 'status': 'accepted', 'is_improved': True, 'improvement_sufficient': True, 'vuln_score_before': 62.5, 'vuln_score_after': 53.12, 'vuln_score_improvement': 9.38, 'asr_before': 0.625, 'asr_after': 0.5312, 'asr_reduction': 0.0938, 'clean_accuracy_before': 0.97, 'clean_accuracy_after': 0.97, 'clean_accuracy_drop': 0.0, 'clean_accuracy_drop_pct_points': 0.0, 'latency_ms': 15.0, 'latency_overhead_ms': 0.0, 'parameters': {'kernel_size': 3}, 'reason': 'Defense satisfies all acceptance thresholds with meaningful improvement.'}]
  - metrics_before: {'clean_accuracy': 0.97, 'asr': 0.625, 'vulnerability_score': 62.5, 'latency_ms': 15.0}
  - metrics_after: {'clean_accuracy': 0.97, 'asr': 0.5312, 'vulnerability_score': 53.12, 'latency_ms': 15.0}
  - recommendations: ['Applied Median Filter defense (kernel_size=3).', "Iterative Defense Selection: Accepted 'median_filter' after 4 attempt(s) (satisfied acceptance thresholds: Defense satisfies all acceptance thresholds with meaningful improvement.)."]
  - hardened_model_class: HardenedModelWrapper
  - has_hardened_inputs: True

11. RE-TEST RESULTS
------------------------------------------------------------------------
  - Hardened Model    : GTSRB_ViT_Demo
  - Dataset / Samples : bazyl/GTSRB (100 samples)
  - Defense Mechanism : Input Preprocessing / Hardening
  - Re-Test Status    : ✅ PASSED — Defense Successfully Mitigated Attacks

12. BEFORE VS AFTER COMPARISON
------------------------------------------------------------------------
| Attack Vector | Before Defense (Adv Acc) | After Defense (Hardened Acc) | Robustness Gain | Before Risk | After Risk | Defense Status |
|---|---|---|---|---|---|---|
| FGSM          | 31.00%                   | 37.50%                       | +6.50%          | MEDIUM      | MEDIUM     | **✅ WORKED** |

  ▶ Defense Verification & Retest Summary:
    • Verdict: ✅ DEFENSE SUCCESSFUL — The applied defense effectively neutralized all attack vectors.

13. EXECUTION PERFORMANCE
------------------------------------------------------------------------
  Run Label    : AdverScan [full]
  Started At   : 2026-09-21 15:22:05
  Total Time   : 0.00s
  Overall      : SUCCESS

  MODULE                         STATUS          TIME
  ········································································
  M1 INGESTION                   ✅ SUCCESS      0.32s
  M2 BASELINE                    ✅ SUCCESS      0.54s
  M3 ATTACK ENGINE               ✅ SUCCESS      6.42s
  M5 VULNERABILITY ANALYSIS      ✅ SUCCESS      0.10s
  M6 EXPLAINABILITY              ✅ SUCCESS      0.83s
  M7 HARDENING                   ✅ SUCCESS      6.23s
  M8 RETEST                      ✅ SUCCESS      1.19s

14. RECOMMENDATIONS
------------------------------------------------------------------------
  [01] [MEDIUM] Vulnerability score 45.77 warrants attention. Implement input sanitization and monitor inference traffic for anomalies.
  [02] XAI attribution maps are available. Review highlighted input regions disproportionately targeted by adversarial perturbations to guide robustness patches.
  [03] Adversarial defense was applied. Validate post-hardening accuracy retention and conduct periodic re-tests to ensure defense durability.

15. FINAL SECURITY SUMMARY
------------------------------------------------------------------------
  - risk_level: MEDIUM
  - vulnerability_score: 45.77
  - baseline_accuracy: 97.00%
  - mean_adversarial_accuracy: 31.00%
  - attacks_evaluated: ['fgsm']
  - hardening_applied: True
  - retest_conducted: True
  - total_recommendations: 3
  - primary_recommendation: [MEDIUM] Vulnerability score 45.77 warrants attention. Implement input sanitization and monitor inference traffic for anomalies.

========================================================================
  Generated by AdverScan — 2026-09-21 15:22:20
========================================================================