========================================================================
          ADVERSCAN SECURITY ASSESSMENT REPORT           
========================================================================
  Report ID   : RPT-6C7F2DF9
  Scan ID     : SCAN-8DF59F
  Timestamp   : 2026-09-21 15:21:54
  Risk Level  : HIGH
  Vuln. Score : 51.74
========================================================================

1. EXECUTIVE SUMMARY
------------------------------------------------------------------------
  Scan ID            : SCAN-8DF59F
  Risk Level         : HIGH
  Vulnerability Score: 51.74
  Baseline Accuracy  : 97.00%
  Attacks Evaluated  : pgd
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
  - timestamp: 2026-09-21 15:21:23
  - extra_metadata: {}

5. ADVERSARIAL ATTACK RESULTS
------------------------------------------------------------------------
| Attack Vector | Parameters | Exec Time | Baseline Acc | Adv Acc | Acc Drop | Attack Success Rate (ASR) |
|---|---|---|---|---|---|---|
| PGD           | Default    | 11.56s    | 97.00%       | 0.00%   | 97.00%   | **100.00%** |

  ▶ How Attacks Were Performed:
    • PGD: Configured with [Default parameters].

6. VULNERABILITY ASSESSMENT
------------------------------------------------------------------------
  ▶ Vector: PGD
    - Attack Success Rate (ASR) : 100.00%
    - Vulnerability Score       : 51.74
    - Risk Level                : HIGH
    - Clean vs Adversarial Acc  : 97.00% ➔ 0.00% (Drop: 97.00%)
    - Perturbation Magnitude    : linf_mean=91.55%, l2_mean=218.51, l0_mean=99.30%

7. VULNERABILITY SCORE & RISK LEVEL
------------------------------------------------------------------------
  Overall Vulnerability Score : 51.74
  Risk Level                  : HIGH

8. MITRE ATLAS MAPPING
------------------------------------------------------------------------
  ▶ PGD
    tactic      : AML.TA0000 — ML Attack Staging
    technique   : AML.T0043.001 — Craft Adversarial Data (PGD / Iterative)
    mitigation  : AML.M0003 — Adversarial Input Detection + AML.M0002 — Model Hardening

9. XAI FINDINGS
------------------------------------------------------------------------
  ▶ Technique: pgd_shap
    - attack_name: pgd
    - technique: shap
    - clean_confidence: 0.9898384809494019
    - adversarial_confidence: 0.7099232077598572
    - prediction_changed: True
    - attack_caused_failure: False
    - attribution: [Detailed Data Omitted]
    - comparison: [Detailed Data Omitted]
    - failure_analysis: [Detailed Data Omitted]

10. HARDENING
------------------------------------------------------------------------
  - metadata: {'defense_name': 'none', 'defense_type': 'none', 'parameters': {}, 'execution_time_seconds': 0.0, 'timestamp': None, 'extra_metadata': {'tested_candidates': ['feature_alignment', 'confidence_rejection', 'feature_denoising', 'randomized_smoothing', 'feature_squeezing'], 'total_attempts': 5, 'selection_outcome': 'no_defense_accepted'}}
  - success: True
  - status: insufficient
  - selected_defense: none
  - num_attempts: 5
  - is_improved: True
  - improvement_sufficient: False
  - thresholds: {'min_vuln_score_improvement': 5.0, 'min_asr_reduction': 0.1, 'max_clean_accuracy_drop': 0.02, 'max_latency_overhead_ms': 30.0, 'respect_latency_sensitive': True, 'severe_clean_drop_threshold': 0.1, 'extra_thresholds': {}}
  - defense_attempts: [{'defense_name': 'feature_alignment', 'status': 'harmful', 'is_improved': False, 'improvement_sufficient': False, 'vuln_score_before': None, 'vuln_score_after': None, 'vuln_score_improvement': None, 'asr_before': None, 'asr_after': None, 'asr_reduction': None, 'clean_accuracy_before': None, 'clean_accuracy_after': None, 'clean_accuracy_drop': None, 'clean_accuracy_drop_pct_points': None, 'latency_ms': None, 'latency_overhead_ms': None, 'parameters': {'alignment_weight': 0.5, 'noise_std': 0.02}, 'reason': 'Defense execution failed: FeatureAlignmentDefense failed: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor'}, {'defense_name': 'confidence_rejection', 'status': 'harmful', 'is_improved': False, 'improvement_sufficient': False, 'vuln_score_before': None, 'vuln_score_after': None, 'vuln_score_improvement': None, 'asr_before': None, 'asr_after': None, 'asr_reduction': None, 'clean_accuracy_before': None, 'clean_accuracy_after': None, 'clean_accuracy_drop': None, 'clean_accuracy_drop_pct_points': None, 'latency_ms': None, 'latency_overhead_ms': None, 'parameters': {'threshold': 0.6}, 'reason': 'Defense execution failed: ConfidenceRejectionDefense failed: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor'}, {'defense_name': 'feature_denoising', 'status': 'insufficient', 'is_improved': False, 'improvement_sufficient': False, 'vuln_score_before': 100.0, 'vuln_score_after': 100.0, 'vuln_score_improvement': 0.0, 'asr_before': 1.0, 'asr_after': 1.0, 'asr_reduction': 0.0, 'clean_accuracy_before': 0.97, 'clean_accuracy_after': 0.95, 'clean_accuracy_drop': 0.02, 'clean_accuracy_drop_pct_points': 2.0, 'latency_ms': 15.0, 'latency_overhead_ms': 0.0, 'parameters': {'method': 'mean', 'strength': 0.2}, 'reason': 'Insufficient robustness improvement: Vuln score improvement 0.0 < 5.0 points and ASR reduction 0.0% < 10.0%'}, {'defense_name': 'randomized_smoothing', 'status': 'insufficient', 'is_improved': False, 'improvement_sufficient': False, 'vuln_score_before': 100.0, 'vuln_score_after': None, 'vuln_score_improvement': None, 'asr_before': 1.0, 'asr_after': None, 'asr_reduction': None, 'clean_accuracy_before': 0.97, 'clean_accuracy_after': None, 'clean_accuracy_drop': None, 'clean_accuracy_drop_pct_points': None, 'latency_ms': None, 'latency_overhead_ms': None, 'parameters': {'sigma': 0.1, 'num_samples': 20}, 'reason': 'No vulnerability score or ASR metrics available to verify improvement'}, {'defense_name': 'feature_squeezing', 'status': 'harmful', 'is_improved': True, 'improvement_sufficient': False, 'vuln_score_before': 100.0, 'vuln_score_after': 40.62, 'vuln_score_improvement': 59.38, 'asr_before': 1.0, 'asr_after': 0.4062, 'asr_reduction': 0.5938, 'clean_accuracy_before': 0.97, 'clean_accuracy_after': 0.41, 'clean_accuracy_drop': 0.56, 'clean_accuracy_drop_pct_points': 56.0, 'latency_ms': 15.0, 'latency_overhead_ms': 0.0, 'parameters': {'bit_depth': 4}, 'reason': 'Severe clean accuracy drop: 56.0 percentage points exceeds harmful threshold 10.0%'}]
  - metrics_before: {'clean_accuracy': 0.97, 'asr': 1.0, 'vulnerability_score': 100.0, 'latency_ms': 15.0}
  - metrics_after: {'clean_accuracy': 0.41, 'asr': 0.4062, 'vulnerability_score': 40.62, 'latency_ms': 15.0}
  - recommendations: ["Iterative Defense Selection: No tested defense (5 attempted: ['feature_alignment', 'confidence_rejection', 'feature_denoising', 'randomized_smoothing', 'feature_squeezing']) provided sufficient improvement meeting the configured acceptance criteria. The baseline model was retained without modification."]
  - hardened_model_class: ViTForImageClassification
  - has_hardened_inputs: False

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
| PGD           | 0.00%                    | 0.00%                        | +0.00%          | HIGH        | MEDIUM     | **✅ WORKED** |

  ▶ Defense Verification & Retest Summary:
    • Verdict: ✅ DEFENSE SUCCESSFUL — The applied defense effectively neutralized all attack vectors.

13. EXECUTION PERFORMANCE
------------------------------------------------------------------------
  Run Label    : AdverScan [full]
  Started At   : 2026-09-21 15:21:22
  Total Time   : 0.00s
  Overall      : SUCCESS

  MODULE                         STATUS          TIME
  ········································································
  M1 INGESTION                   ✅ SUCCESS      0.33s
  M2 BASELINE                    ✅ SUCCESS      0.56s
  M3 ATTACK ENGINE               ✅ SUCCESS     14.73s
  M5 VULNERABILITY ANALYSIS      ✅ SUCCESS      0.10s
  M6 EXPLAINABILITY              ✅ SUCCESS      0.89s
  M7 HARDENING                   ✅ SUCCESS     11.81s
  M8 RETEST                      ✅ SUCCESS      3.91s

14. RECOMMENDATIONS
------------------------------------------------------------------------
  [01] [HIGH] Model presents a HIGH security risk (Score: 51.74). Deploy adversarial training or certified defenses before production release.
  [02] XAI attribution maps are available. Review highlighted input regions disproportionately targeted by adversarial perturbations to guide robustness patches.
  [03] Adversarial defense was applied. Validate post-hardening accuracy retention and conduct periodic re-tests to ensure defense durability.

15. FINAL SECURITY SUMMARY
------------------------------------------------------------------------
  - risk_level: HIGH
  - vulnerability_score: 51.74
  - baseline_accuracy: 97.00%
  - mean_adversarial_accuracy: None
  - attacks_evaluated: ['pgd']
  - hardening_applied: True
  - retest_conducted: True
  - total_recommendations: 3
  - primary_recommendation: [HIGH] Model presents a HIGH security risk (Score: 51.74). Deploy adversarial training or certified defenses before production release.

========================================================================
  Generated by AdverScan — 2026-09-21 15:21:54
========================================================================