========================================================================
          ADVERSCAN SECURITY ASSESSMENT REPORT           
========================================================================
  Report ID   : RPT-7A7DA4CB
  Scan ID     : SCAN-6A97FA
  Timestamp   : 2026-09-19 16:02:22
  Risk Level  : HIGH
  Vuln. Score : 50.09
========================================================================

1. EXECUTIVE SUMMARY
------------------------------------------------------------------------
  Scan ID            : SCAN-6A97FA
  Risk Level         : HIGH
  Vulnerability Score: 50.09
  Baseline Accuracy  : 97.18%
  Attacks Evaluated  : pgd
  XAI Enabled        : False
  Hardening Applied  : True
  Re-Test Conducted  : True
  Pipeline Status    : PARTIAL_SUCCESS

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
  - num_samples: 5000.00
  - num_classes: 43.00
  - accuracy: 97.18%
  - precision_macro: 93.93%
  - recall_macro: 93.10%
  - f1_macro: 92.84%
  - precision_weighted: 97.45%
  - recall_weighted: 97.18%
  - f1_weighted: 97.08%
  - average_confidence: 99.39%
  - average_entropy: 3.97%
  - batch_size: 32.00
  - device: cuda
  - timestamp: 2026-09-19 15:53:29
  - extra_metadata: {}

5. ADVERSARIAL ATTACK RESULTS
------------------------------------------------------------------------
| Attack Vector | Parameters | Exec Time | Baseline Acc | Adv Acc | Acc Drop | Attack Success Rate (ASR) |
|---|---|---|---|---|---|---|
| PGD           | Default    | 340.36s   | 97.18%       | 0.70%   | 96.48%   | **99.28%** |

  ▶ How Attacks Were Performed:
    • PGD: Configured with [Default parameters].

6. VULNERABILITY ASSESSMENT
------------------------------------------------------------------------
  ▶ Vector: PGD
    - Attack Success Rate (ASR) : 99.28%
    - Vulnerability Score       : 50.09
    - Risk Level                : HIGH
    - Clean vs Adversarial Acc  : 97.18% ➔ 0.70% (Drop: 96.48%)
    - Perturbation Magnitude    : linf_mean=88.67%, l2_mean=208.63, l0_mean=99.02%

7. VULNERABILITY SCORE & RISK LEVEL
------------------------------------------------------------------------
  Overall Vulnerability Score : 50.09
  Risk Level                  : HIGH

8. MITRE ATLAS MAPPING
------------------------------------------------------------------------
  ▶ PGD
    tactic      : AML.TA0000 — ML Attack Staging
    technique   : AML.T0043.001 — Craft Adversarial Data (PGD / Iterative)
    mitigation  : AML.M0003 — Adversarial Input Detection + AML.M0002 — Model Hardening

9. XAI FINDINGS
------------------------------------------------------------------------
  No XAI findings generated.

10. HARDENING
------------------------------------------------------------------------
  - metadata: {'defense_name': 'spatial_smoothing', 'defense_type': 'preprocessing', 'parameters': {'kernel_size': 3, 'sigma': 1.0}, 'execution_time_seconds': 0.025139570236206055, 'timestamp': '2026-09-19 16:01:58', 'extra_metadata': {'defense_family': 'preprocessing'}}
  - success: True
  - metrics_before: {}
  - metrics_after: {}
  - recommendations: ['Applied Gaussian Filter defense (kernel_size=3, sigma=1.00).', "Selector Context: Selected 'adversarial_detection' (score: 385.5) based on context: risk=HIGH, attack=pgd (iterative=True), domain=image, latency_sensitive=False, retraining_allowed=True."]
  - hardened_model_class: HardenedModelWrapper
  - has_hardened_inputs: True

11. RE-TEST RESULTS
------------------------------------------------------------------------
  - Hardened Model    : GTSRB_ViT_Demo
  - Dataset / Samples : bazyl/GTSRB (5000 samples)
  - Defense Mechanism : Input Preprocessing / Hardening
  - Re-Test Status    : ✅ PASSED — Defense Successfully Mitigated Attacks

12. BEFORE VS AFTER COMPARISON
------------------------------------------------------------------------
| Attack Vector | Before Defense (Adv Acc) | After Defense (Hardened Acc) | Robustness Gain | Before Risk | After Risk | Defense Status |
|---|---|---|---|---|---|---|
| PGD           | 0.70%                    | 0.00%                        | -0.70%          | HIGH        | MEDIUM     | **✅ WORKED** |

  ▶ Defense Verification & Retest Summary:
    • Verdict: ✅ DEFENSE SUCCESSFUL — The applied defense effectively neutralized all attack vectors.

13. EXECUTION PERFORMANCE
------------------------------------------------------------------------
  Run Label    : AdverScan [full]
  Started At   : 2026-09-19 15:53:07
  Total Time   : 0.00s
  Overall      : PARTIAL_SUCCESS

  MODULE                         STATUS          TIME
  ········································································
  M1 INGESTION                   ✅ SUCCESS      0.30s
  M2 BASELINE                    ✅ SUCCESS     21.15s
  M3 ATTACK ENGINE               ✅ SUCCESS    492.14s
  M5 VULNERABILITY ANALYSIS      ✅ SUCCESS      4.54s
  M6 EXPLAINABILITY              ❌ FAILED      10.96s
  M7 HARDENING                   ✅ SUCCESS      0.03s
  M8 RETEST                      ✅ SUCCESS     23.99s

14. RECOMMENDATIONS
------------------------------------------------------------------------
  [01] [HIGH] Model presents a HIGH security risk (Score: 50.09). Deploy adversarial training or certified defenses before production release.
  [02] Adversarial defense was applied. Validate post-hardening accuracy retention and conduct periodic re-tests to ensure defense durability.
  [03] Pipeline modules [M6 EXPLAINABILITY] encountered errors during this scan. Review error logs and re-run the pipeline after resolving the issues.

15. FINAL SECURITY SUMMARY
------------------------------------------------------------------------
  - risk_level: HIGH
  - vulnerability_score: 50.09
  - baseline_accuracy: 97.18%
  - mean_adversarial_accuracy: 0.70%
  - attacks_evaluated: ['pgd']
  - hardening_applied: True
  - retest_conducted: True
  - total_recommendations: 3
  - primary_recommendation: [HIGH] Model presents a HIGH security risk (Score: 50.09). Deploy adversarial training or certified defenses before production release.

========================================================================
  Generated by AdverScan — 2026-09-19 16:02:22
========================================================================