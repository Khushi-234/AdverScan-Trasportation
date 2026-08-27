========================================================================
          ADVERSCAN SECURITY ASSESSMENT REPORT           
========================================================================
  Report ID   : RPT-D4E2CA38
  Scan ID     : SCAN-7BD011
  Timestamp   : 2026-08-27 17:07:38
  Risk Level  : MEDIUM
  Vuln. Score : 33.63
========================================================================

1. EXECUTIVE SUMMARY
------------------------------------------------------------------------
  Scan ID            : SCAN-7BD011
  Risk Level         : MEDIUM
  Vulnerability Score: 33.63
  Baseline Accuracy  : 97.00%
  Attacks Evaluated  : fgsm, pgd, deepfool
  XAI Enabled        : True
  Hardening Applied  : True
  Re-Test Conducted  : True
  Pipeline Status    : UNKNOWN

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
  - num_samples: 12630
  - num_classes: 43
  - accuracy: 97.00%
  - precision_macro: 93.97%
  - recall_macro: 93.38%
  - f1_macro: 93.11%
  - precision_weighted: 97.32%
  - recall_weighted: 97.00%
  - f1_weighted: 96.95%
  - average_confidence: 99.37%
  - average_entropy: 4.01%
  - per_class_metrics: {'0': {'precision': 0.9523809523809523, 'recall': 1.0, 'f1': 0.975609756097561, 'support': 60}, '1': {'precision': 1.0, 'recall': 0.9944444444444445, 'f1': 0.9972144846796658, 'support': 720}, '2': {'precision': 0.9664948453608248, 'recall': 1.0, 'f1': 0.9829619921363041, 'support': 750}, '3': {'precision': 0.9977064220183486, 'recall': 0.9666666666666667, 'f1': 0.981941309255079, 'support': 450}, '4': {'precision': 0.9969512195121951, 'recall': 0.990909090909091, 'f1': 0.993920972644377, 'support': 660}, '5': {'precision': 0.9750390015600624, 'recall': 0.9920634920634921, 'f1': 0.983477576711251, 'support': 630}, '6': {'precision': 1.0, 'recall': 0.8933333333333333, 'f1': 0.9436619718309859, 'support': 150}, '7': {'precision': 0.9977777777777778, 'recall': 0.9977777777777778, 'f1': 0.9977777777777778, 'support': 450}, '8': {'precision': 0.9884259259259259, 'recall': 0.9488888888888889, 'f1': 0.9682539682539683, 'support': 450}, '9': {'precision': 0.997920997920998, 'recall': 1.0, 'f1': 0.9989594172736732, 'support': 480}, '10': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 660}, '11': {'precision': 0.9902912621359223, 'recall': 0.9714285714285714, 'f1': 0.9807692307692307, 'support': 420}, '12': {'precision': 0.9985528219971056, 'recall': 1.0, 'f1': 0.999275887038378, 'support': 690}, '13': {'precision': 0.9986130374479889, 'recall': 1.0, 'f1': 0.9993060374739764, 'support': 720}, '14': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 270}, '15': {'precision': 1.0, 'recall': 0.9952380952380953, 'f1': 0.9976133651551312, 'support': 210}, '16': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 150}, '17': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 360}, '18': {'precision': 0.9972222222222222, 'recall': 0.9205128205128205, 'f1': 0.9573333333333334, 'support': 390}, '19': {'precision': 0.8611111111111112, 'recall': 0.5166666666666667, 'f1': 0.6458333333333334, 'support': 60}, '20': {'precision': 0.7203389830508474, 'recall': 0.9444444444444444, 'f1': 0.8173076923076923, 'support': 90}, '21': {'precision': 0.75, 'recall': 1.0, 'f1': 0.8571428571428571, 'support': 90}, '22': {'precision': 1.0, 'recall': 0.8666666666666667, 'f1': 0.9285714285714286, 'support': 120}, '23': {'precision': 0.9736842105263158, 'recall': 0.9866666666666667, 'f1': 0.9801324503311258, 'support': 150}, '24': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 90}, '25': {'precision': 0.9692622950819673, 'recall': 0.9854166666666667, 'f1': 0.9772727272727273, 'support': 480}, '26': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 180}, '27': {'precision': 0.9491525423728814, 'recall': 0.9333333333333333, 'f1': 0.9411764705882353, 'support': 60}, '28': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 150}, '29': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 90}, '30': {'precision': 0.9259259259259259, 'recall': 1.0, 'f1': 0.9615384615384616, 'support': 150}, '31': {'precision': 0.9888475836431226, 'recall': 0.9851851851851852, 'f1': 0.987012987012987, 'support': 270}, '32': {'precision': 0.8955223880597015, 'recall': 1.0, 'f1': 0.9448818897637795, 'support': 60}, '33': {'precision': 0.958904109589041, 'recall': 0.6666666666666666, 'f1': 0.7865168539325843, 'support': 210}, '34': {'precision': 0.6052631578947368, 'recall': 0.9583333333333334, 'f1': 0.7419354838709677, 'support': 120}, '35': {'precision': 0.9974093264248705, 'recall': 0.9871794871794872, 'f1': 0.9922680412371134, 'support': 390}, '36': {'precision': 0.7815126050420168, 'recall': 0.775, 'f1': 0.7782426778242678, 'support': 120}, '37': {'precision': 0.55, 'recall': 0.55, 'f1': 0.55, 'support': 60}, '38': {'precision': 0.9346049046321526, 'recall': 0.9942028985507246, 'f1': 0.9634831460674157, 'support': 690}, '39': {'precision': 0.9333333333333333, 'recall': 0.4666666666666667, 'f1': 0.6222222222222222, 'support': 90}, '40': {'precision': 0.9886363636363636, 'recall': 0.9666666666666667, 'f1': 0.9775280898876404, 'support': 90}, '41': {'precision': 0.8571428571428571, 'recall': 1.0, 'f1': 0.9230769230769231, 'support': 60}, '42': {'precision': 0.9101123595505618, 'recall': 0.9, 'f1': 0.9050279329608939, 'support': 90}}
  - confusion_matrix: [[60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 716, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 750, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 12, 435, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 654, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 1, 625, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 134, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 8], [0, 0, 0, 0, 0, 0, 0, 449, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 9, 1, 1, 12, 0, 0, 427, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 480, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 660, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 408, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 690, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 720, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 270, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 360, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 359, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 104, 4, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 148, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 473, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 266, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 140, 70, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 115, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 385, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 27, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 26, 33, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 686, 3, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 42, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 87, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 81]]
  - batch_size: 32
  - device: cuda
  - timestamp: 2026-08-27 17:05:51
  - extra_metadata: {}

5. ADVERSARIAL ATTACK RESULTS
------------------------------------------------------------------------
  ▶ FGSM
    execution_time_seconds   : 41.52%

  ▶ PGD
    execution_time_seconds   : 2.1442525386810303

  ▶ DEEPFOOL
    execution_time_seconds   : 22.429787397384644

6. VULNERABILITY ASSESSMENT
------------------------------------------------------------------------
  ▶ Vector: fgsm
    [Assessment] attack_name: fgsm
    [Assessment] dataset_name: bazyl/GTSRB
    [Assessment] num_samples: 12630
    [Assessment] attack_success_rate: None
    [Assessment] perturbation: {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}
    [Assessment] accuracy_drop: 0.5949920823436263
    [Assessment] f1_drop: 0.6343487829563317
    [Assessment] confidence_drop: 0.5016941428184509
    [Assessment] model_degradation: 0.577011669372803
    [Assessment] clean_accuracy: 0.9699920823436263
    [Assessment] adversarial_accuracy: 0.375
    [Assessment] clean_f1: 0.9311453197528684
    [Assessment] adversarial_f1: 0.2967965367965368
    [Assessment] clean_confidence: 0.993720531463623
    [Assessment] adversarial_confidence: 0.4920263886451721
    [Assessment] timestamp: 2026-08-27 17:06:17
    [Scoring] attack_name: fgsm
    [Scoring] vulnerability_score: 40.39
    [Scoring] risk_level: MEDIUM
    [Scoring] sub_scores: {'asr_score': None, 'degradation_score': 57.7, 'stealth_score': 0.0}
    [Scoring] timestamp: 2026-08-27 17:06:17
    [Scoring] metadata: {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}

  ▶ Vector: pgd
    [Assessment] attack_name: pgd
    [Assessment] dataset_name: bazyl/GTSRB
    [Assessment] num_samples: 12630
    [Assessment] attack_success_rate: None
    [Assessment] perturbation: {'linf_mean': 0.9169117696583271, 'l2_mean': 200.46237961644826, 'l0_mean': 0.9932491795546343, 'is_estimated': False}
    [Assessment] accuracy_drop: 0.9699920823436263
    [Assessment] f1_drop: 0.9311453197528684
    [Assessment] confidence_drop: 0.19553357362747192
    [Assessment] model_degradation: 0.6988903252413222
    [Assessment] clean_accuracy: 0.9699920823436263
    [Assessment] adversarial_accuracy: 0.0
    [Assessment] clean_f1: 0.9311453197528684
    [Assessment] adversarial_f1: 0.0
    [Assessment] clean_confidence: 0.993720531463623
    [Assessment] adversarial_confidence: 0.7981869578361511
    [Assessment] timestamp: 2026-08-27 17:06:17
    [Scoring] attack_name: pgd
    [Scoring] vulnerability_score: 48.92
    [Scoring] risk_level: MEDIUM
    [Scoring] sub_scores: {'asr_score': None, 'degradation_score': 69.89, 'stealth_score': 0.0}
    [Scoring] timestamp: 2026-08-27 17:06:17
    [Scoring] metadata: {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}

  ▶ Vector: deepfool
    [Assessment] attack_name: deepfool
    [Assessment] dataset_name: bazyl/GTSRB
    [Assessment] num_samples: 12630
    [Assessment] attack_success_rate: None
    [Assessment] perturbation: {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}
    [Assessment] accuracy_drop: 0.4074920823436263
    [Assessment] f1_drop: 0.4836254784830271
    [Assessment] confidence_drop: 0.5500268936157227
    [Assessment] model_degradation: 0.4803814848141254
    [Assessment] clean_accuracy: 0.9699920823436263
    [Assessment] adversarial_accuracy: 0.5625
    [Assessment] clean_f1: 0.9311453197528684
    [Assessment] adversarial_f1: 0.4475198412698413
    [Assessment] clean_confidence: 0.993720531463623
    [Assessment] adversarial_confidence: 0.4436936378479004
    [Assessment] timestamp: 2026-08-27 17:06:17
    [Scoring] attack_name: deepfool
    [Scoring] vulnerability_score: 33.63
    [Scoring] risk_level: MEDIUM
    [Scoring] sub_scores: {'asr_score': None, 'degradation_score': 48.04, 'stealth_score': 0.0}
    [Scoring] timestamp: 2026-08-27 17:06:18
    [Scoring] metadata: {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}

7. VULNERABILITY SCORE & RISK LEVEL
------------------------------------------------------------------------
  Overall Vulnerability Score : 33.63
  Risk Level                  : MEDIUM

8. MITRE ATLAS MAPPING
------------------------------------------------------------------------
  ▶ FGSM
    tactic      : AML.TA0000 — ML Attack Staging
    technique   : AML.T0043 — Craft Adversarial Data (FGSM)
    mitigation  : AML.M0003 — Adversarial Input Detection

  ▶ PGD
    tactic      : AML.TA0000 — ML Attack Staging
    technique   : AML.T0043.001 — Craft Adversarial Data (PGD / Iterative)
    mitigation  : AML.M0003 — Adversarial Input Detection + AML.M0002 — Model Hardening

  ▶ DEEPFOOL
    tactic      : AML.TA0000 — ML Attack Staging
    technique   : AML.T0043.002 — Craft Adversarial Data (Minimal Perturbation)
    mitigation  : AML.M0003 — Adversarial Input Detection

9. XAI FINDINGS
------------------------------------------------------------------------
  ▶ Technique: fgsm_shap
    - attack_name: fgsm
    - technique: shap
    - clean_prediction: [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17]
    - adversarial_prediction: [16, 1, 38, 34, 13, 38, 18, 13, 25, 35, 38, 16, 13, 15, 38, 13, 21, 13, 11, 13, 13, 34, 15, 13, 2, 11, 13, 13, 35, 11, 5, 8]
    - clean_confidence: 0.998881459236145
    - adversarial_confidence: 0.4918379485607147
    - prediction_changed: True
    - true_label: tensor([16,  1, 38, 33, 11, 38, 18, 12, 25, 35, 12,  7, 23,  7,  4,  9, 21, 20,
        27, 38,  4, 33,  9,  3,  1, 11, 13, 10,  9, 11,  5, 17])
    - attack_caused_failure: False
    - attribution: {'technique': 'shap', 'clean': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}, 'adversarial': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}}
    - comparison: {'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [16, 1, 38, 34, 13, 38, 18, 13, 25, 35, 38, 16, 13, 15, 38, 13, 21, 13, 11, 13, 13, 34, 15, 13, 2, 11, 13, 13, 35, 11, 5, 8], 'clean_confidence': 0.998881459236145, 'adversarial_confidence': 0.4918379485607147, 'prediction_changed': True, 'confidence_difference': 0.5070435106754303, 'attribution_comparison_status': 'unavailable', 'attribution_l1': None, 'attribution_l2': None, 'attribution_cosine_similarity': None, 'attribution_mean_difference': None}
    - failure_analysis: {'clean_correct': False, 'adversarial_correct': False, 'prediction_changed': True, 'attack_caused_failure': False, 'true_label': [16, 1, 38, 33, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [16, 1, 38, 34, 13, 38, 18, 13, 25, 35, 38, 16, 13, 15, 38, 13, 21, 13, 11, 13, 13, 34, 15, 13, 2, 11, 13, 13, 35, 11, 5, 8], 'failure_mode': 'clean_incorrect_to_adversarial_incorrect'}
    - metadata: {'attack_name': 'fgsm', 'technique': 'shap', 'assessment_result': {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5949920823436263, 'f1_drop': 0.6343487829563317, 'confidence_drop': 0.5016941428184509, 'model_degradation': 0.577011669372803, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.4920263886451721, 'timestamp': '2026-08-27 17:06:17', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.41515207290649414}}}

  ▶ Technique: pgd_shap
    - attack_name: pgd
    - technique: shap
    - clean_prediction: [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17]
    - adversarial_prediction: [10, 5, 12, 9, 13, 12, 26, 32, 18, 16, 25, 16, 13, 15, 38, 20, 19, 41, 25, 11, 13, 2, 32, 38, 5, 25, 12, 13, 12, 42, 6, 38]
    - clean_confidence: 0.998881459236145
    - adversarial_confidence: 0.7980610132217407
    - prediction_changed: True
    - true_label: tensor([16,  1, 38, 33, 11, 38, 18, 12, 25, 35, 12,  7, 23,  7,  4,  9, 21, 20,
        27, 38,  4, 33,  9,  3,  1, 11, 13, 10,  9, 11,  5, 17])
    - attack_caused_failure: False
    - attribution: {'technique': 'shap', 'clean': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}, 'adversarial': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}}
    - comparison: {'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [10, 5, 12, 9, 13, 12, 26, 32, 18, 16, 25, 16, 13, 15, 38, 20, 19, 41, 25, 11, 13, 2, 32, 38, 5, 25, 12, 13, 12, 42, 6, 38], 'clean_confidence': 0.998881459236145, 'adversarial_confidence': 0.7980610132217407, 'prediction_changed': True, 'confidence_difference': 0.2008204460144043, 'attribution_comparison_status': 'unavailable', 'attribution_l1': None, 'attribution_l2': None, 'attribution_cosine_similarity': None, 'attribution_mean_difference': None}
    - failure_analysis: {'clean_correct': False, 'adversarial_correct': False, 'prediction_changed': True, 'attack_caused_failure': False, 'true_label': [16, 1, 38, 33, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [10, 5, 12, 9, 13, 12, 26, 32, 18, 16, 25, 16, 13, 15, 38, 20, 19, 41, 25, 11, 13, 2, 32, 38, 5, 25, 12, 13, 12, 42, 6, 38], 'failure_mode': 'clean_incorrect_to_adversarial_incorrect'}
    - metadata: {'attack_name': 'pgd', 'technique': 'shap', 'assessment_result': {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.46237961644826, 'l0_mean': 0.9932491795546343, 'is_estimated': False}, 'accuracy_drop': 0.9699920823436263, 'f1_drop': 0.9311453197528684, 'confidence_drop': 0.19553357362747192, 'model_degradation': 0.6988903252413222, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.0, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.7981869578361511, 'timestamp': '2026-08-27 17:06:17', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1442525386810303}}}

  ▶ Technique: deepfool_shap
    - attack_name: deepfool
    - technique: shap
    - clean_prediction: [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17]
    - adversarial_prediction: [16, 1, 38, 34, 13, 38, 18, 12, 25, 35, 12, 13, 13, 15, 38, 13, 23, 31, 25, 38, 13, 33, 15, 31, 1, 11, 13, 13, 9, 11, 5, 17]
    - clean_confidence: 0.998881459236145
    - adversarial_confidence: 0.44342663884162903
    - prediction_changed: True
    - true_label: tensor([16,  1, 38, 33, 11, 38, 18, 12, 25, 35, 12,  7, 23,  7,  4,  9, 21, 20,
        27, 38,  4, 33,  9,  3,  1, 11, 13, 10,  9, 11,  5, 17])
    - attack_caused_failure: False
    - attribution: {'technique': 'shap', 'clean': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}, 'adversarial': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}}
    - comparison: {'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [16, 1, 38, 34, 13, 38, 18, 12, 25, 35, 12, 13, 13, 15, 38, 13, 23, 31, 25, 38, 13, 33, 15, 31, 1, 11, 13, 13, 9, 11, 5, 17], 'clean_confidence': 0.998881459236145, 'adversarial_confidence': 0.44342663884162903, 'prediction_changed': True, 'confidence_difference': 0.555454820394516, 'attribution_comparison_status': 'unavailable', 'attribution_l1': None, 'attribution_l2': None, 'attribution_cosine_similarity': None, 'attribution_mean_difference': None}
    - failure_analysis: {'clean_correct': False, 'adversarial_correct': False, 'prediction_changed': True, 'attack_caused_failure': False, 'true_label': [16, 1, 38, 33, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [16, 1, 38, 34, 13, 38, 18, 12, 25, 35, 12, 13, 13, 15, 38, 13, 23, 31, 25, 38, 13, 33, 15, 31, 1, 11, 13, 13, 9, 11, 5, 17], 'failure_mode': 'clean_incorrect_to_adversarial_incorrect'}
    - metadata: {'attack_name': 'deepfool', 'technique': 'shap', 'assessment_result': {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4074920823436263, 'f1_drop': 0.4836254784830271, 'confidence_drop': 0.5500268936157227, 'model_degradation': 0.4803814848141254, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 17:06:17', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.429787397384644}}}

10. HARDENING
------------------------------------------------------------------------
  - metadata: {'defense_name': 'spatial_smoothing', 'defense_type': 'preprocessing', 'parameters': {'kernel_size': 3, 'sigma': 1.0}, 'execution_time_seconds': 0.027127742767333984, 'timestamp': '2026-08-27 17:06:18', 'extra_metadata': {}}
  - success: True
  - metrics_before: {}
  - metrics_after: {}
  - recommendations: ['Evaluated model with Spatial Smoothing (Gaussian Kernel) defense.', "Selector Context: Single-step or moderate risk attack ('fgsm'). Recommending Spatial Smoothing input preprocessing."]
  - hardened_model_class: PreprocessedModelWrapper
  - has_hardened_inputs: True

11. RE-TEST RESULTS
------------------------------------------------------------------------
  - hardened_model_name: GTSRB_ViT_Demo
  - dataset_name: bazyl/GTSRB
  - num_samples: 12630
  - before_baseline_evaluation: {'dataset_name': 'bazyl/GTSRB', 'model_name': 'GTSRB_ViT_Demo', 'num_samples': 12630, 'num_classes': 43, 'accuracy': 0.9699920823436263, 'precision_macro': 0.9397241986808867, 'recall_macro': 0.933822291394334, 'f1_macro': 0.9311453197528684, 'precision_weighted': 0.9732145452689693, 'recall_weighted': 0.9699920823436263, 'f1_weighted': 0.969502903024833, 'average_confidence': 0.993720531463623, 'average_entropy': 0.040141209959983826, 'per_class_metrics': {'0': {'precision': 0.9523809523809523, 'recall': 1.0, 'f1': 0.975609756097561, 'support': 60}, '1': {'precision': 1.0, 'recall': 0.9944444444444445, 'f1': 0.9972144846796658, 'support': 720}, '2': {'precision': 0.9664948453608248, 'recall': 1.0, 'f1': 0.9829619921363041, 'support': 750}, '3': {'precision': 0.9977064220183486, 'recall': 0.9666666666666667, 'f1': 0.981941309255079, 'support': 450}, '4': {'precision': 0.9969512195121951, 'recall': 0.990909090909091, 'f1': 0.993920972644377, 'support': 660}, '5': {'precision': 0.9750390015600624, 'recall': 0.9920634920634921, 'f1': 0.983477576711251, 'support': 630}, '6': {'precision': 1.0, 'recall': 0.8933333333333333, 'f1': 0.9436619718309859, 'support': 150}, '7': {'precision': 0.9977777777777778, 'recall': 0.9977777777777778, 'f1': 0.9977777777777778, 'support': 450}, '8': {'precision': 0.9884259259259259, 'recall': 0.9488888888888889, 'f1': 0.9682539682539683, 'support': 450}, '9': {'precision': 0.997920997920998, 'recall': 1.0, 'f1': 0.9989594172736732, 'support': 480}, '10': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 660}, '11': {'precision': 0.9902912621359223, 'recall': 0.9714285714285714, 'f1': 0.9807692307692307, 'support': 420}, '12': {'precision': 0.9985528219971056, 'recall': 1.0, 'f1': 0.999275887038378, 'support': 690}, '13': {'precision': 0.9986130374479889, 'recall': 1.0, 'f1': 0.9993060374739764, 'support': 720}, '14': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 270}, '15': {'precision': 1.0, 'recall': 0.9952380952380953, 'f1': 0.9976133651551312, 'support': 210}, '16': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 150}, '17': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 360}, '18': {'precision': 0.9972222222222222, 'recall': 0.9205128205128205, 'f1': 0.9573333333333334, 'support': 390}, '19': {'precision': 0.8611111111111112, 'recall': 0.5166666666666667, 'f1': 0.6458333333333334, 'support': 60}, '20': {'precision': 0.7203389830508474, 'recall': 0.9444444444444444, 'f1': 0.8173076923076923, 'support': 90}, '21': {'precision': 0.75, 'recall': 1.0, 'f1': 0.8571428571428571, 'support': 90}, '22': {'precision': 1.0, 'recall': 0.8666666666666667, 'f1': 0.9285714285714286, 'support': 120}, '23': {'precision': 0.9736842105263158, 'recall': 0.9866666666666667, 'f1': 0.9801324503311258, 'support': 150}, '24': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 90}, '25': {'precision': 0.9692622950819673, 'recall': 0.9854166666666667, 'f1': 0.9772727272727273, 'support': 480}, '26': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 180}, '27': {'precision': 0.9491525423728814, 'recall': 0.9333333333333333, 'f1': 0.9411764705882353, 'support': 60}, '28': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 150}, '29': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 90}, '30': {'precision': 0.9259259259259259, 'recall': 1.0, 'f1': 0.9615384615384616, 'support': 150}, '31': {'precision': 0.9888475836431226, 'recall': 0.9851851851851852, 'f1': 0.987012987012987, 'support': 270}, '32': {'precision': 0.8955223880597015, 'recall': 1.0, 'f1': 0.9448818897637795, 'support': 60}, '33': {'precision': 0.958904109589041, 'recall': 0.6666666666666666, 'f1': 0.7865168539325843, 'support': 210}, '34': {'precision': 0.6052631578947368, 'recall': 0.9583333333333334, 'f1': 0.7419354838709677, 'support': 120}, '35': {'precision': 0.9974093264248705, 'recall': 0.9871794871794872, 'f1': 0.9922680412371134, 'support': 390}, '36': {'precision': 0.7815126050420168, 'recall': 0.775, 'f1': 0.7782426778242678, 'support': 120}, '37': {'precision': 0.55, 'recall': 0.55, 'f1': 0.55, 'support': 60}, '38': {'precision': 0.9346049046321526, 'recall': 0.9942028985507246, 'f1': 0.9634831460674157, 'support': 690}, '39': {'precision': 0.9333333333333333, 'recall': 0.4666666666666667, 'f1': 0.6222222222222222, 'support': 90}, '40': {'precision': 0.9886363636363636, 'recall': 0.9666666666666667, 'f1': 0.9775280898876404, 'support': 90}, '41': {'precision': 0.8571428571428571, 'recall': 1.0, 'f1': 0.9230769230769231, 'support': 60}, '42': {'precision': 0.9101123595505618, 'recall': 0.9, 'f1': 0.9050279329608939, 'support': 90}}, 'confusion_matrix': [[60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 716, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 750, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 12, 435, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 654, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 1, 625, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 134, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 8], [0, 0, 0, 0, 0, 0, 0, 449, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 9, 1, 1, 12, 0, 0, 427, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 480, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 660, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 408, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 690, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 720, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 270, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 360, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 359, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 104, 4, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 148, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 473, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 266, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 140, 70, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 115, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 385, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 27, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 26, 33, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 686, 3, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 42, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 87, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 81]], 'batch_size': 32, 'device': 'cuda', 'timestamp': '2026-08-27 17:05:51', 'extra_metadata': {}}
  - after_baseline_evaluation: {'dataset_name': 'bazyl/GTSRB', 'model_name': 'GTSRB_ViT_Demo', 'num_samples': 12630, 'num_classes': 43, 'accuracy': 0.9699920823436263, 'precision_macro': 0.9397241986808867, 'recall_macro': 0.933822291394334, 'f1_macro': 0.9311453197528684, 'precision_weighted': 0.9732145452689693, 'recall_weighted': 0.9699920823436263, 'f1_weighted': 0.969502903024833, 'average_confidence': 0.993720531463623, 'average_entropy': 0.040141209959983826, 'per_class_metrics': {'0': {'precision': 0.9523809523809523, 'recall': 1.0, 'f1': 0.975609756097561, 'support': 60}, '1': {'precision': 1.0, 'recall': 0.9944444444444445, 'f1': 0.9972144846796658, 'support': 720}, '2': {'precision': 0.9664948453608248, 'recall': 1.0, 'f1': 0.9829619921363041, 'support': 750}, '3': {'precision': 0.9977064220183486, 'recall': 0.9666666666666667, 'f1': 0.981941309255079, 'support': 450}, '4': {'precision': 0.9969512195121951, 'recall': 0.990909090909091, 'f1': 0.993920972644377, 'support': 660}, '5': {'precision': 0.9750390015600624, 'recall': 0.9920634920634921, 'f1': 0.983477576711251, 'support': 630}, '6': {'precision': 1.0, 'recall': 0.8933333333333333, 'f1': 0.9436619718309859, 'support': 150}, '7': {'precision': 0.9977777777777778, 'recall': 0.9977777777777778, 'f1': 0.9977777777777778, 'support': 450}, '8': {'precision': 0.9884259259259259, 'recall': 0.9488888888888889, 'f1': 0.9682539682539683, 'support': 450}, '9': {'precision': 0.997920997920998, 'recall': 1.0, 'f1': 0.9989594172736732, 'support': 480}, '10': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 660}, '11': {'precision': 0.9902912621359223, 'recall': 0.9714285714285714, 'f1': 0.9807692307692307, 'support': 420}, '12': {'precision': 0.9985528219971056, 'recall': 1.0, 'f1': 0.999275887038378, 'support': 690}, '13': {'precision': 0.9986130374479889, 'recall': 1.0, 'f1': 0.9993060374739764, 'support': 720}, '14': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 270}, '15': {'precision': 1.0, 'recall': 0.9952380952380953, 'f1': 0.9976133651551312, 'support': 210}, '16': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 150}, '17': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 360}, '18': {'precision': 0.9972222222222222, 'recall': 0.9205128205128205, 'f1': 0.9573333333333334, 'support': 390}, '19': {'precision': 0.8611111111111112, 'recall': 0.5166666666666667, 'f1': 0.6458333333333334, 'support': 60}, '20': {'precision': 0.7203389830508474, 'recall': 0.9444444444444444, 'f1': 0.8173076923076923, 'support': 90}, '21': {'precision': 0.75, 'recall': 1.0, 'f1': 0.8571428571428571, 'support': 90}, '22': {'precision': 1.0, 'recall': 0.8666666666666667, 'f1': 0.9285714285714286, 'support': 120}, '23': {'precision': 0.9736842105263158, 'recall': 0.9866666666666667, 'f1': 0.9801324503311258, 'support': 150}, '24': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 90}, '25': {'precision': 0.9692622950819673, 'recall': 0.9854166666666667, 'f1': 0.9772727272727273, 'support': 480}, '26': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 180}, '27': {'precision': 0.9491525423728814, 'recall': 0.9333333333333333, 'f1': 0.9411764705882353, 'support': 60}, '28': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 150}, '29': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 90}, '30': {'precision': 0.9259259259259259, 'recall': 1.0, 'f1': 0.9615384615384616, 'support': 150}, '31': {'precision': 0.9888475836431226, 'recall': 0.9851851851851852, 'f1': 0.987012987012987, 'support': 270}, '32': {'precision': 0.8955223880597015, 'recall': 1.0, 'f1': 0.9448818897637795, 'support': 60}, '33': {'precision': 0.958904109589041, 'recall': 0.6666666666666666, 'f1': 0.7865168539325843, 'support': 210}, '34': {'precision': 0.6052631578947368, 'recall': 0.9583333333333334, 'f1': 0.7419354838709677, 'support': 120}, '35': {'precision': 0.9974093264248705, 'recall': 0.9871794871794872, 'f1': 0.9922680412371134, 'support': 390}, '36': {'precision': 0.7815126050420168, 'recall': 0.775, 'f1': 0.7782426778242678, 'support': 120}, '37': {'precision': 0.55, 'recall': 0.55, 'f1': 0.55, 'support': 60}, '38': {'precision': 0.9346049046321526, 'recall': 0.9942028985507246, 'f1': 0.9634831460674157, 'support': 690}, '39': {'precision': 0.9333333333333333, 'recall': 0.4666666666666667, 'f1': 0.6222222222222222, 'support': 90}, '40': {'precision': 0.9886363636363636, 'recall': 0.9666666666666667, 'f1': 0.9775280898876404, 'support': 90}, '41': {'precision': 0.8571428571428571, 'recall': 1.0, 'f1': 0.9230769230769231, 'support': 60}, '42': {'precision': 0.9101123595505618, 'recall': 0.9, 'f1': 0.9050279329608939, 'support': 90}}, 'confusion_matrix': [[60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 716, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 750, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 12, 435, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 654, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 1, 625, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 134, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 8], [0, 0, 0, 0, 0, 0, 0, 449, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 9, 1, 1, 12, 0, 0, 427, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 480, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 660, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 408, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 690, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 720, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 270, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 360, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 359, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 104, 4, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 148, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 473, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 266, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 140, 70, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 115, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 385, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 27, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 26, 33, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 686, 3, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 42, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 87, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 81]], 'batch_size': 32, 'device': 'cuda', 'timestamp': '2026-08-27 17:07:12', 'extra_metadata': {}}
  - before_attack_results: {'fgsm': {'attack_name': 'fgsm', 'attack_class': 'FGSM', 'execution_time_seconds': 0.41515207290649414, 'parameters': {}}, 'pgd': {'attack_name': 'pgd', 'attack_class': 'PGD', 'execution_time_seconds': 2.1442525386810303, 'parameters': {}}, 'deepfool': {'attack_name': 'deepfool', 'attack_class': 'DeepFool', 'execution_time_seconds': 22.429787397384644, 'parameters': {}}}
  - after_attack_results: {'fgsm': {'attack_name': 'fgsm', 'attack_class': 'FGSM', 'execution_time_seconds': 0.02113628387451172, 'parameters': {}}, 'pgd': {'attack_name': 'pgd', 'attack_class': 'PGD', 'execution_time_seconds': 2.145221710205078, 'parameters': {}}, 'deepfool': {'attack_name': 'deepfool', 'attack_class': 'DeepFool', 'execution_time_seconds': 22.489156246185303, 'parameters': {}}}
  - before_vulnerability_analysis: {'fgsm': {'assessment': {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5949920823436263, 'f1_drop': 0.6343487829563317, 'confidence_drop': 0.5016941428184509, 'model_degradation': 0.577011669372803, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.4920263886451721, 'timestamp': '2026-08-27 17:06:17', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.41515207290649414}}, 'scoring': {'attack_name': 'fgsm', 'vulnerability_score': 40.39, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.7, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:06:17', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}}}, 'pgd': {'assessment': {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.46237961644826, 'l0_mean': 0.9932491795546343, 'is_estimated': False}, 'accuracy_drop': 0.9699920823436263, 'f1_drop': 0.9311453197528684, 'confidence_drop': 0.19553357362747192, 'model_degradation': 0.6988903252413222, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.0, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.7981869578361511, 'timestamp': '2026-08-27 17:06:17', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1442525386810303}}, 'scoring': {'attack_name': 'pgd', 'vulnerability_score': 48.92, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 69.89, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:06:17', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}}}, 'deepfool': {'assessment': {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4074920823436263, 'f1_drop': 0.4836254784830271, 'confidence_drop': 0.5500268936157227, 'model_degradation': 0.4803814848141254, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 17:06:17', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.429787397384644}}, 'scoring': {'attack_name': 'deepfool', 'vulnerability_score': 33.63, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.04, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:06:18', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}}}}
  - after_vulnerability_analysis: {'fgsm': {'assessment': {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5949920823436263, 'f1_drop': 0.6343487829563317, 'confidence_drop': 0.5016941428184509, 'model_degradation': 0.577011669372803, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.4920263886451721, 'timestamp': '2026-08-27 17:07:38', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.02113628387451172}}, 'scoring': {'attack_name': 'fgsm', 'vulnerability_score': 40.39, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.7, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:07:38', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}}}, 'pgd': {'assessment': {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.46649667631584, 'l0_mean': 0.9932041297964498, 'is_estimated': False}, 'accuracy_drop': 0.9699920823436263, 'f1_drop': 0.9311453197528684, 'confidence_drop': 0.16048479080200195, 'model_degradation': 0.6872073976328322, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.0, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.8332357406616211, 'timestamp': '2026-08-27 17:07:38', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.145221710205078}}, 'scoring': {'attack_name': 'pgd', 'vulnerability_score': 48.1, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 68.72, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:07:38', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}}}, 'deepfool': {'assessment': {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4074920823436263, 'f1_drop': 0.4836254784830271, 'confidence_drop': 0.5500268936157227, 'model_degradation': 0.4803814848141254, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 17:07:38', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.489156246185303}}, 'scoring': {'attack_name': 'deepfool', 'vulnerability_score': 33.63, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.04, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:07:38', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}}}}
  - overall_improved: False
  - timestamp: 2026-08-27 17:06:18
  - execution_time_seconds: 79.777
  - extra_metadata: {}

12. BEFORE VS AFTER COMPARISON
------------------------------------------------------------------------
  ▶ Vector: fgsm
    - attack_name: fgsm
    - before_assessment: {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5949920823436263, 'f1_drop': 0.6343487829563317, 'confidence_drop': 0.5016941428184509, 'model_degradation': 0.577011669372803, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.4920263886451721, 'timestamp': '2026-08-27 17:06:17', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.41515207290649414}}
    - after_assessment: {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5949920823436263, 'f1_drop': 0.6343487829563317, 'confidence_drop': 0.5016941428184509, 'model_degradation': 0.577011669372803, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.4920263886451721, 'timestamp': '2026-08-27 17:07:38', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.02113628387451172}}
    - before_scoring: {'attack_name': 'fgsm', 'vulnerability_score': 40.39, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.7, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:06:17', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}}
    - after_scoring: {'attack_name': 'fgsm', 'vulnerability_score': 40.39, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.7, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:07:38', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}}
    - delta_attack_success_rate: None
    - delta_accuracy_drop: 0.0
    - delta_f1_drop: 0.0
    - delta_confidence_drop: 0.0
    - delta_model_degradation: 0.0
    - delta_vulnerability_score: 0.0
    - delta_clean_accuracy: 0.0
    - delta_adversarial_accuracy: 0.0
    - delta_perturbation: {'l0_mean': 0.0, 'l2_mean': 0.0, 'linf_mean': 0.0, 'is_estimated': 0.0}
    - before_risk_level: MEDIUM
    - after_risk_level: MEDIUM
    - risk_level_changed: False
    - is_improved: False
    - summary_notes: ['Vulnerability score remained unchanged.']
    - timestamp: 2026-08-27 17:07:38
    - extra_metadata: {}

  ▶ Vector: deepfool
    - attack_name: deepfool
    - before_assessment: {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4074920823436263, 'f1_drop': 0.4836254784830271, 'confidence_drop': 0.5500268936157227, 'model_degradation': 0.4803814848141254, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 17:06:17', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.429787397384644}}
    - after_assessment: {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4074920823436263, 'f1_drop': 0.4836254784830271, 'confidence_drop': 0.5500268936157227, 'model_degradation': 0.4803814848141254, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 17:07:38', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.489156246185303}}
    - before_scoring: {'attack_name': 'deepfool', 'vulnerability_score': 33.63, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.04, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:06:18', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}}
    - after_scoring: {'attack_name': 'deepfool', 'vulnerability_score': 33.63, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.04, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:07:38', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}}
    - delta_attack_success_rate: None
    - delta_accuracy_drop: 0.0
    - delta_f1_drop: 0.0
    - delta_confidence_drop: 0.0
    - delta_model_degradation: 0.0
    - delta_vulnerability_score: 0.0
    - delta_clean_accuracy: 0.0
    - delta_adversarial_accuracy: 0.0
    - delta_perturbation: {'l0_mean': 0.0, 'l2_mean': 0.0, 'linf_mean': 0.0, 'is_estimated': 0.0}
    - before_risk_level: MEDIUM
    - after_risk_level: MEDIUM
    - risk_level_changed: False
    - is_improved: False
    - summary_notes: ['Vulnerability score remained unchanged.']
    - timestamp: 2026-08-27 17:07:38
    - extra_metadata: {}

  ▶ Vector: pgd
    - attack_name: pgd
    - before_assessment: {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.46237961644826, 'l0_mean': 0.9932491795546343, 'is_estimated': False}, 'accuracy_drop': 0.9699920823436263, 'f1_drop': 0.9311453197528684, 'confidence_drop': 0.19553357362747192, 'model_degradation': 0.6988903252413222, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.0, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.7981869578361511, 'timestamp': '2026-08-27 17:06:17', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1442525386810303}}
    - after_assessment: {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.46649667631584, 'l0_mean': 0.9932041297964498, 'is_estimated': False}, 'accuracy_drop': 0.9699920823436263, 'f1_drop': 0.9311453197528684, 'confidence_drop': 0.16048479080200195, 'model_degradation': 0.6872073976328322, 'clean_accuracy': 0.9699920823436263, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9311453197528684, 'adversarial_f1': 0.0, 'clean_confidence': 0.993720531463623, 'adversarial_confidence': 0.8332357406616211, 'timestamp': '2026-08-27 17:07:38', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.145221710205078}}
    - before_scoring: {'attack_name': 'pgd', 'vulnerability_score': 48.92, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 69.89, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:06:17', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}}
    - after_scoring: {'attack_name': 'pgd', 'vulnerability_score': 48.1, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 68.72, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:07:38', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 12630}}
    - delta_attack_success_rate: None
    - delta_accuracy_drop: 0.0
    - delta_f1_drop: 0.0
    - delta_confidence_drop: -0.035
    - delta_model_degradation: -0.0117
    - delta_vulnerability_score: -0.82
    - delta_clean_accuracy: 0.0
    - delta_adversarial_accuracy: 0.0
    - delta_perturbation: {'l0_mean': -0.0, 'l2_mean': 0.0041, 'linf_mean': 0.0, 'is_estimated': 0.0}
    - before_risk_level: MEDIUM
    - after_risk_level: MEDIUM
    - risk_level_changed: False
    - is_improved: True
    - summary_notes: ['Vulnerability score decreased by 0.82 points.']
    - timestamp: 2026-08-27 17:07:38
    - extra_metadata: {}

13. EXECUTION PERFORMANCE
------------------------------------------------------------------------
  Run Label    : AdverScan [full]
  Started At   : 2026-08-27 17:04:58
  Total Time   : 0.00s
  Overall      : UNKNOWN

  MODULE                         STATUS          TIME
  ········································································

14. RECOMMENDATIONS
------------------------------------------------------------------------
  [01] [MEDIUM] Vulnerability score 33.63 warrants attention. Implement input sanitization and monitor inference traffic for anomalies.
  [02] XAI attribution maps are available. Review highlighted input regions disproportionately targeted by adversarial perturbations to guide robustness patches.
  [03] Adversarial defense was applied. Validate post-hardening accuracy retention and conduct periodic re-tests to ensure defense durability.
  [04] Re-test indicates persistent vulnerability for vector 'fgsm'. Increase adversarial training epochs or broaden epsilon schedules.
  [05] Re-test indicates persistent vulnerability for vector 'deepfool'. Increase adversarial training epochs or broaden epsilon schedules.

15. FINAL SECURITY SUMMARY
------------------------------------------------------------------------
  - risk_level: MEDIUM
  - vulnerability_score: 33.63
  - baseline_accuracy: 97.00%
  - mean_adversarial_accuracy: 46.88%
  - attacks_evaluated: ['fgsm', 'pgd', 'deepfool']
  - hardening_applied: True
  - retest_conducted: True
  - total_recommendations: 5
  - primary_recommendation: [MEDIUM] Vulnerability score 33.63 warrants attention. Implement input sanitization and monitor inference traffic for anomalies.

========================================================================
  Generated by AdverScan — 2026-08-27 17:07:38
========================================================================