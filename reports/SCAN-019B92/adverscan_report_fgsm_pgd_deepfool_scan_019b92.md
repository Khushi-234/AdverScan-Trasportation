========================================================================
          ADVERSCAN SECURITY ASSESSMENT REPORT           
========================================================================
  Report ID   : RPT-0741EA03
  Scan ID     : SCAN-019B92
  Timestamp   : 2026-08-27 17:00:19
  Risk Level  : MEDIUM
  Vuln. Score : 33.73
========================================================================

1. EXECUTIVE SUMMARY
------------------------------------------------------------------------
  Scan ID            : SCAN-019B92
  Risk Level         : MEDIUM
  Vulnerability Score: 33.73
  Baseline Accuracy  : 97.20%
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
  - num_samples: 8000
  - num_classes: 43
  - accuracy: 97.20%
  - precision_macro: 94.17%
  - recall_macro: 93.59%
  - f1_macro: 93.31%
  - precision_weighted: 97.48%
  - recall_weighted: 97.20%
  - f1_weighted: 97.14%
  - average_confidence: 99.40%
  - average_entropy: 3.93%
  - per_class_metrics: {'0': {'precision': 0.96, 'recall': 1.0, 'f1': 0.9795918367346939, 'support': 48}, '1': {'precision': 1.0, 'recall': 0.9936842105263158, 'f1': 0.996832101372756, 'support': 475}, '2': {'precision': 0.9607438016528925, 'recall': 1.0, 'f1': 0.9799789251844047, 'support': 465}, '3': {'precision': 0.996031746031746, 'recall': 0.9691119691119691, 'f1': 0.9823874755381604, 'support': 259}, '4': {'precision': 0.9975609756097561, 'recall': 0.9951338199513382, 'f1': 0.9963459196102314, 'support': 411}, '5': {'precision': 0.9808153477218226, 'recall': 0.9903147699757869, 'f1': 0.9855421686746988, 'support': 413}, '6': {'precision': 1.0, 'recall': 0.9032258064516129, 'f1': 0.9491525423728814, 'support': 93}, '7': {'precision': 0.9965986394557823, 'recall': 0.9965986394557823, 'f1': 0.9965986394557823, 'support': 294}, '8': {'precision': 0.9962121212121212, 'recall': 0.9460431654676259, 'f1': 0.9704797047970479, 'support': 278}, '9': {'precision': 0.9964788732394366, 'recall': 1.0, 'f1': 0.9982363315696648, 'support': 283}, '10': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 427}, '11': {'precision': 0.9923371647509579, 'recall': 0.9773584905660377, 'f1': 0.9847908745247148, 'support': 265}, '12': {'precision': 0.9978494623655914, 'recall': 1.0, 'f1': 0.9989235737351991, 'support': 464}, '13': {'precision': 0.997737556561086, 'recall': 1.0, 'f1': 0.9988674971687429, 'support': 441}, '14': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 172}, '15': {'precision': 1.0, 'recall': 0.9922480620155039, 'f1': 0.9961089494163424, 'support': 129}, '16': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 94}, '17': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 241}, '18': {'precision': 0.995475113122172, 'recall': 0.9166666666666666, 'f1': 0.9544468546637744, 'support': 240}, '19': {'precision': 0.8095238095238095, 'recall': 0.4594594594594595, 'f1': 0.5862068965517241, 'support': 37}, '20': {'precision': 0.7083333333333334, 'recall': 0.9272727272727272, 'f1': 0.8031496062992126, 'support': 55}, '21': {'precision': 0.696969696969697, 'recall': 1.0, 'f1': 0.8214285714285714, 'support': 46}, '22': {'precision': 1.0, 'recall': 0.8695652173913043, 'f1': 0.9302325581395349, 'support': 69}, '23': {'precision': 0.9787234042553191, 'recall': 1.0, 'f1': 0.989247311827957, 'support': 92}, '24': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 46}, '25': {'precision': 0.9712460063897763, 'recall': 0.9806451612903225, 'f1': 0.9759229534510433, 'support': 310}, '26': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 107}, '27': {'precision': 0.926829268292683, 'recall': 0.95, 'f1': 0.9382716049382716, 'support': 40}, '28': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 101}, '29': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 61}, '30': {'precision': 0.9439252336448598, 'recall': 1.0, 'f1': 0.9711538461538461, 'support': 101}, '31': {'precision': 1.0, 'recall': 0.9943502824858758, 'f1': 0.9971671388101983, 'support': 177}, '32': {'precision': 0.8461538461538461, 'recall': 1.0, 'f1': 0.9166666666666666, 'support': 33}, '33': {'precision': 0.9574468085106383, 'recall': 0.6870229007633588, 'f1': 0.8, 'support': 131}, '34': {'precision': 0.6532258064516129, 'recall': 0.9642857142857143, 'f1': 0.7788461538461539, 'support': 84}, '35': {'precision': 0.9960474308300395, 'recall': 0.9921259842519685, 'f1': 0.9940828402366864, 'support': 254}, '36': {'precision': 0.7848101265822784, 'recall': 0.8266666666666667, 'f1': 0.8051948051948052, 'support': 75}, '37': {'precision': 0.6176470588235294, 'recall': 0.5384615384615384, 'f1': 0.5753424657534246, 'support': 39}, '38': {'precision': 0.9415584415584416, 'recall': 0.9954233409610984, 'f1': 0.967741935483871, 'support': 437}, '39': {'precision': 0.9629629629629629, 'recall': 0.49056603773584906, 'f1': 0.65, 'support': 53}, '40': {'precision': 0.984375, 'recall': 0.9545454545454546, 'f1': 0.9692307692307692, 'support': 66}, '41': {'precision': 0.9, 'recall': 1.0, 'f1': 0.9473684210526315, 'support': 36}, '42': {'precision': 0.9473684210526315, 'recall': 0.9310344827586207, 'f1': 0.9391304347826087, 'support': 58}}
  - confusion_matrix: [[48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 472, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 465, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 251, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 409, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 409, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 293, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 1, 1, 5, 0, 0, 263, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 283, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 427, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 259, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 464, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 441, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 172, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 220, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 2, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 304, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 41, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 81, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 252, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 13, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 17, 21, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 435, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 26, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 63, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 54]]
  - batch_size: 32
  - device: cuda
  - timestamp: 2026-08-27 16:58:53
  - extra_metadata: {}

5. ADVERSARIAL ATTACK RESULTS
------------------------------------------------------------------------
  ▶ FGSM
    execution_time_seconds   : 41.26%

  ▶ PGD
    execution_time_seconds   : 2.1179165840148926

  ▶ DEEPFOOL
    execution_time_seconds   : 22.234885215759277

6. VULNERABILITY ASSESSMENT
------------------------------------------------------------------------
  ▶ Vector: fgsm
    [Assessment] attack_name: fgsm
    [Assessment] dataset_name: bazyl/GTSRB
    [Assessment] num_samples: 8000
    [Assessment] attack_success_rate: None
    [Assessment] perturbation: {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}
    [Assessment] accuracy_drop: 0.597
    [Assessment] f1_drop: 0.6363352858701394
    [Assessment] confidence_drop: 0.5019688010215759
    [Assessment] model_degradation: 0.5784346956305718
    [Assessment] clean_accuracy: 0.972
    [Assessment] adversarial_accuracy: 0.375
    [Assessment] clean_f1: 0.9331318226666762
    [Assessment] adversarial_f1: 0.2967965367965368
    [Assessment] clean_confidence: 0.993995189666748
    [Assessment] adversarial_confidence: 0.4920263886451721
    [Assessment] timestamp: 2026-08-27 16:59:19
    [Scoring] attack_name: fgsm
    [Scoring] vulnerability_score: 40.49
    [Scoring] risk_level: MEDIUM
    [Scoring] sub_scores: {'asr_score': None, 'degradation_score': 57.84, 'stealth_score': 0.0}
    [Scoring] timestamp: 2026-08-27 16:59:19
    [Scoring] metadata: {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}

  ▶ Vector: pgd
    [Assessment] attack_name: pgd
    [Assessment] dataset_name: bazyl/GTSRB
    [Assessment] num_samples: 8000
    [Assessment] attack_success_rate: None
    [Assessment] perturbation: {'linf_mean': 0.9169117696583271, 'l2_mean': 200.4627685131295, 'l0_mean': 0.9933492439944729, 'is_estimated': False}
    [Assessment] accuracy_drop: 0.972
    [Assessment] f1_drop: 0.9331318226666762
    [Assessment] confidence_drop: 0.20490765571594238
    [Assessment] model_degradation: 0.7033464927942061
    [Assessment] clean_accuracy: 0.972
    [Assessment] adversarial_accuracy: 0.0
    [Assessment] clean_f1: 0.9331318226666762
    [Assessment] adversarial_f1: 0.0
    [Assessment] clean_confidence: 0.993995189666748
    [Assessment] adversarial_confidence: 0.7890875339508057
    [Assessment] timestamp: 2026-08-27 16:59:19
    [Scoring] attack_name: pgd
    [Scoring] vulnerability_score: 49.23
    [Scoring] risk_level: MEDIUM
    [Scoring] sub_scores: {'asr_score': None, 'degradation_score': 70.33, 'stealth_score': 0.0}
    [Scoring] timestamp: 2026-08-27 16:59:19
    [Scoring] metadata: {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}

  ▶ Vector: deepfool
    [Assessment] attack_name: deepfool
    [Assessment] dataset_name: bazyl/GTSRB
    [Assessment] num_samples: 8000
    [Assessment] attack_success_rate: None
    [Assessment] perturbation: {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}
    [Assessment] accuracy_drop: 0.4095
    [Assessment] f1_drop: 0.48561198139683487
    [Assessment] confidence_drop: 0.5503015518188477
    [Assessment] model_degradation: 0.4818045110718942
    [Assessment] clean_accuracy: 0.972
    [Assessment] adversarial_accuracy: 0.5625
    [Assessment] clean_f1: 0.9331318226666762
    [Assessment] adversarial_f1: 0.4475198412698413
    [Assessment] clean_confidence: 0.993995189666748
    [Assessment] adversarial_confidence: 0.4436936378479004
    [Assessment] timestamp: 2026-08-27 16:59:19
    [Scoring] attack_name: deepfool
    [Scoring] vulnerability_score: 33.73
    [Scoring] risk_level: MEDIUM
    [Scoring] sub_scores: {'asr_score': None, 'degradation_score': 48.18, 'stealth_score': 0.0}
    [Scoring] timestamp: 2026-08-27 16:59:19
    [Scoring] metadata: {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}

7. VULNERABILITY SCORE & RISK LEVEL
------------------------------------------------------------------------
  Overall Vulnerability Score : 33.73
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
    - metadata: {'attack_name': 'fgsm', 'technique': 'shap', 'assessment_result': {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.597, 'f1_drop': 0.6363352858701394, 'confidence_drop': 0.5019688010215759, 'model_degradation': 0.5784346956305718, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.4920263886451721, 'timestamp': '2026-08-27 16:59:19', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.41261959075927734}}}

  ▶ Technique: pgd_shap
    - attack_name: pgd
    - technique: shap
    - clean_prediction: [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17]
    - adversarial_prediction: [10, 5, 12, 38, 13, 12, 26, 32, 18, 41, 25, 17, 13, 15, 38, 38, 19, 12, 25, 11, 13, 9, 32, 38, 5, 25, 12, 13, 12, 42, 6, 13]
    - clean_confidence: 0.998881459236145
    - adversarial_confidence: 0.7889636754989624
    - prediction_changed: True
    - true_label: tensor([16,  1, 38, 33, 11, 38, 18, 12, 25, 35, 12,  7, 23,  7,  4,  9, 21, 20,
        27, 38,  4, 33,  9,  3,  1, 11, 13, 10,  9, 11,  5, 17])
    - attack_caused_failure: False
    - attribution: {'technique': 'shap', 'clean': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}, 'adversarial': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}}
    - comparison: {'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [10, 5, 12, 38, 13, 12, 26, 32, 18, 41, 25, 17, 13, 15, 38, 38, 19, 12, 25, 11, 13, 9, 32, 38, 5, 25, 12, 13, 12, 42, 6, 13], 'clean_confidence': 0.998881459236145, 'adversarial_confidence': 0.7889636754989624, 'prediction_changed': True, 'confidence_difference': 0.20991778373718262, 'attribution_comparison_status': 'unavailable', 'attribution_l1': None, 'attribution_l2': None, 'attribution_cosine_similarity': None, 'attribution_mean_difference': None}
    - failure_analysis: {'clean_correct': False, 'adversarial_correct': False, 'prediction_changed': True, 'attack_caused_failure': False, 'true_label': [16, 1, 38, 33, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [10, 5, 12, 38, 13, 12, 26, 32, 18, 41, 25, 17, 13, 15, 38, 38, 19, 12, 25, 11, 13, 9, 32, 38, 5, 25, 12, 13, 12, 42, 6, 13], 'failure_mode': 'clean_incorrect_to_adversarial_incorrect'}
    - metadata: {'attack_name': 'pgd', 'technique': 'shap', 'assessment_result': {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.4627685131295, 'l0_mean': 0.9933492439944729, 'is_estimated': False}, 'accuracy_drop': 0.972, 'f1_drop': 0.9331318226666762, 'confidence_drop': 0.20490765571594238, 'model_degradation': 0.7033464927942061, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.0, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.7890875339508057, 'timestamp': '2026-08-27 16:59:19', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1179165840148926}}}

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
    - metadata: {'attack_name': 'deepfool', 'technique': 'shap', 'assessment_result': {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4095, 'f1_drop': 0.48561198139683487, 'confidence_drop': 0.5503015518188477, 'model_degradation': 0.4818045110718942, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 16:59:19', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.234885215759277}}}

10. HARDENING
------------------------------------------------------------------------
  - metadata: {'defense_name': 'spatial_smoothing', 'defense_type': 'preprocessing', 'parameters': {'kernel_size': 3, 'sigma': 1.0}, 'execution_time_seconds': 0.027561187744140625, 'timestamp': '2026-08-27 16:59:19', 'extra_metadata': {}}
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
  - num_samples: 8000
  - before_baseline_evaluation: {'dataset_name': 'bazyl/GTSRB', 'model_name': 'GTSRB_ViT_Demo', 'num_samples': 8000, 'num_classes': 43, 'accuracy': 0.972, 'precision_macro': 0.9417438943502051, 'recall_macro': 0.9358560597329906, 'f1_macro': 0.9331318226666762, 'precision_weighted': 0.9747749046568499, 'recall_weighted': 0.972, 'f1_weighted': 0.9714331977474795, 'average_confidence': 0.993995189666748, 'average_entropy': 0.039279449731111526, 'per_class_metrics': {'0': {'precision': 0.96, 'recall': 1.0, 'f1': 0.9795918367346939, 'support': 48}, '1': {'precision': 1.0, 'recall': 0.9936842105263158, 'f1': 0.996832101372756, 'support': 475}, '2': {'precision': 0.9607438016528925, 'recall': 1.0, 'f1': 0.9799789251844047, 'support': 465}, '3': {'precision': 0.996031746031746, 'recall': 0.9691119691119691, 'f1': 0.9823874755381604, 'support': 259}, '4': {'precision': 0.9975609756097561, 'recall': 0.9951338199513382, 'f1': 0.9963459196102314, 'support': 411}, '5': {'precision': 0.9808153477218226, 'recall': 0.9903147699757869, 'f1': 0.9855421686746988, 'support': 413}, '6': {'precision': 1.0, 'recall': 0.9032258064516129, 'f1': 0.9491525423728814, 'support': 93}, '7': {'precision': 0.9965986394557823, 'recall': 0.9965986394557823, 'f1': 0.9965986394557823, 'support': 294}, '8': {'precision': 0.9962121212121212, 'recall': 0.9460431654676259, 'f1': 0.9704797047970479, 'support': 278}, '9': {'precision': 0.9964788732394366, 'recall': 1.0, 'f1': 0.9982363315696648, 'support': 283}, '10': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 427}, '11': {'precision': 0.9923371647509579, 'recall': 0.9773584905660377, 'f1': 0.9847908745247148, 'support': 265}, '12': {'precision': 0.9978494623655914, 'recall': 1.0, 'f1': 0.9989235737351991, 'support': 464}, '13': {'precision': 0.997737556561086, 'recall': 1.0, 'f1': 0.9988674971687429, 'support': 441}, '14': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 172}, '15': {'precision': 1.0, 'recall': 0.9922480620155039, 'f1': 0.9961089494163424, 'support': 129}, '16': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 94}, '17': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 241}, '18': {'precision': 0.995475113122172, 'recall': 0.9166666666666666, 'f1': 0.9544468546637744, 'support': 240}, '19': {'precision': 0.8095238095238095, 'recall': 0.4594594594594595, 'f1': 0.5862068965517241, 'support': 37}, '20': {'precision': 0.7083333333333334, 'recall': 0.9272727272727272, 'f1': 0.8031496062992126, 'support': 55}, '21': {'precision': 0.696969696969697, 'recall': 1.0, 'f1': 0.8214285714285714, 'support': 46}, '22': {'precision': 1.0, 'recall': 0.8695652173913043, 'f1': 0.9302325581395349, 'support': 69}, '23': {'precision': 0.9787234042553191, 'recall': 1.0, 'f1': 0.989247311827957, 'support': 92}, '24': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 46}, '25': {'precision': 0.9712460063897763, 'recall': 0.9806451612903225, 'f1': 0.9759229534510433, 'support': 310}, '26': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 107}, '27': {'precision': 0.926829268292683, 'recall': 0.95, 'f1': 0.9382716049382716, 'support': 40}, '28': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 101}, '29': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 61}, '30': {'precision': 0.9439252336448598, 'recall': 1.0, 'f1': 0.9711538461538461, 'support': 101}, '31': {'precision': 1.0, 'recall': 0.9943502824858758, 'f1': 0.9971671388101983, 'support': 177}, '32': {'precision': 0.8461538461538461, 'recall': 1.0, 'f1': 0.9166666666666666, 'support': 33}, '33': {'precision': 0.9574468085106383, 'recall': 0.6870229007633588, 'f1': 0.8, 'support': 131}, '34': {'precision': 0.6532258064516129, 'recall': 0.9642857142857143, 'f1': 0.7788461538461539, 'support': 84}, '35': {'precision': 0.9960474308300395, 'recall': 0.9921259842519685, 'f1': 0.9940828402366864, 'support': 254}, '36': {'precision': 0.7848101265822784, 'recall': 0.8266666666666667, 'f1': 0.8051948051948052, 'support': 75}, '37': {'precision': 0.6176470588235294, 'recall': 0.5384615384615384, 'f1': 0.5753424657534246, 'support': 39}, '38': {'precision': 0.9415584415584416, 'recall': 0.9954233409610984, 'f1': 0.967741935483871, 'support': 437}, '39': {'precision': 0.9629629629629629, 'recall': 0.49056603773584906, 'f1': 0.65, 'support': 53}, '40': {'precision': 0.984375, 'recall': 0.9545454545454546, 'f1': 0.9692307692307692, 'support': 66}, '41': {'precision': 0.9, 'recall': 1.0, 'f1': 0.9473684210526315, 'support': 36}, '42': {'precision': 0.9473684210526315, 'recall': 0.9310344827586207, 'f1': 0.9391304347826087, 'support': 58}}, 'confusion_matrix': [[48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 472, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 465, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 251, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 409, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 409, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 293, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 1, 1, 5, 0, 0, 263, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 283, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 427, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 259, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 464, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 441, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 172, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 220, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 2, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 304, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 41, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 81, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 252, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 13, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 17, 21, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 435, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 26, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 63, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 54]], 'batch_size': 32, 'device': 'cuda', 'timestamp': '2026-08-27 16:58:53', 'extra_metadata': {}}
  - after_baseline_evaluation: {'dataset_name': 'bazyl/GTSRB', 'model_name': 'GTSRB_ViT_Demo', 'num_samples': 8000, 'num_classes': 43, 'accuracy': 0.972, 'precision_macro': 0.9417438943502051, 'recall_macro': 0.9358560597329906, 'f1_macro': 0.9331318226666762, 'precision_weighted': 0.9747749046568499, 'recall_weighted': 0.972, 'f1_weighted': 0.9714331977474795, 'average_confidence': 0.993995189666748, 'average_entropy': 0.039279449731111526, 'per_class_metrics': {'0': {'precision': 0.96, 'recall': 1.0, 'f1': 0.9795918367346939, 'support': 48}, '1': {'precision': 1.0, 'recall': 0.9936842105263158, 'f1': 0.996832101372756, 'support': 475}, '2': {'precision': 0.9607438016528925, 'recall': 1.0, 'f1': 0.9799789251844047, 'support': 465}, '3': {'precision': 0.996031746031746, 'recall': 0.9691119691119691, 'f1': 0.9823874755381604, 'support': 259}, '4': {'precision': 0.9975609756097561, 'recall': 0.9951338199513382, 'f1': 0.9963459196102314, 'support': 411}, '5': {'precision': 0.9808153477218226, 'recall': 0.9903147699757869, 'f1': 0.9855421686746988, 'support': 413}, '6': {'precision': 1.0, 'recall': 0.9032258064516129, 'f1': 0.9491525423728814, 'support': 93}, '7': {'precision': 0.9965986394557823, 'recall': 0.9965986394557823, 'f1': 0.9965986394557823, 'support': 294}, '8': {'precision': 0.9962121212121212, 'recall': 0.9460431654676259, 'f1': 0.9704797047970479, 'support': 278}, '9': {'precision': 0.9964788732394366, 'recall': 1.0, 'f1': 0.9982363315696648, 'support': 283}, '10': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 427}, '11': {'precision': 0.9923371647509579, 'recall': 0.9773584905660377, 'f1': 0.9847908745247148, 'support': 265}, '12': {'precision': 0.9978494623655914, 'recall': 1.0, 'f1': 0.9989235737351991, 'support': 464}, '13': {'precision': 0.997737556561086, 'recall': 1.0, 'f1': 0.9988674971687429, 'support': 441}, '14': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 172}, '15': {'precision': 1.0, 'recall': 0.9922480620155039, 'f1': 0.9961089494163424, 'support': 129}, '16': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 94}, '17': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 241}, '18': {'precision': 0.995475113122172, 'recall': 0.9166666666666666, 'f1': 0.9544468546637744, 'support': 240}, '19': {'precision': 0.8095238095238095, 'recall': 0.4594594594594595, 'f1': 0.5862068965517241, 'support': 37}, '20': {'precision': 0.7083333333333334, 'recall': 0.9272727272727272, 'f1': 0.8031496062992126, 'support': 55}, '21': {'precision': 0.696969696969697, 'recall': 1.0, 'f1': 0.8214285714285714, 'support': 46}, '22': {'precision': 1.0, 'recall': 0.8695652173913043, 'f1': 0.9302325581395349, 'support': 69}, '23': {'precision': 0.9787234042553191, 'recall': 1.0, 'f1': 0.989247311827957, 'support': 92}, '24': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 46}, '25': {'precision': 0.9712460063897763, 'recall': 0.9806451612903225, 'f1': 0.9759229534510433, 'support': 310}, '26': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 107}, '27': {'precision': 0.926829268292683, 'recall': 0.95, 'f1': 0.9382716049382716, 'support': 40}, '28': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 101}, '29': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 61}, '30': {'precision': 0.9439252336448598, 'recall': 1.0, 'f1': 0.9711538461538461, 'support': 101}, '31': {'precision': 1.0, 'recall': 0.9943502824858758, 'f1': 0.9971671388101983, 'support': 177}, '32': {'precision': 0.8461538461538461, 'recall': 1.0, 'f1': 0.9166666666666666, 'support': 33}, '33': {'precision': 0.9574468085106383, 'recall': 0.6870229007633588, 'f1': 0.8, 'support': 131}, '34': {'precision': 0.6532258064516129, 'recall': 0.9642857142857143, 'f1': 0.7788461538461539, 'support': 84}, '35': {'precision': 0.9960474308300395, 'recall': 0.9921259842519685, 'f1': 0.9940828402366864, 'support': 254}, '36': {'precision': 0.7848101265822784, 'recall': 0.8266666666666667, 'f1': 0.8051948051948052, 'support': 75}, '37': {'precision': 0.6176470588235294, 'recall': 0.5384615384615384, 'f1': 0.5753424657534246, 'support': 39}, '38': {'precision': 0.9415584415584416, 'recall': 0.9954233409610984, 'f1': 0.967741935483871, 'support': 437}, '39': {'precision': 0.9629629629629629, 'recall': 0.49056603773584906, 'f1': 0.65, 'support': 53}, '40': {'precision': 0.984375, 'recall': 0.9545454545454546, 'f1': 0.9692307692307692, 'support': 66}, '41': {'precision': 0.9, 'recall': 1.0, 'f1': 0.9473684210526315, 'support': 36}, '42': {'precision': 0.9473684210526315, 'recall': 0.9310344827586207, 'f1': 0.9391304347826087, 'support': 58}}, 'confusion_matrix': [[48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 472, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 465, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 251, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 409, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 409, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 293, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 1, 1, 5, 0, 0, 263, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 283, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 427, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 259, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 464, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 441, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 172, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 220, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 2, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 304, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 41, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 81, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 252, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 13, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 17, 21, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 435, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 26, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 63, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 54]], 'batch_size': 32, 'device': 'cuda', 'timestamp': '2026-08-27 16:59:53', 'extra_metadata': {}}
  - before_attack_results: {'fgsm': {'attack_name': 'fgsm', 'attack_class': 'FGSM', 'execution_time_seconds': 0.41261959075927734, 'parameters': {}}, 'pgd': {'attack_name': 'pgd', 'attack_class': 'PGD', 'execution_time_seconds': 2.1179165840148926, 'parameters': {}}, 'deepfool': {'attack_name': 'deepfool', 'attack_class': 'DeepFool', 'execution_time_seconds': 22.234885215759277, 'parameters': {}}}
  - after_attack_results: {'fgsm': {'attack_name': 'fgsm', 'attack_class': 'FGSM', 'execution_time_seconds': 0.021084070205688477, 'parameters': {}}, 'pgd': {'attack_name': 'pgd', 'attack_class': 'PGD', 'execution_time_seconds': 2.128424882888794, 'parameters': {}}, 'deepfool': {'attack_name': 'deepfool', 'attack_class': 'DeepFool', 'execution_time_seconds': 22.357267379760742, 'parameters': {}}}
  - before_vulnerability_analysis: {'fgsm': {'assessment': {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.597, 'f1_drop': 0.6363352858701394, 'confidence_drop': 0.5019688010215759, 'model_degradation': 0.5784346956305718, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.4920263886451721, 'timestamp': '2026-08-27 16:59:19', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.41261959075927734}}, 'scoring': {'attack_name': 'fgsm', 'vulnerability_score': 40.49, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.84, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:59:19', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}}}, 'pgd': {'assessment': {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.4627685131295, 'l0_mean': 0.9933492439944729, 'is_estimated': False}, 'accuracy_drop': 0.972, 'f1_drop': 0.9331318226666762, 'confidence_drop': 0.20490765571594238, 'model_degradation': 0.7033464927942061, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.0, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.7890875339508057, 'timestamp': '2026-08-27 16:59:19', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1179165840148926}}, 'scoring': {'attack_name': 'pgd', 'vulnerability_score': 49.23, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 70.33, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:59:19', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}}}, 'deepfool': {'assessment': {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4095, 'f1_drop': 0.48561198139683487, 'confidence_drop': 0.5503015518188477, 'model_degradation': 0.4818045110718942, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 16:59:19', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.234885215759277}}, 'scoring': {'attack_name': 'deepfool', 'vulnerability_score': 33.73, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.18, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:59:19', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}}}}
  - after_vulnerability_analysis: {'fgsm': {'assessment': {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.597, 'f1_drop': 0.6363352858701394, 'confidence_drop': 0.5019687414169312, 'model_degradation': 0.5784346757623569, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.4920264482498169, 'timestamp': '2026-08-27 17:00:19', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.021084070205688477}}, 'scoring': {'attack_name': 'fgsm', 'vulnerability_score': 40.49, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.84, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:00:19', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}}}, 'pgd': {'assessment': {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.46123441857014, 'l0_mean': 0.9932657877604166, 'is_estimated': False}, 'accuracy_drop': 0.972, 'f1_drop': 0.9331318226666762, 'confidence_drop': 0.21495389938354492, 'model_degradation': 0.7066952406834069, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.0, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.7790412902832031, 'timestamp': '2026-08-27 17:00:19', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.128424882888794}}, 'scoring': {'attack_name': 'pgd', 'vulnerability_score': 49.47, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 70.67, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:00:19', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}}}, 'deepfool': {'assessment': {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4095, 'f1_drop': 0.48561198139683487, 'confidence_drop': 0.5503015518188477, 'model_degradation': 0.4818045110718942, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 17:00:19', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.357267379760742}}, 'scoring': {'attack_name': 'deepfool', 'vulnerability_score': 33.73, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.18, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:00:19', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}}}}
  - overall_improved: False
  - timestamp: 2026-08-27 16:59:19
  - execution_time_seconds: 59.8201
  - extra_metadata: {}

12. BEFORE VS AFTER COMPARISON
------------------------------------------------------------------------
  ▶ Vector: pgd
    - attack_name: pgd
    - before_assessment: {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.4627685131295, 'l0_mean': 0.9933492439944729, 'is_estimated': False}, 'accuracy_drop': 0.972, 'f1_drop': 0.9331318226666762, 'confidence_drop': 0.20490765571594238, 'model_degradation': 0.7033464927942061, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.0, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.7890875339508057, 'timestamp': '2026-08-27 16:59:19', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1179165840148926}}
    - after_assessment: {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.46123441857014, 'l0_mean': 0.9932657877604166, 'is_estimated': False}, 'accuracy_drop': 0.972, 'f1_drop': 0.9331318226666762, 'confidence_drop': 0.21495389938354492, 'model_degradation': 0.7066952406834069, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.0, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.7790412902832031, 'timestamp': '2026-08-27 17:00:19', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.128424882888794}}
    - before_scoring: {'attack_name': 'pgd', 'vulnerability_score': 49.23, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 70.33, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:59:19', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}}
    - after_scoring: {'attack_name': 'pgd', 'vulnerability_score': 49.47, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 70.67, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:00:19', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}}
    - delta_attack_success_rate: None
    - delta_accuracy_drop: 0.0
    - delta_f1_drop: 0.0
    - delta_confidence_drop: 0.01
    - delta_model_degradation: 0.0033
    - delta_vulnerability_score: 0.24
    - delta_clean_accuracy: 0.0
    - delta_adversarial_accuracy: 0.0
    - delta_perturbation: {'l0_mean': -0.0001, 'linf_mean': 0.0, 'l2_mean': -0.0015, 'is_estimated': 0.0}
    - before_risk_level: MEDIUM
    - after_risk_level: MEDIUM
    - risk_level_changed: False
    - is_improved: False
    - summary_notes: ['Vulnerability score increased by 0.24 points.']
    - timestamp: 2026-08-27 17:00:19
    - extra_metadata: {}

  ▶ Vector: fgsm
    - attack_name: fgsm
    - before_assessment: {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.597, 'f1_drop': 0.6363352858701394, 'confidence_drop': 0.5019688010215759, 'model_degradation': 0.5784346956305718, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.4920263886451721, 'timestamp': '2026-08-27 16:59:19', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.41261959075927734}}
    - after_assessment: {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.597, 'f1_drop': 0.6363352858701394, 'confidence_drop': 0.5019687414169312, 'model_degradation': 0.5784346757623569, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.4920264482498169, 'timestamp': '2026-08-27 17:00:19', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.021084070205688477}}
    - before_scoring: {'attack_name': 'fgsm', 'vulnerability_score': 40.49, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.84, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:59:19', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}}
    - after_scoring: {'attack_name': 'fgsm', 'vulnerability_score': 40.49, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.84, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:00:19', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}}
    - delta_attack_success_rate: None
    - delta_accuracy_drop: 0.0
    - delta_f1_drop: 0.0
    - delta_confidence_drop: -0.0
    - delta_model_degradation: -0.0
    - delta_vulnerability_score: 0.0
    - delta_clean_accuracy: 0.0
    - delta_adversarial_accuracy: 0.0
    - delta_perturbation: {'l0_mean': 0.0, 'linf_mean': 0.0, 'l2_mean': 0.0, 'is_estimated': 0.0}
    - before_risk_level: MEDIUM
    - after_risk_level: MEDIUM
    - risk_level_changed: False
    - is_improved: False
    - summary_notes: ['Vulnerability score remained unchanged.']
    - timestamp: 2026-08-27 17:00:19
    - extra_metadata: {}

  ▶ Vector: deepfool
    - attack_name: deepfool
    - before_assessment: {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4095, 'f1_drop': 0.48561198139683487, 'confidence_drop': 0.5503015518188477, 'model_degradation': 0.4818045110718942, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 16:59:19', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.234885215759277}}
    - after_assessment: {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4095, 'f1_drop': 0.48561198139683487, 'confidence_drop': 0.5503015518188477, 'model_degradation': 0.4818045110718942, 'clean_accuracy': 0.972, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9331318226666762, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.993995189666748, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 17:00:19', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.357267379760742}}
    - before_scoring: {'attack_name': 'deepfool', 'vulnerability_score': 33.73, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.18, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:59:19', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}}
    - after_scoring: {'attack_name': 'deepfool', 'vulnerability_score': 33.73, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.18, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:00:19', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 8000}}
    - delta_attack_success_rate: None
    - delta_accuracy_drop: 0.0
    - delta_f1_drop: 0.0
    - delta_confidence_drop: 0.0
    - delta_model_degradation: 0.0
    - delta_vulnerability_score: 0.0
    - delta_clean_accuracy: 0.0
    - delta_adversarial_accuracy: 0.0
    - delta_perturbation: {'l0_mean': 0.0, 'linf_mean': 0.0, 'l2_mean': 0.0, 'is_estimated': 0.0}
    - before_risk_level: MEDIUM
    - after_risk_level: MEDIUM
    - risk_level_changed: False
    - is_improved: False
    - summary_notes: ['Vulnerability score remained unchanged.']
    - timestamp: 2026-08-27 17:00:19
    - extra_metadata: {}

13. EXECUTION PERFORMANCE
------------------------------------------------------------------------
  Run Label    : AdverScan [full]
  Started At   : 2026-08-27 16:58:19
  Total Time   : 0.00s
  Overall      : UNKNOWN

  MODULE                         STATUS          TIME
  ········································································

14. RECOMMENDATIONS
------------------------------------------------------------------------
  [01] [MEDIUM] Vulnerability score 33.73 warrants attention. Implement input sanitization and monitor inference traffic for anomalies.
  [02] XAI attribution maps are available. Review highlighted input regions disproportionately targeted by adversarial perturbations to guide robustness patches.
  [03] Adversarial defense was applied. Validate post-hardening accuracy retention and conduct periodic re-tests to ensure defense durability.
  [04] Re-test indicates persistent vulnerability for vector 'pgd'. Increase adversarial training epochs or broaden epsilon schedules.
  [05] Re-test indicates persistent vulnerability for vector 'fgsm'. Increase adversarial training epochs or broaden epsilon schedules.
  [06] Re-test indicates persistent vulnerability for vector 'deepfool'. Increase adversarial training epochs or broaden epsilon schedules.

15. FINAL SECURITY SUMMARY
------------------------------------------------------------------------
  - risk_level: MEDIUM
  - vulnerability_score: 33.73
  - baseline_accuracy: 97.20%
  - mean_adversarial_accuracy: 46.88%
  - attacks_evaluated: ['fgsm', 'pgd', 'deepfool']
  - hardening_applied: True
  - retest_conducted: True
  - total_recommendations: 6
  - primary_recommendation: [MEDIUM] Vulnerability score 33.73 warrants attention. Implement input sanitization and monitor inference traffic for anomalies.

========================================================================
  Generated by AdverScan — 2026-08-27 17:00:19
========================================================================