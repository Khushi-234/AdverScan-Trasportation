========================================================================
          ADVERSCAN SECURITY ASSESSMENT REPORT           
========================================================================
  Report ID   : RPT-16C42529
  Scan ID     : SCAN-861FB8
  Timestamp   : 2026-08-27 16:28:23
  Risk Level  : MEDIUM
  Vuln. Score : 33.61
========================================================================

1. EXECUTIVE SUMMARY
------------------------------------------------------------------------
  Scan ID            : SCAN-861FB8
  Risk Level         : MEDIUM
  Vulnerability Score: 33.61
  Baseline Accuracy  : 97.18%
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
  - num_samples: 5000
  - num_classes: 43
  - accuracy: 97.18%
  - precision_macro: 93.93%
  - recall_macro: 93.10%
  - f1_macro: 92.84%
  - precision_weighted: 97.45%
  - recall_weighted: 97.18%
  - f1_weighted: 97.08%
  - average_confidence: 99.39%
  - average_entropy: 3.97%
  - per_class_metrics: {'0': {'precision': 0.95, 'recall': 1.0, 'f1': 0.9743589743589743, 'support': 38}, '1': {'precision': 1.0, 'recall': 0.9933554817275747, 'f1': 0.9966666666666667, 'support': 301}, '2': {'precision': 0.9606557377049181, 'recall': 1.0, 'f1': 0.979933110367893, 'support': 293}, '3': {'precision': 0.993103448275862, 'recall': 0.96, 'f1': 0.976271186440678, 'support': 150}, '4': {'precision': 1.0, 'recall': 0.9921875, 'f1': 0.996078431372549, 'support': 256}, '5': {'precision': 0.9762845849802372, 'recall': 0.988, 'f1': 0.9821073558648111, 'support': 250}, '6': {'precision': 1.0, 'recall': 0.9074074074074074, 'f1': 0.9514563106796117, 'support': 54}, '7': {'precision': 0.9947916666666666, 'recall': 1.0, 'f1': 0.9973890339425587, 'support': 191}, '8': {'precision': 1.0, 'recall': 0.9476744186046512, 'f1': 0.9731343283582089, 'support': 172}, '9': {'precision': 0.9945652173913043, 'recall': 1.0, 'f1': 0.997275204359673, 'support': 183}, '10': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 274}, '11': {'precision': 1.0, 'recall': 0.9757575757575757, 'f1': 0.9877300613496932, 'support': 165}, '12': {'precision': 0.9965635738831615, 'recall': 1.0, 'f1': 0.9982788296041308, 'support': 290}, '13': {'precision': 0.9964539007092199, 'recall': 1.0, 'f1': 0.9982238010657194, 'support': 281}, '14': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 112}, '15': {'precision': 1.0, 'recall': 0.9880952380952381, 'f1': 0.9940119760479041, 'support': 84}, '16': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 62}, '17': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 158}, '18': {'precision': 0.9924242424242424, 'recall': 0.9225352112676056, 'f1': 0.9562043795620438, 'support': 142}, '19': {'precision': 0.8, 'recall': 0.32, 'f1': 0.45714285714285713, 'support': 25}, '20': {'precision': 0.6470588235294118, 'recall': 0.9428571428571428, 'f1': 0.7674418604651163, 'support': 35}, '21': {'precision': 0.7317073170731707, 'recall': 1.0, 'f1': 0.8450704225352113, 'support': 30}, '22': {'precision': 1.0, 'recall': 0.9761904761904762, 'f1': 0.9879518072289156, 'support': 42}, '23': {'precision': 0.9821428571428571, 'recall': 1.0, 'f1': 0.990990990990991, 'support': 55}, '24': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 28}, '25': {'precision': 0.9900990099009901, 'recall': 0.9900990099009901, 'f1': 0.9900990099009901, 'support': 202}, '26': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 60}, '27': {'precision': 0.9583333333333334, 'recall': 0.92, 'f1': 0.9387755102040817, 'support': 25}, '28': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 63}, '29': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 41}, '30': {'precision': 0.9411764705882353, 'recall': 1.0, 'f1': 0.9696969696969697, 'support': 64}, '31': {'precision': 1.0, 'recall': 0.9906542056074766, 'f1': 0.9953051643192489, 'support': 107}, '32': {'precision': 0.9130434782608695, 'recall': 1.0, 'f1': 0.9545454545454546, 'support': 21}, '33': {'precision': 0.9444444444444444, 'recall': 0.6538461538461539, 'f1': 0.7727272727272727, 'support': 78}, '34': {'precision': 0.654320987654321, 'recall': 0.9636363636363636, 'f1': 0.7794117647058824, 'support': 55}, '35': {'precision': 1.0, 'recall': 0.9935064935064936, 'f1': 0.996742671009772, 'support': 154}, '36': {'precision': 0.782608695652174, 'recall': 0.8372093023255814, 'f1': 0.8089887640449438, 'support': 43}, '37': {'precision': 0.5, 'recall': 0.3888888888888889, 'f1': 0.4375, 'support': 18}, '38': {'precision': 0.9314079422382672, 'recall': 0.9961389961389961, 'f1': 0.9626865671641791, 'support': 259}, '39': {'precision': 0.9545454545454546, 'recall': 0.525, 'f1': 0.6774193548387096, 'support': 40}, '40': {'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 40}, '41': {'precision': 0.896551724137931, 'recall': 1.0, 'f1': 0.9454545454545454, 'support': 26}, '42': {'precision': 0.9090909090909091, 'recall': 0.9090909090909091, 'f1': 0.9090909090909091, 'support': 33}}
  - confusion_matrix: [[38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 299, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 293, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 5, 144, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 247, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 4, 1, 0, 4, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 183, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 274, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 290, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 281, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 158, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 200, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 27, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 53, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 153, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 10, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 258, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 21, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 30]]
  - batch_size: 32
  - device: cuda
  - timestamp: 2026-08-27 16:27:09
  - extra_metadata: {}

5. ADVERSARIAL ATTACK RESULTS
------------------------------------------------------------------------
  ▶ FGSM
    execution_time_seconds   : 41.03%

  ▶ PGD
    execution_time_seconds   : 2.1138906478881836

  ▶ DEEPFOOL
    execution_time_seconds   : 22.137073278427124

6. VULNERABILITY ASSESSMENT
------------------------------------------------------------------------
  ▶ Vector: fgsm
    [Assessment] attack_name: fgsm
    [Assessment] dataset_name: bazyl/GTSRB
    [Assessment] num_samples: 5000
    [Assessment] attack_success_rate: None
    [Assessment] perturbation: {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}
    [Assessment] accuracy_drop: 0.5968
    [Assessment] f1_drop: 0.6315876613538385
    [Assessment] confidence_drop: 0.5018666982650757
    [Assessment] model_degradation: 0.5767514532063047
    [Assessment] clean_accuracy: 0.9718
    [Assessment] adversarial_accuracy: 0.375
    [Assessment] clean_f1: 0.9283841981503752
    [Assessment] adversarial_f1: 0.2967965367965368
    [Assessment] clean_confidence: 0.9938931465148926
    [Assessment] adversarial_confidence: 0.4920264482498169
    [Assessment] timestamp: 2026-08-27 16:27:35
    [Scoring] attack_name: fgsm
    [Scoring] vulnerability_score: 40.37
    [Scoring] risk_level: MEDIUM
    [Scoring] sub_scores: {'asr_score': None, 'degradation_score': 57.68, 'stealth_score': 0.0}
    [Scoring] timestamp: 2026-08-27 16:27:35
    [Scoring] metadata: {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}

  ▶ Vector: pgd
    [Assessment] attack_name: pgd
    [Assessment] dataset_name: bazyl/GTSRB
    [Assessment] num_samples: 5000
    [Assessment] attack_success_rate: None
    [Assessment] perturbation: {'linf_mean': 0.9169117696583271, 'l2_mean': 200.46013909524748, 'l0_mean': 0.9932593520806761, 'is_estimated': False}
    [Assessment] accuracy_drop: 0.9718
    [Assessment] f1_drop: 0.9283841981503752
    [Assessment] confidence_drop: 0.2035258412361145
    [Assessment] model_degradation: 0.7012366797954965
    [Assessment] clean_accuracy: 0.9718
    [Assessment] adversarial_accuracy: 0.0
    [Assessment] clean_f1: 0.9283841981503752
    [Assessment] adversarial_f1: 0.0
    [Assessment] clean_confidence: 0.9938931465148926
    [Assessment] adversarial_confidence: 0.7903673052787781
    [Assessment] timestamp: 2026-08-27 16:27:35
    [Scoring] attack_name: pgd
    [Scoring] vulnerability_score: 49.09
    [Scoring] risk_level: MEDIUM
    [Scoring] sub_scores: {'asr_score': None, 'degradation_score': 70.12, 'stealth_score': 0.0}
    [Scoring] timestamp: 2026-08-27 16:27:35
    [Scoring] metadata: {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}

  ▶ Vector: deepfool
    [Assessment] attack_name: deepfool
    [Assessment] dataset_name: bazyl/GTSRB
    [Assessment] num_samples: 5000
    [Assessment] attack_success_rate: None
    [Assessment] perturbation: {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}
    [Assessment] accuracy_drop: 0.4093
    [Assessment] f1_drop: 0.4808643568805339
    [Assessment] confidence_drop: 0.5501995086669922
    [Assessment] model_degradation: 0.48012128851584207
    [Assessment] clean_accuracy: 0.9718
    [Assessment] adversarial_accuracy: 0.5625
    [Assessment] clean_f1: 0.9283841981503752
    [Assessment] adversarial_f1: 0.4475198412698413
    [Assessment] clean_confidence: 0.9938931465148926
    [Assessment] adversarial_confidence: 0.4436936378479004
    [Assessment] timestamp: 2026-08-27 16:27:35
    [Scoring] attack_name: deepfool
    [Scoring] vulnerability_score: 33.61
    [Scoring] risk_level: MEDIUM
    [Scoring] sub_scores: {'asr_score': None, 'degradation_score': 48.01, 'stealth_score': 0.0}
    [Scoring] timestamp: 2026-08-27 16:27:35
    [Scoring] metadata: {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}

7. VULNERABILITY SCORE & RISK LEVEL
------------------------------------------------------------------------
  Overall Vulnerability Score : 33.61
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
    - adversarial_confidence: 0.4918379783630371
    - prediction_changed: True
    - true_label: tensor([16,  1, 38, 33, 11, 38, 18, 12, 25, 35, 12,  7, 23,  7,  4,  9, 21, 20,
        27, 38,  4, 33,  9,  3,  1, 11, 13, 10,  9, 11,  5, 17])
    - attack_caused_failure: False
    - attribution: {'technique': 'shap', 'clean': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}, 'adversarial': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}}
    - comparison: {'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [16, 1, 38, 34, 13, 38, 18, 13, 25, 35, 38, 16, 13, 15, 38, 13, 21, 13, 11, 13, 13, 34, 15, 13, 2, 11, 13, 13, 35, 11, 5, 8], 'clean_confidence': 0.998881459236145, 'adversarial_confidence': 0.4918379783630371, 'prediction_changed': True, 'confidence_difference': 0.5070434808731079, 'attribution_comparison_status': 'unavailable', 'attribution_l1': None, 'attribution_l2': None, 'attribution_cosine_similarity': None, 'attribution_mean_difference': None}
    - failure_analysis: {'clean_correct': False, 'adversarial_correct': False, 'prediction_changed': True, 'attack_caused_failure': False, 'true_label': [16, 1, 38, 33, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [16, 1, 38, 34, 13, 38, 18, 13, 25, 35, 38, 16, 13, 15, 38, 13, 21, 13, 11, 13, 13, 34, 15, 13, 2, 11, 13, 13, 35, 11, 5, 8], 'failure_mode': 'clean_incorrect_to_adversarial_incorrect'}
    - metadata: {'attack_name': 'fgsm', 'technique': 'shap', 'assessment_result': {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5968, 'f1_drop': 0.6315876613538385, 'confidence_drop': 0.5018666982650757, 'model_degradation': 0.5767514532063047, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.4920264482498169, 'timestamp': '2026-08-27 16:27:35', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.4103410243988037}}}

  ▶ Technique: pgd_shap
    - attack_name: pgd
    - technique: shap
    - clean_prediction: [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17]
    - adversarial_prediction: [10, 5, 12, 13, 13, 12, 26, 13, 18, 41, 1, 16, 13, 15, 38, 31, 19, 32, 25, 11, 13, 9, 32, 12, 5, 25, 12, 13, 12, 42, 6, 13]
    - clean_confidence: 0.998881459236145
    - adversarial_confidence: 0.7902059555053711
    - prediction_changed: True
    - true_label: tensor([16,  1, 38, 33, 11, 38, 18, 12, 25, 35, 12,  7, 23,  7,  4,  9, 21, 20,
        27, 38,  4, 33,  9,  3,  1, 11, 13, 10,  9, 11,  5, 17])
    - attack_caused_failure: False
    - attribution: {'technique': 'shap', 'clean': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}, 'adversarial': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}}
    - comparison: {'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [10, 5, 12, 13, 13, 12, 26, 13, 18, 41, 1, 16, 13, 15, 38, 31, 19, 32, 25, 11, 13, 9, 32, 12, 5, 25, 12, 13, 12, 42, 6, 13], 'clean_confidence': 0.998881459236145, 'adversarial_confidence': 0.7902059555053711, 'prediction_changed': True, 'confidence_difference': 0.20867550373077393, 'attribution_comparison_status': 'unavailable', 'attribution_l1': None, 'attribution_l2': None, 'attribution_cosine_similarity': None, 'attribution_mean_difference': None}
    - failure_analysis: {'clean_correct': False, 'adversarial_correct': False, 'prediction_changed': True, 'attack_caused_failure': False, 'true_label': [16, 1, 38, 33, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [10, 5, 12, 13, 13, 12, 26, 13, 18, 41, 1, 16, 13, 15, 38, 31, 19, 32, 25, 11, 13, 9, 32, 12, 5, 25, 12, 13, 12, 42, 6, 13], 'failure_mode': 'clean_incorrect_to_adversarial_incorrect'}
    - metadata: {'attack_name': 'pgd', 'technique': 'shap', 'assessment_result': {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.46013909524748, 'l0_mean': 0.9932593520806761, 'is_estimated': False}, 'accuracy_drop': 0.9718, 'f1_drop': 0.9283841981503752, 'confidence_drop': 0.2035258412361145, 'model_degradation': 0.7012366797954965, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.0, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.7903673052787781, 'timestamp': '2026-08-27 16:27:35', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1138906478881836}}}

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
    - metadata: {'attack_name': 'deepfool', 'technique': 'shap', 'assessment_result': {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4093, 'f1_drop': 0.4808643568805339, 'confidence_drop': 0.5501995086669922, 'model_degradation': 0.48012128851584207, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 16:27:35', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.137073278427124}}}

10. HARDENING
------------------------------------------------------------------------
  - metadata: {'defense_name': 'spatial_smoothing', 'defense_type': 'preprocessing', 'parameters': {'kernel_size': 3, 'sigma': 1.0}, 'execution_time_seconds': 0.02914595603942871, 'timestamp': '2026-08-27 16:27:36', 'extra_metadata': {}}
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
  - num_samples: 5000
  - before_baseline_evaluation: {'dataset_name': 'bazyl/GTSRB', 'model_name': 'GTSRB_ViT_Demo', 'num_samples': 5000, 'num_classes': 43, 'accuracy': 0.9718, 'precision_macro': 0.9393342748750693, 'recall_macro': 0.9309797854616169, 'f1_macro': 0.9283841981503752, 'precision_weighted': 0.9745135307244851, 'recall_weighted': 0.9718, 'f1_weighted': 0.970824743912144, 'average_confidence': 0.9938931465148926, 'average_entropy': 0.03971618786454201, 'per_class_metrics': {'0': {'precision': 0.95, 'recall': 1.0, 'f1': 0.9743589743589743, 'support': 38}, '1': {'precision': 1.0, 'recall': 0.9933554817275747, 'f1': 0.9966666666666667, 'support': 301}, '2': {'precision': 0.9606557377049181, 'recall': 1.0, 'f1': 0.979933110367893, 'support': 293}, '3': {'precision': 0.993103448275862, 'recall': 0.96, 'f1': 0.976271186440678, 'support': 150}, '4': {'precision': 1.0, 'recall': 0.9921875, 'f1': 0.996078431372549, 'support': 256}, '5': {'precision': 0.9762845849802372, 'recall': 0.988, 'f1': 0.9821073558648111, 'support': 250}, '6': {'precision': 1.0, 'recall': 0.9074074074074074, 'f1': 0.9514563106796117, 'support': 54}, '7': {'precision': 0.9947916666666666, 'recall': 1.0, 'f1': 0.9973890339425587, 'support': 191}, '8': {'precision': 1.0, 'recall': 0.9476744186046512, 'f1': 0.9731343283582089, 'support': 172}, '9': {'precision': 0.9945652173913043, 'recall': 1.0, 'f1': 0.997275204359673, 'support': 183}, '10': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 274}, '11': {'precision': 1.0, 'recall': 0.9757575757575757, 'f1': 0.9877300613496932, 'support': 165}, '12': {'precision': 0.9965635738831615, 'recall': 1.0, 'f1': 0.9982788296041308, 'support': 290}, '13': {'precision': 0.9964539007092199, 'recall': 1.0, 'f1': 0.9982238010657194, 'support': 281}, '14': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 112}, '15': {'precision': 1.0, 'recall': 0.9880952380952381, 'f1': 0.9940119760479041, 'support': 84}, '16': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 62}, '17': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 158}, '18': {'precision': 0.9924242424242424, 'recall': 0.9225352112676056, 'f1': 0.9562043795620438, 'support': 142}, '19': {'precision': 0.8, 'recall': 0.32, 'f1': 0.45714285714285713, 'support': 25}, '20': {'precision': 0.6470588235294118, 'recall': 0.9428571428571428, 'f1': 0.7674418604651163, 'support': 35}, '21': {'precision': 0.7317073170731707, 'recall': 1.0, 'f1': 0.8450704225352113, 'support': 30}, '22': {'precision': 1.0, 'recall': 0.9761904761904762, 'f1': 0.9879518072289156, 'support': 42}, '23': {'precision': 0.9821428571428571, 'recall': 1.0, 'f1': 0.990990990990991, 'support': 55}, '24': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 28}, '25': {'precision': 0.9900990099009901, 'recall': 0.9900990099009901, 'f1': 0.9900990099009901, 'support': 202}, '26': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 60}, '27': {'precision': 0.9583333333333334, 'recall': 0.92, 'f1': 0.9387755102040817, 'support': 25}, '28': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 63}, '29': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 41}, '30': {'precision': 0.9411764705882353, 'recall': 1.0, 'f1': 0.9696969696969697, 'support': 64}, '31': {'precision': 1.0, 'recall': 0.9906542056074766, 'f1': 0.9953051643192489, 'support': 107}, '32': {'precision': 0.9130434782608695, 'recall': 1.0, 'f1': 0.9545454545454546, 'support': 21}, '33': {'precision': 0.9444444444444444, 'recall': 0.6538461538461539, 'f1': 0.7727272727272727, 'support': 78}, '34': {'precision': 0.654320987654321, 'recall': 0.9636363636363636, 'f1': 0.7794117647058824, 'support': 55}, '35': {'precision': 1.0, 'recall': 0.9935064935064936, 'f1': 0.996742671009772, 'support': 154}, '36': {'precision': 0.782608695652174, 'recall': 0.8372093023255814, 'f1': 0.8089887640449438, 'support': 43}, '37': {'precision': 0.5, 'recall': 0.3888888888888889, 'f1': 0.4375, 'support': 18}, '38': {'precision': 0.9314079422382672, 'recall': 0.9961389961389961, 'f1': 0.9626865671641791, 'support': 259}, '39': {'precision': 0.9545454545454546, 'recall': 0.525, 'f1': 0.6774193548387096, 'support': 40}, '40': {'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 40}, '41': {'precision': 0.896551724137931, 'recall': 1.0, 'f1': 0.9454545454545454, 'support': 26}, '42': {'precision': 0.9090909090909091, 'recall': 0.9090909090909091, 'f1': 0.9090909090909091, 'support': 33}}, 'confusion_matrix': [[38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 299, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 293, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 5, 144, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 247, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 4, 1, 0, 4, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 183, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 274, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 290, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 281, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 158, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 200, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 27, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 53, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 153, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 10, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 258, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 21, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 30]], 'batch_size': 32, 'device': 'cuda', 'timestamp': '2026-08-27 16:27:09', 'extra_metadata': {}}
  - after_baseline_evaluation: {'dataset_name': 'bazyl/GTSRB', 'model_name': 'GTSRB_ViT_Demo', 'num_samples': 5000, 'num_classes': 43, 'accuracy': 0.9718, 'precision_macro': 0.9393342748750693, 'recall_macro': 0.9309797854616169, 'f1_macro': 0.9283841981503752, 'precision_weighted': 0.9745135307244851, 'recall_weighted': 0.9718, 'f1_weighted': 0.970824743912144, 'average_confidence': 0.9938931465148926, 'average_entropy': 0.03971618786454201, 'per_class_metrics': {'0': {'precision': 0.95, 'recall': 1.0, 'f1': 0.9743589743589743, 'support': 38}, '1': {'precision': 1.0, 'recall': 0.9933554817275747, 'f1': 0.9966666666666667, 'support': 301}, '2': {'precision': 0.9606557377049181, 'recall': 1.0, 'f1': 0.979933110367893, 'support': 293}, '3': {'precision': 0.993103448275862, 'recall': 0.96, 'f1': 0.976271186440678, 'support': 150}, '4': {'precision': 1.0, 'recall': 0.9921875, 'f1': 0.996078431372549, 'support': 256}, '5': {'precision': 0.9762845849802372, 'recall': 0.988, 'f1': 0.9821073558648111, 'support': 250}, '6': {'precision': 1.0, 'recall': 0.9074074074074074, 'f1': 0.9514563106796117, 'support': 54}, '7': {'precision': 0.9947916666666666, 'recall': 1.0, 'f1': 0.9973890339425587, 'support': 191}, '8': {'precision': 1.0, 'recall': 0.9476744186046512, 'f1': 0.9731343283582089, 'support': 172}, '9': {'precision': 0.9945652173913043, 'recall': 1.0, 'f1': 0.997275204359673, 'support': 183}, '10': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 274}, '11': {'precision': 1.0, 'recall': 0.9757575757575757, 'f1': 0.9877300613496932, 'support': 165}, '12': {'precision': 0.9965635738831615, 'recall': 1.0, 'f1': 0.9982788296041308, 'support': 290}, '13': {'precision': 0.9964539007092199, 'recall': 1.0, 'f1': 0.9982238010657194, 'support': 281}, '14': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 112}, '15': {'precision': 1.0, 'recall': 0.9880952380952381, 'f1': 0.9940119760479041, 'support': 84}, '16': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 62}, '17': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 158}, '18': {'precision': 0.9924242424242424, 'recall': 0.9225352112676056, 'f1': 0.9562043795620438, 'support': 142}, '19': {'precision': 0.8, 'recall': 0.32, 'f1': 0.45714285714285713, 'support': 25}, '20': {'precision': 0.6470588235294118, 'recall': 0.9428571428571428, 'f1': 0.7674418604651163, 'support': 35}, '21': {'precision': 0.7317073170731707, 'recall': 1.0, 'f1': 0.8450704225352113, 'support': 30}, '22': {'precision': 1.0, 'recall': 0.9761904761904762, 'f1': 0.9879518072289156, 'support': 42}, '23': {'precision': 0.9821428571428571, 'recall': 1.0, 'f1': 0.990990990990991, 'support': 55}, '24': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 28}, '25': {'precision': 0.9900990099009901, 'recall': 0.9900990099009901, 'f1': 0.9900990099009901, 'support': 202}, '26': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 60}, '27': {'precision': 0.9583333333333334, 'recall': 0.92, 'f1': 0.9387755102040817, 'support': 25}, '28': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 63}, '29': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 41}, '30': {'precision': 0.9411764705882353, 'recall': 1.0, 'f1': 0.9696969696969697, 'support': 64}, '31': {'precision': 1.0, 'recall': 0.9906542056074766, 'f1': 0.9953051643192489, 'support': 107}, '32': {'precision': 0.9130434782608695, 'recall': 1.0, 'f1': 0.9545454545454546, 'support': 21}, '33': {'precision': 0.9444444444444444, 'recall': 0.6538461538461539, 'f1': 0.7727272727272727, 'support': 78}, '34': {'precision': 0.654320987654321, 'recall': 0.9636363636363636, 'f1': 0.7794117647058824, 'support': 55}, '35': {'precision': 1.0, 'recall': 0.9935064935064936, 'f1': 0.996742671009772, 'support': 154}, '36': {'precision': 0.782608695652174, 'recall': 0.8372093023255814, 'f1': 0.8089887640449438, 'support': 43}, '37': {'precision': 0.5, 'recall': 0.3888888888888889, 'f1': 0.4375, 'support': 18}, '38': {'precision': 0.9314079422382672, 'recall': 0.9961389961389961, 'f1': 0.9626865671641791, 'support': 259}, '39': {'precision': 0.9545454545454546, 'recall': 0.525, 'f1': 0.6774193548387096, 'support': 40}, '40': {'precision': 1.0, 'recall': 0.95, 'f1': 0.9743589743589743, 'support': 40}, '41': {'precision': 0.896551724137931, 'recall': 1.0, 'f1': 0.9454545454545454, 'support': 26}, '42': {'precision': 0.9090909090909091, 'recall': 0.9090909090909091, 'f1': 0.9090909090909091, 'support': 33}}, 'confusion_matrix': [[38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 299, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 293, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 5, 144, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 247, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 4, 1, 0, 4, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 183, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 274, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 290, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 281, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 158, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 200, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 27, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 53, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 153, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 10, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 258, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 21, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 30]], 'batch_size': 32, 'device': 'cuda', 'timestamp': '2026-08-27 16:27:57', 'extra_metadata': {}}
  - before_attack_results: {'fgsm': {'attack_name': 'fgsm', 'attack_class': 'FGSM', 'execution_time_seconds': 0.4103410243988037, 'parameters': {}}, 'pgd': {'attack_name': 'pgd', 'attack_class': 'PGD', 'execution_time_seconds': 2.1138906478881836, 'parameters': {}}, 'deepfool': {'attack_name': 'deepfool', 'attack_class': 'DeepFool', 'execution_time_seconds': 22.137073278427124, 'parameters': {}}}
  - after_attack_results: {'fgsm': {'attack_name': 'fgsm', 'attack_class': 'FGSM', 'execution_time_seconds': 0.021466732025146484, 'parameters': {}}, 'pgd': {'attack_name': 'pgd', 'attack_class': 'PGD', 'execution_time_seconds': 2.1287896633148193, 'parameters': {}}, 'deepfool': {'attack_name': 'deepfool', 'attack_class': 'DeepFool', 'execution_time_seconds': 22.301685094833374, 'parameters': {}}}
  - before_vulnerability_analysis: {'fgsm': {'assessment': {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5968, 'f1_drop': 0.6315876613538385, 'confidence_drop': 0.5018666982650757, 'model_degradation': 0.5767514532063047, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.4920264482498169, 'timestamp': '2026-08-27 16:27:35', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.4103410243988037}}, 'scoring': {'attack_name': 'fgsm', 'vulnerability_score': 40.37, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.68, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:27:35', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}}}, 'pgd': {'assessment': {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.46013909524748, 'l0_mean': 0.9932593520806761, 'is_estimated': False}, 'accuracy_drop': 0.9718, 'f1_drop': 0.9283841981503752, 'confidence_drop': 0.2035258412361145, 'model_degradation': 0.7012366797954965, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.0, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.7903673052787781, 'timestamp': '2026-08-27 16:27:35', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1138906478881836}}, 'scoring': {'attack_name': 'pgd', 'vulnerability_score': 49.09, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 70.12, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:27:35', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}}}, 'deepfool': {'assessment': {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4093, 'f1_drop': 0.4808643568805339, 'confidence_drop': 0.5501995086669922, 'model_degradation': 0.48012128851584207, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 16:27:35', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.137073278427124}}, 'scoring': {'attack_name': 'deepfool', 'vulnerability_score': 33.61, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.01, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:27:35', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}}}}
  - after_vulnerability_analysis: {'fgsm': {'assessment': {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568545, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5968, 'f1_drop': 0.6315876613538385, 'confidence_drop': 0.5018690824508667, 'model_degradation': 0.5767522479349018, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.4920240640640259, 'timestamp': '2026-08-27 16:28:23', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.021466732025146484}}, 'scoring': {'attack_name': 'fgsm', 'vulnerability_score': 40.37, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.68, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:28:23', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}}}, 'pgd': {'assessment': {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.4613501135192, 'l0_mean': 0.9932861328125, 'is_estimated': False}, 'accuracy_drop': 0.9718, 'f1_drop': 0.9283841981503752, 'confidence_drop': 0.17878562211990356, 'model_degradation': 0.6929899400900928, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.0, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.815107524394989, 'timestamp': '2026-08-27 16:28:23', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1287896633148193}}, 'scoring': {'attack_name': 'pgd', 'vulnerability_score': 48.51, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 69.3, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:28:23', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}}}, 'deepfool': {'assessment': {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4093, 'f1_drop': 0.4808643568805339, 'confidence_drop': 0.5501995086669922, 'model_degradation': 0.48012128851584207, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 16:28:23', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.301685094833374}}, 'scoring': {'attack_name': 'deepfool', 'vulnerability_score': 33.61, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.01, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:28:23', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}}}}
  - overall_improved: False
  - timestamp: 2026-08-27 16:27:36
  - execution_time_seconds: 47.0743
  - extra_metadata: {}

12. BEFORE VS AFTER COMPARISON
------------------------------------------------------------------------
  ▶ Vector: deepfool
    - attack_name: deepfool
    - before_assessment: {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4093, 'f1_drop': 0.4808643568805339, 'confidence_drop': 0.5501995086669922, 'model_degradation': 0.48012128851584207, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 16:27:35', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.137073278427124}}
    - after_assessment: {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4093, 'f1_drop': 0.4808643568805339, 'confidence_drop': 0.5501995086669922, 'model_degradation': 0.48012128851584207, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 16:28:23', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.301685094833374}}
    - before_scoring: {'attack_name': 'deepfool', 'vulnerability_score': 33.61, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.01, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:27:35', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}}
    - after_scoring: {'attack_name': 'deepfool', 'vulnerability_score': 33.61, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.01, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:28:23', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}}
    - delta_attack_success_rate: None
    - delta_accuracy_drop: 0.0
    - delta_f1_drop: 0.0
    - delta_confidence_drop: 0.0
    - delta_model_degradation: 0.0
    - delta_vulnerability_score: 0.0
    - delta_clean_accuracy: 0.0
    - delta_adversarial_accuracy: 0.0
    - delta_perturbation: {'is_estimated': 0.0, 'l0_mean': 0.0, 'l2_mean': 0.0, 'linf_mean': 0.0}
    - before_risk_level: MEDIUM
    - after_risk_level: MEDIUM
    - risk_level_changed: False
    - is_improved: False
    - summary_notes: ['Vulnerability score remained unchanged.']
    - timestamp: 2026-08-27 16:28:23
    - extra_metadata: {}

  ▶ Vector: fgsm
    - attack_name: fgsm
    - before_assessment: {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5968, 'f1_drop': 0.6315876613538385, 'confidence_drop': 0.5018666982650757, 'model_degradation': 0.5767514532063047, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.4920264482498169, 'timestamp': '2026-08-27 16:27:35', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.4103410243988037}}
    - after_assessment: {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568545, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5968, 'f1_drop': 0.6315876613538385, 'confidence_drop': 0.5018690824508667, 'model_degradation': 0.5767522479349018, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.4920240640640259, 'timestamp': '2026-08-27 16:28:23', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.021466732025146484}}
    - before_scoring: {'attack_name': 'fgsm', 'vulnerability_score': 40.37, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.68, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:27:35', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}}
    - after_scoring: {'attack_name': 'fgsm', 'vulnerability_score': 40.37, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.68, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:28:23', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}}
    - delta_attack_success_rate: None
    - delta_accuracy_drop: 0.0
    - delta_f1_drop: 0.0
    - delta_confidence_drop: 0.0
    - delta_model_degradation: 0.0
    - delta_vulnerability_score: 0.0
    - delta_clean_accuracy: 0.0
    - delta_adversarial_accuracy: 0.0
    - delta_perturbation: {'is_estimated': 0.0, 'l0_mean': 0.0, 'l2_mean': -0.0, 'linf_mean': 0.0}
    - before_risk_level: MEDIUM
    - after_risk_level: MEDIUM
    - risk_level_changed: False
    - is_improved: False
    - summary_notes: ['Vulnerability score remained unchanged.']
    - timestamp: 2026-08-27 16:28:23
    - extra_metadata: {}

  ▶ Vector: pgd
    - attack_name: pgd
    - before_assessment: {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.46013909524748, 'l0_mean': 0.9932593520806761, 'is_estimated': False}, 'accuracy_drop': 0.9718, 'f1_drop': 0.9283841981503752, 'confidence_drop': 0.2035258412361145, 'model_degradation': 0.7012366797954965, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.0, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.7903673052787781, 'timestamp': '2026-08-27 16:27:35', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1138906478881836}}
    - after_assessment: {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.4613501135192, 'l0_mean': 0.9932861328125, 'is_estimated': False}, 'accuracy_drop': 0.9718, 'f1_drop': 0.9283841981503752, 'confidence_drop': 0.17878562211990356, 'model_degradation': 0.6929899400900928, 'clean_accuracy': 0.9718, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9283841981503752, 'adversarial_f1': 0.0, 'clean_confidence': 0.9938931465148926, 'adversarial_confidence': 0.815107524394989, 'timestamp': '2026-08-27 16:28:23', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1287896633148193}}
    - before_scoring: {'attack_name': 'pgd', 'vulnerability_score': 49.09, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 70.12, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:27:35', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}}
    - after_scoring: {'attack_name': 'pgd', 'vulnerability_score': 48.51, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 69.3, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 16:28:23', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 5000}}
    - delta_attack_success_rate: None
    - delta_accuracy_drop: 0.0
    - delta_f1_drop: 0.0
    - delta_confidence_drop: -0.0247
    - delta_model_degradation: -0.0082
    - delta_vulnerability_score: -0.58
    - delta_clean_accuracy: 0.0
    - delta_adversarial_accuracy: 0.0
    - delta_perturbation: {'is_estimated': 0.0, 'l0_mean': 0.0, 'l2_mean': 0.0012, 'linf_mean': 0.0}
    - before_risk_level: MEDIUM
    - after_risk_level: MEDIUM
    - risk_level_changed: False
    - is_improved: True
    - summary_notes: ['Vulnerability score decreased by 0.58 points.']
    - timestamp: 2026-08-27 16:28:23
    - extra_metadata: {}

13. EXECUTION PERFORMANCE
------------------------------------------------------------------------
  Run Label    : AdverScan [full]
  Started At   : 2026-08-27 16:26:48
  Total Time   : 0.00s
  Overall      : UNKNOWN

  MODULE                         STATUS          TIME
  ········································································

14. RECOMMENDATIONS
------------------------------------------------------------------------
  [01] [MEDIUM] Vulnerability score 33.61 warrants attention. Implement input sanitization and monitor inference traffic for anomalies.
  [02] XAI attribution maps are available. Review highlighted input regions disproportionately targeted by adversarial perturbations to guide robustness patches.
  [03] Adversarial defense was applied. Validate post-hardening accuracy retention and conduct periodic re-tests to ensure defense durability.
  [04] Re-test indicates persistent vulnerability for vector 'deepfool'. Increase adversarial training epochs or broaden epsilon schedules.
  [05] Re-test indicates persistent vulnerability for vector 'fgsm'. Increase adversarial training epochs or broaden epsilon schedules.

15. FINAL SECURITY SUMMARY
------------------------------------------------------------------------
  - risk_level: MEDIUM
  - vulnerability_score: 33.61
  - baseline_accuracy: 97.18%
  - mean_adversarial_accuracy: 46.88%
  - attacks_evaluated: ['fgsm', 'pgd', 'deepfool']
  - hardening_applied: True
  - retest_conducted: True
  - total_recommendations: 5
  - primary_recommendation: [MEDIUM] Vulnerability score 33.61 warrants attention. Implement input sanitization and monitor inference traffic for anomalies.

========================================================================
  Generated by AdverScan — 2026-08-27 16:28:23
========================================================================