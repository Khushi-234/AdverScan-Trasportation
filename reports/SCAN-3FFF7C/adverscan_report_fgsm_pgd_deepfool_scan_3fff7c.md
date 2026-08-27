========================================================================
          ADVERSCAN SECURITY ASSESSMENT REPORT           
========================================================================
  Report ID   : RPT-5BBE2015
  Scan ID     : SCAN-3FFF7C
  Timestamp   : 2026-08-27 17:04:38
  Risk Level  : MEDIUM
  Vuln. Score : 33.71
========================================================================

1. EXECUTIVE SUMMARY
------------------------------------------------------------------------
  Scan ID            : SCAN-3FFF7C
  Risk Level         : MEDIUM
  Vulnerability Score: 33.71
  Baseline Accuracy  : 97.17%
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
  - num_samples: 10000
  - num_classes: 43
  - accuracy: 97.17%
  - precision_macro: 94.15%
  - recall_macro: 93.50%
  - f1_macro: 93.29%
  - precision_weighted: 97.42%
  - recall_weighted: 97.17%
  - f1_weighted: 97.11%
  - average_confidence: 99.37%
  - average_entropy: 3.99%
  - per_class_metrics: {'0': {'precision': 0.9423076923076923, 'recall': 1.0, 'f1': 0.9702970297029703, 'support': 49}, '1': {'precision': 1.0, 'recall': 0.9930915371329879, 'f1': 0.9965337954939342, 'support': 579}, '2': {'precision': 0.9688524590163935, 'recall': 1.0, 'f1': 0.984179850124896, 'support': 591}, '3': {'precision': 0.9969135802469136, 'recall': 0.9758308157099698, 'f1': 0.9862595419847329, 'support': 331}, '4': {'precision': 0.998062015503876, 'recall': 0.9903846153846154, 'f1': 0.9942084942084942, 'support': 520}, '5': {'precision': 0.9805068226120858, 'recall': 0.9921104536489151, 'f1': 0.9862745098039216, 'support': 507}, '6': {'precision': 1.0, 'recall': 0.9016393442622951, 'f1': 0.9482758620689655, 'support': 122}, '7': {'precision': 0.9972067039106145, 'recall': 0.9972067039106145, 'f1': 0.9972067039106145, 'support': 358}, '8': {'precision': 0.9880239520958084, 'recall': 0.9510086455331412, 'f1': 0.9691629955947136, 'support': 347}, '9': {'precision': 0.9972677595628415, 'recall': 1.0, 'f1': 0.9986320109439124, 'support': 365}, '10': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 527}, '11': {'precision': 0.9907120743034056, 'recall': 0.9726443768996961, 'f1': 0.9815950920245399, 'support': 329}, '12': {'precision': 0.9982456140350877, 'recall': 1.0, 'f1': 0.9991220368744512, 'support': 569}, '13': {'precision': 0.9982758620689656, 'recall': 1.0, 'f1': 0.999137187230371, 'support': 579}, '14': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 215}, '15': {'precision': 1.0, 'recall': 0.9940119760479041, 'f1': 0.996996996996997, 'support': 167}, '16': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 116}, '17': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 286}, '18': {'precision': 0.99644128113879, 'recall': 0.9240924092409241, 'f1': 0.958904109589041, 'support': 303}, '19': {'precision': 0.8571428571428571, 'recall': 0.5, 'f1': 0.631578947368421, 'support': 48}, '20': {'precision': 0.7078651685393258, 'recall': 0.9402985074626866, 'f1': 0.8076923076923077, 'support': 67}, '21': {'precision': 0.7411764705882353, 'recall': 1.0, 'f1': 0.8513513513513513, 'support': 63}, '22': {'precision': 1.0, 'recall': 0.8705882352941177, 'f1': 0.9308176100628931, 'support': 85}, '23': {'precision': 0.983739837398374, 'recall': 0.9918032786885246, 'f1': 0.9877551020408163, 'support': 122}, '24': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 64}, '25': {'precision': 0.968421052631579, 'recall': 0.983957219251337, 'f1': 0.9761273209549072, 'support': 374}, '26': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 136}, '27': {'precision': 0.9423076923076923, 'recall': 0.9245283018867925, 'f1': 0.9333333333333333, 'support': 53}, '28': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 125}, '29': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 78}, '30': {'precision': 0.9307692307692308, 'recall': 1.0, 'f1': 0.9641434262948207, 'support': 121}, '31': {'precision': 0.9907834101382489, 'recall': 0.9907834101382489, 'f1': 0.9907834101382489, 'support': 217}, '32': {'precision': 0.8727272727272727, 'recall': 1.0, 'f1': 0.9320388349514563, 'support': 48}, '33': {'precision': 0.9487179487179487, 'recall': 0.6894409937888198, 'f1': 0.7985611510791367, 'support': 161}, '34': {'precision': 0.6601307189542484, 'recall': 0.9528301886792453, 'f1': 0.7799227799227799, 'support': 106}, '35': {'precision': 0.9968051118210862, 'recall': 0.9936305732484076, 'f1': 0.9952153110047847, 'support': 314}, '36': {'precision': 0.7692307692307693, 'recall': 0.7777777777777778, 'f1': 0.7734806629834254, 'support': 90}, '37': {'precision': 0.5652173913043478, 'recall': 0.5416666666666666, 'f1': 0.5531914893617021, 'support': 48}, '38': {'precision': 0.937394247038917, 'recall': 0.9946140035906643, 'f1': 0.9651567944250871, 'support': 557}, '39': {'precision': 0.9428571428571428, 'recall': 0.4714285714285714, 'f1': 0.6285714285714286, 'support': 70}, '40': {'precision': 0.9863013698630136, 'recall': 0.96, 'f1': 0.972972972972973, 'support': 75}, '41': {'precision': 0.8846153846153846, 'recall': 1.0, 'f1': 0.9387755102040817, 'support': 46}, '42': {'precision': 0.9436619718309859, 'recall': 0.9305555555555556, 'f1': 0.9370629370629371, 'support': 72}}
  - confusion_matrix: [[49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 575, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 591, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 323, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 515, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 503, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4], [0, 0, 0, 0, 0, 0, 0, 357, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 1, 1, 7, 0, 0, 330, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 365, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 527, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 320, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 579, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 215, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 286, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 280, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 2, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 368, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 215, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 111, 50, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 101, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 312, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 20, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 21, 26, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 554, 2, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 33, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 72, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 67]]
  - batch_size: 32
  - device: cuda
  - timestamp: 2026-08-27 17:03:02
  - extra_metadata: {}

5. ADVERSARIAL ATTACK RESULTS
------------------------------------------------------------------------
  ▶ FGSM
    execution_time_seconds   : 41.29%

  ▶ PGD
    execution_time_seconds   : 2.1367502212524414

  ▶ DEEPFOOL
    execution_time_seconds   : 22.38034749031067

6. VULNERABILITY ASSESSMENT
------------------------------------------------------------------------
  ▶ Vector: fgsm
    [Assessment] attack_name: fgsm
    [Assessment] dataset_name: bazyl/GTSRB
    [Assessment] num_samples: 10000
    [Assessment] attack_success_rate: None
    [Assessment] perturbation: {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568545, 'l0_mean': 0.9860084170386905, 'is_estimated': False}
    [Assessment] accuracy_drop: 0.5967
    [Assessment] f1_drop: 0.6361178561878691
    [Assessment] confidence_drop: 0.501677930355072
    [Assessment] model_degradation: 0.5781652621809804
    [Assessment] clean_accuracy: 0.9717
    [Assessment] adversarial_accuracy: 0.375
    [Assessment] clean_f1: 0.9329143929844058
    [Assessment] adversarial_f1: 0.2967965367965368
    [Assessment] clean_confidence: 0.9937019348144531
    [Assessment] adversarial_confidence: 0.4920240044593811
    [Assessment] timestamp: 2026-08-27 17:03:28
    [Scoring] attack_name: fgsm
    [Scoring] vulnerability_score: 40.47
    [Scoring] risk_level: MEDIUM
    [Scoring] sub_scores: {'asr_score': None, 'degradation_score': 57.82, 'stealth_score': 0.0}
    [Scoring] timestamp: 2026-08-27 17:03:28
    [Scoring] metadata: {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}

  ▶ Vector: pgd
    [Assessment] attack_name: pgd
    [Assessment] dataset_name: bazyl/GTSRB
    [Assessment] num_samples: 10000
    [Assessment] attack_success_rate: None
    [Assessment] perturbation: {'linf_mean': 0.9169117696583271, 'l2_mean': 200.457100555142, 'l0_mean': 0.9934488932291667, 'is_estimated': False}
    [Assessment] accuracy_drop: 0.9717
    [Assessment] f1_drop: 0.9329143929844058
    [Assessment] confidence_drop: 0.1887420415878296
    [Assessment] model_degradation: 0.6977854781907452
    [Assessment] clean_accuracy: 0.9717
    [Assessment] adversarial_accuracy: 0.0
    [Assessment] clean_f1: 0.9329143929844058
    [Assessment] adversarial_f1: 0.0
    [Assessment] clean_confidence: 0.9937019348144531
    [Assessment] adversarial_confidence: 0.8049598932266235
    [Assessment] timestamp: 2026-08-27 17:03:28
    [Scoring] attack_name: pgd
    [Scoring] vulnerability_score: 48.84
    [Scoring] risk_level: MEDIUM
    [Scoring] sub_scores: {'asr_score': None, 'degradation_score': 69.78, 'stealth_score': 0.0}
    [Scoring] timestamp: 2026-08-27 17:03:28
    [Scoring] metadata: {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}

  ▶ Vector: deepfool
    [Assessment] attack_name: deepfool
    [Assessment] dataset_name: bazyl/GTSRB
    [Assessment] num_samples: 10000
    [Assessment] attack_success_rate: None
    [Assessment] perturbation: {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}
    [Assessment] accuracy_drop: 0.4092
    [Assessment] f1_drop: 0.4853945517145645
    [Assessment] confidence_drop: 0.5500082969665527
    [Assessment] model_degradation: 0.48153428289370576
    [Assessment] clean_accuracy: 0.9717
    [Assessment] adversarial_accuracy: 0.5625
    [Assessment] clean_f1: 0.9329143929844058
    [Assessment] adversarial_f1: 0.4475198412698413
    [Assessment] clean_confidence: 0.9937019348144531
    [Assessment] adversarial_confidence: 0.4436936378479004
    [Assessment] timestamp: 2026-08-27 17:03:28
    [Scoring] attack_name: deepfool
    [Scoring] vulnerability_score: 33.71
    [Scoring] risk_level: MEDIUM
    [Scoring] sub_scores: {'asr_score': None, 'degradation_score': 48.15, 'stealth_score': 0.0}
    [Scoring] timestamp: 2026-08-27 17:03:28
    [Scoring] metadata: {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}

7. VULNERABILITY SCORE & RISK LEVEL
------------------------------------------------------------------------
  Overall Vulnerability Score : 33.71
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
    - adversarial_confidence: 0.4918355345726013
    - prediction_changed: True
    - true_label: tensor([16,  1, 38, 33, 11, 38, 18, 12, 25, 35, 12,  7, 23,  7,  4,  9, 21, 20,
        27, 38,  4, 33,  9,  3,  1, 11, 13, 10,  9, 11,  5, 17])
    - attack_caused_failure: False
    - attribution: {'technique': 'shap', 'clean': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}, 'adversarial': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}}
    - comparison: {'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [16, 1, 38, 34, 13, 38, 18, 13, 25, 35, 38, 16, 13, 15, 38, 13, 21, 13, 11, 13, 13, 34, 15, 13, 2, 11, 13, 13, 35, 11, 5, 8], 'clean_confidence': 0.998881459236145, 'adversarial_confidence': 0.4918355345726013, 'prediction_changed': True, 'confidence_difference': 0.5070459246635437, 'attribution_comparison_status': 'unavailable', 'attribution_l1': None, 'attribution_l2': None, 'attribution_cosine_similarity': None, 'attribution_mean_difference': None}
    - failure_analysis: {'clean_correct': False, 'adversarial_correct': False, 'prediction_changed': True, 'attack_caused_failure': False, 'true_label': [16, 1, 38, 33, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [16, 1, 38, 34, 13, 38, 18, 13, 25, 35, 38, 16, 13, 15, 38, 13, 21, 13, 11, 13, 13, 34, 15, 13, 2, 11, 13, 13, 35, 11, 5, 8], 'failure_mode': 'clean_incorrect_to_adversarial_incorrect'}
    - metadata: {'attack_name': 'fgsm', 'technique': 'shap', 'assessment_result': {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568545, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5967, 'f1_drop': 0.6361178561878691, 'confidence_drop': 0.501677930355072, 'model_degradation': 0.5781652621809804, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.4920240044593811, 'timestamp': '2026-08-27 17:03:28', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.41286802291870117}}}

  ▶ Technique: pgd_shap
    - attack_name: pgd
    - technique: shap
    - clean_prediction: [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17]
    - adversarial_prediction: [10, 5, 12, 40, 13, 12, 26, 32, 11, 16, 1, 16, 13, 15, 42, 20, 19, 12, 25, 11, 13, 9, 32, 13, 5, 25, 12, 13, 12, 42, 6, 13]
    - clean_confidence: 0.998881459236145
    - adversarial_confidence: 0.8048253059387207
    - prediction_changed: True
    - true_label: tensor([16,  1, 38, 33, 11, 38, 18, 12, 25, 35, 12,  7, 23,  7,  4,  9, 21, 20,
        27, 38,  4, 33,  9,  3,  1, 11, 13, 10,  9, 11,  5, 17])
    - attack_caused_failure: False
    - attribution: {'technique': 'shap', 'clean': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}, 'adversarial': {'status': 'unavailable', 'executed': False, 'technique': 'shap', 'message': 'SHAP library is not installed in the current environment.', 'attribution': None}}
    - comparison: {'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [10, 5, 12, 40, 13, 12, 26, 32, 11, 16, 1, 16, 13, 15, 42, 20, 19, 12, 25, 11, 13, 9, 32, 13, 5, 25, 12, 13, 12, 42, 6, 13], 'clean_confidence': 0.998881459236145, 'adversarial_confidence': 0.8048253059387207, 'prediction_changed': True, 'confidence_difference': 0.19405615329742432, 'attribution_comparison_status': 'unavailable', 'attribution_l1': None, 'attribution_l2': None, 'attribution_cosine_similarity': None, 'attribution_mean_difference': None}
    - failure_analysis: {'clean_correct': False, 'adversarial_correct': False, 'prediction_changed': True, 'attack_caused_failure': False, 'true_label': [16, 1, 38, 33, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'clean_prediction': [16, 1, 38, 34, 11, 38, 18, 12, 25, 35, 12, 7, 23, 7, 4, 9, 21, 20, 27, 38, 4, 33, 9, 3, 1, 11, 13, 10, 9, 11, 5, 17], 'adversarial_prediction': [10, 5, 12, 40, 13, 12, 26, 32, 11, 16, 1, 16, 13, 15, 42, 20, 19, 12, 25, 11, 13, 9, 32, 13, 5, 25, 12, 13, 12, 42, 6, 13], 'failure_mode': 'clean_incorrect_to_adversarial_incorrect'}
    - metadata: {'attack_name': 'pgd', 'technique': 'shap', 'assessment_result': {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.457100555142, 'l0_mean': 0.9934488932291667, 'is_estimated': False}, 'accuracy_drop': 0.9717, 'f1_drop': 0.9329143929844058, 'confidence_drop': 0.1887420415878296, 'model_degradation': 0.6977854781907452, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.0, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.8049598932266235, 'timestamp': '2026-08-27 17:03:28', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1367502212524414}}}

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
    - metadata: {'attack_name': 'deepfool', 'technique': 'shap', 'assessment_result': {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4092, 'f1_drop': 0.4853945517145645, 'confidence_drop': 0.5500082969665527, 'model_degradation': 0.48153428289370576, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 17:03:28', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.38034749031067}}}

10. HARDENING
------------------------------------------------------------------------
  - metadata: {'defense_name': 'spatial_smoothing', 'defense_type': 'preprocessing', 'parameters': {'kernel_size': 3, 'sigma': 1.0}, 'execution_time_seconds': 0.028142452239990234, 'timestamp': '2026-08-27 17:03:29', 'extra_metadata': {}}
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
  - num_samples: 10000
  - before_baseline_evaluation: {'dataset_name': 'bazyl/GTSRB', 'model_name': 'GTSRB_ViT_Demo', 'num_samples': 10000, 'num_classes': 43, 'accuracy': 0.9717, 'precision_macro': 0.9414576945413753, 'recall_macro': 0.9350214921215926, 'f1_macro': 0.9329143929844058, 'precision_weighted': 0.9742307576879926, 'recall_weighted': 0.9717, 'f1_weighted': 0.9710902576438774, 'average_confidence': 0.9937019348144531, 'average_entropy': 0.03992879018187523, 'per_class_metrics': {'0': {'precision': 0.9423076923076923, 'recall': 1.0, 'f1': 0.9702970297029703, 'support': 49}, '1': {'precision': 1.0, 'recall': 0.9930915371329879, 'f1': 0.9965337954939342, 'support': 579}, '2': {'precision': 0.9688524590163935, 'recall': 1.0, 'f1': 0.984179850124896, 'support': 591}, '3': {'precision': 0.9969135802469136, 'recall': 0.9758308157099698, 'f1': 0.9862595419847329, 'support': 331}, '4': {'precision': 0.998062015503876, 'recall': 0.9903846153846154, 'f1': 0.9942084942084942, 'support': 520}, '5': {'precision': 0.9805068226120858, 'recall': 0.9921104536489151, 'f1': 0.9862745098039216, 'support': 507}, '6': {'precision': 1.0, 'recall': 0.9016393442622951, 'f1': 0.9482758620689655, 'support': 122}, '7': {'precision': 0.9972067039106145, 'recall': 0.9972067039106145, 'f1': 0.9972067039106145, 'support': 358}, '8': {'precision': 0.9880239520958084, 'recall': 0.9510086455331412, 'f1': 0.9691629955947136, 'support': 347}, '9': {'precision': 0.9972677595628415, 'recall': 1.0, 'f1': 0.9986320109439124, 'support': 365}, '10': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 527}, '11': {'precision': 0.9907120743034056, 'recall': 0.9726443768996961, 'f1': 0.9815950920245399, 'support': 329}, '12': {'precision': 0.9982456140350877, 'recall': 1.0, 'f1': 0.9991220368744512, 'support': 569}, '13': {'precision': 0.9982758620689656, 'recall': 1.0, 'f1': 0.999137187230371, 'support': 579}, '14': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 215}, '15': {'precision': 1.0, 'recall': 0.9940119760479041, 'f1': 0.996996996996997, 'support': 167}, '16': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 116}, '17': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 286}, '18': {'precision': 0.99644128113879, 'recall': 0.9240924092409241, 'f1': 0.958904109589041, 'support': 303}, '19': {'precision': 0.8571428571428571, 'recall': 0.5, 'f1': 0.631578947368421, 'support': 48}, '20': {'precision': 0.7078651685393258, 'recall': 0.9402985074626866, 'f1': 0.8076923076923077, 'support': 67}, '21': {'precision': 0.7411764705882353, 'recall': 1.0, 'f1': 0.8513513513513513, 'support': 63}, '22': {'precision': 1.0, 'recall': 0.8705882352941177, 'f1': 0.9308176100628931, 'support': 85}, '23': {'precision': 0.983739837398374, 'recall': 0.9918032786885246, 'f1': 0.9877551020408163, 'support': 122}, '24': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 64}, '25': {'precision': 0.968421052631579, 'recall': 0.983957219251337, 'f1': 0.9761273209549072, 'support': 374}, '26': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 136}, '27': {'precision': 0.9423076923076923, 'recall': 0.9245283018867925, 'f1': 0.9333333333333333, 'support': 53}, '28': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 125}, '29': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 78}, '30': {'precision': 0.9307692307692308, 'recall': 1.0, 'f1': 0.9641434262948207, 'support': 121}, '31': {'precision': 0.9907834101382489, 'recall': 0.9907834101382489, 'f1': 0.9907834101382489, 'support': 217}, '32': {'precision': 0.8727272727272727, 'recall': 1.0, 'f1': 0.9320388349514563, 'support': 48}, '33': {'precision': 0.9487179487179487, 'recall': 0.6894409937888198, 'f1': 0.7985611510791367, 'support': 161}, '34': {'precision': 0.6601307189542484, 'recall': 0.9528301886792453, 'f1': 0.7799227799227799, 'support': 106}, '35': {'precision': 0.9968051118210862, 'recall': 0.9936305732484076, 'f1': 0.9952153110047847, 'support': 314}, '36': {'precision': 0.7692307692307693, 'recall': 0.7777777777777778, 'f1': 0.7734806629834254, 'support': 90}, '37': {'precision': 0.5652173913043478, 'recall': 0.5416666666666666, 'f1': 0.5531914893617021, 'support': 48}, '38': {'precision': 0.937394247038917, 'recall': 0.9946140035906643, 'f1': 0.9651567944250871, 'support': 557}, '39': {'precision': 0.9428571428571428, 'recall': 0.4714285714285714, 'f1': 0.6285714285714286, 'support': 70}, '40': {'precision': 0.9863013698630136, 'recall': 0.96, 'f1': 0.972972972972973, 'support': 75}, '41': {'precision': 0.8846153846153846, 'recall': 1.0, 'f1': 0.9387755102040817, 'support': 46}, '42': {'precision': 0.9436619718309859, 'recall': 0.9305555555555556, 'f1': 0.9370629370629371, 'support': 72}}, 'confusion_matrix': [[49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 575, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 591, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 323, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 515, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 503, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4], [0, 0, 0, 0, 0, 0, 0, 357, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 1, 1, 7, 0, 0, 330, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 365, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 527, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 320, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 579, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 215, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 286, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 280, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 2, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 368, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 215, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 111, 50, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 101, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 312, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 20, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 21, 26, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 554, 2, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 33, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 72, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 67]], 'batch_size': 32, 'device': 'cuda', 'timestamp': '2026-08-27 17:03:02', 'extra_metadata': {}}
  - after_baseline_evaluation: {'dataset_name': 'bazyl/GTSRB', 'model_name': 'GTSRB_ViT_Demo', 'num_samples': 10000, 'num_classes': 43, 'accuracy': 0.9717, 'precision_macro': 0.9414576945413753, 'recall_macro': 0.9350214921215926, 'f1_macro': 0.9329143929844058, 'precision_weighted': 0.9742307576879926, 'recall_weighted': 0.9717, 'f1_weighted': 0.9710902576438774, 'average_confidence': 0.9937019348144531, 'average_entropy': 0.03992879018187523, 'per_class_metrics': {'0': {'precision': 0.9423076923076923, 'recall': 1.0, 'f1': 0.9702970297029703, 'support': 49}, '1': {'precision': 1.0, 'recall': 0.9930915371329879, 'f1': 0.9965337954939342, 'support': 579}, '2': {'precision': 0.9688524590163935, 'recall': 1.0, 'f1': 0.984179850124896, 'support': 591}, '3': {'precision': 0.9969135802469136, 'recall': 0.9758308157099698, 'f1': 0.9862595419847329, 'support': 331}, '4': {'precision': 0.998062015503876, 'recall': 0.9903846153846154, 'f1': 0.9942084942084942, 'support': 520}, '5': {'precision': 0.9805068226120858, 'recall': 0.9921104536489151, 'f1': 0.9862745098039216, 'support': 507}, '6': {'precision': 1.0, 'recall': 0.9016393442622951, 'f1': 0.9482758620689655, 'support': 122}, '7': {'precision': 0.9972067039106145, 'recall': 0.9972067039106145, 'f1': 0.9972067039106145, 'support': 358}, '8': {'precision': 0.9880239520958084, 'recall': 0.9510086455331412, 'f1': 0.9691629955947136, 'support': 347}, '9': {'precision': 0.9972677595628415, 'recall': 1.0, 'f1': 0.9986320109439124, 'support': 365}, '10': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 527}, '11': {'precision': 0.9907120743034056, 'recall': 0.9726443768996961, 'f1': 0.9815950920245399, 'support': 329}, '12': {'precision': 0.9982456140350877, 'recall': 1.0, 'f1': 0.9991220368744512, 'support': 569}, '13': {'precision': 0.9982758620689656, 'recall': 1.0, 'f1': 0.999137187230371, 'support': 579}, '14': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 215}, '15': {'precision': 1.0, 'recall': 0.9940119760479041, 'f1': 0.996996996996997, 'support': 167}, '16': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 116}, '17': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 286}, '18': {'precision': 0.99644128113879, 'recall': 0.9240924092409241, 'f1': 0.958904109589041, 'support': 303}, '19': {'precision': 0.8571428571428571, 'recall': 0.5, 'f1': 0.631578947368421, 'support': 48}, '20': {'precision': 0.7078651685393258, 'recall': 0.9402985074626866, 'f1': 0.8076923076923077, 'support': 67}, '21': {'precision': 0.7411764705882353, 'recall': 1.0, 'f1': 0.8513513513513513, 'support': 63}, '22': {'precision': 1.0, 'recall': 0.8705882352941177, 'f1': 0.9308176100628931, 'support': 85}, '23': {'precision': 0.983739837398374, 'recall': 0.9918032786885246, 'f1': 0.9877551020408163, 'support': 122}, '24': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 64}, '25': {'precision': 0.968421052631579, 'recall': 0.983957219251337, 'f1': 0.9761273209549072, 'support': 374}, '26': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 136}, '27': {'precision': 0.9423076923076923, 'recall': 0.9245283018867925, 'f1': 0.9333333333333333, 'support': 53}, '28': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 125}, '29': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 78}, '30': {'precision': 0.9307692307692308, 'recall': 1.0, 'f1': 0.9641434262948207, 'support': 121}, '31': {'precision': 0.9907834101382489, 'recall': 0.9907834101382489, 'f1': 0.9907834101382489, 'support': 217}, '32': {'precision': 0.8727272727272727, 'recall': 1.0, 'f1': 0.9320388349514563, 'support': 48}, '33': {'precision': 0.9487179487179487, 'recall': 0.6894409937888198, 'f1': 0.7985611510791367, 'support': 161}, '34': {'precision': 0.6601307189542484, 'recall': 0.9528301886792453, 'f1': 0.7799227799227799, 'support': 106}, '35': {'precision': 0.9968051118210862, 'recall': 0.9936305732484076, 'f1': 0.9952153110047847, 'support': 314}, '36': {'precision': 0.7692307692307693, 'recall': 0.7777777777777778, 'f1': 0.7734806629834254, 'support': 90}, '37': {'precision': 0.5652173913043478, 'recall': 0.5416666666666666, 'f1': 0.5531914893617021, 'support': 48}, '38': {'precision': 0.937394247038917, 'recall': 0.9946140035906643, 'f1': 0.9651567944250871, 'support': 557}, '39': {'precision': 0.9428571428571428, 'recall': 0.4714285714285714, 'f1': 0.6285714285714286, 'support': 70}, '40': {'precision': 0.9863013698630136, 'recall': 0.96, 'f1': 0.972972972972973, 'support': 75}, '41': {'precision': 0.8846153846153846, 'recall': 1.0, 'f1': 0.9387755102040817, 'support': 46}, '42': {'precision': 0.9436619718309859, 'recall': 0.9305555555555556, 'f1': 0.9370629370629371, 'support': 72}}, 'confusion_matrix': [[49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 575, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 591, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 323, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 515, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 503, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4], [0, 0, 0, 0, 0, 0, 0, 357, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 1, 1, 7, 0, 0, 330, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 365, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 527, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 320, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 579, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 215, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 286, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 280, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 2, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 368, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 215, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 111, 50, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 101, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 312, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 20, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 21, 26, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 554, 2, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 33, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 72, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 67]], 'batch_size': 32, 'device': 'cuda', 'timestamp': '2026-08-27 17:04:11', 'extra_metadata': {}}
  - before_attack_results: {'fgsm': {'attack_name': 'fgsm', 'attack_class': 'FGSM', 'execution_time_seconds': 0.41286802291870117, 'parameters': {}}, 'pgd': {'attack_name': 'pgd', 'attack_class': 'PGD', 'execution_time_seconds': 2.1367502212524414, 'parameters': {}}, 'deepfool': {'attack_name': 'deepfool', 'attack_class': 'DeepFool', 'execution_time_seconds': 22.38034749031067, 'parameters': {}}}
  - after_attack_results: {'fgsm': {'attack_name': 'fgsm', 'attack_class': 'FGSM', 'execution_time_seconds': 0.02091050148010254, 'parameters': {}}, 'pgd': {'attack_name': 'pgd', 'attack_class': 'PGD', 'execution_time_seconds': 2.1397159099578857, 'parameters': {}}, 'deepfool': {'attack_name': 'deepfool', 'attack_class': 'DeepFool', 'execution_time_seconds': 22.451696157455444, 'parameters': {}}}
  - before_vulnerability_analysis: {'fgsm': {'assessment': {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568545, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5967, 'f1_drop': 0.6361178561878691, 'confidence_drop': 0.501677930355072, 'model_degradation': 0.5781652621809804, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.4920240044593811, 'timestamp': '2026-08-27 17:03:28', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.41286802291870117}}, 'scoring': {'attack_name': 'fgsm', 'vulnerability_score': 40.47, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.82, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:03:28', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}}}, 'pgd': {'assessment': {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.457100555142, 'l0_mean': 0.9934488932291667, 'is_estimated': False}, 'accuracy_drop': 0.9717, 'f1_drop': 0.9329143929844058, 'confidence_drop': 0.1887420415878296, 'model_degradation': 0.6977854781907452, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.0, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.8049598932266235, 'timestamp': '2026-08-27 17:03:28', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1367502212524414}}, 'scoring': {'attack_name': 'pgd', 'vulnerability_score': 48.84, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 69.78, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:03:28', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}}}, 'deepfool': {'assessment': {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4092, 'f1_drop': 0.4853945517145645, 'confidence_drop': 0.5500082969665527, 'model_degradation': 0.48153428289370576, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 17:03:28', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.38034749031067}}, 'scoring': {'attack_name': 'deepfool', 'vulnerability_score': 33.71, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.15, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:03:28', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}}}}
  - after_vulnerability_analysis: {'fgsm': {'assessment': {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5967, 'f1_drop': 0.6361178561878691, 'confidence_drop': 0.501675546169281, 'model_degradation': 0.5781644674523834, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.4920263886451721, 'timestamp': '2026-08-27 17:04:37', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.02091050148010254}}, 'scoring': {'attack_name': 'fgsm', 'vulnerability_score': 40.47, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.82, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:04:37', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}}}, 'pgd': {'assessment': {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.45703681796542, 'l0_mean': 0.9936133144664115, 'is_estimated': False}, 'accuracy_drop': 0.9717, 'f1_drop': 0.9329143929844058, 'confidence_drop': 0.20629745721817017, 'model_degradation': 0.7036372834008587, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.0, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.787404477596283, 'timestamp': '2026-08-27 17:04:37', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1397159099578857}}, 'scoring': {'attack_name': 'pgd', 'vulnerability_score': 49.25, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 70.36, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:04:38', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}}}, 'deepfool': {'assessment': {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4092, 'f1_drop': 0.4853945517145645, 'confidence_drop': 0.5500082969665527, 'model_degradation': 0.48153428289370576, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 17:04:38', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.451696157455444}}, 'scoring': {'attack_name': 'deepfool', 'vulnerability_score': 33.71, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.15, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:04:38', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}}}}
  - overall_improved: False
  - timestamp: 2026-08-27 17:03:29
  - execution_time_seconds: 68.3664
  - extra_metadata: {}

12. BEFORE VS AFTER COMPARISON
------------------------------------------------------------------------
  ▶ Vector: pgd
    - attack_name: pgd
    - before_assessment: {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.457100555142, 'l0_mean': 0.9934488932291667, 'is_estimated': False}, 'accuracy_drop': 0.9717, 'f1_drop': 0.9329143929844058, 'confidence_drop': 0.1887420415878296, 'model_degradation': 0.6977854781907452, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.0, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.8049598932266235, 'timestamp': '2026-08-27 17:03:28', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1367502212524414}}
    - after_assessment: {'attack_name': 'pgd', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 200.45703681796542, 'l0_mean': 0.9936133144664115, 'is_estimated': False}, 'accuracy_drop': 0.9717, 'f1_drop': 0.9329143929844058, 'confidence_drop': 0.20629745721817017, 'model_degradation': 0.7036372834008587, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.0, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.0, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.787404477596283, 'timestamp': '2026-08-27 17:04:37', 'extra_metadata': {'attack_class': 'PGD', 'execution_time_seconds': 2.1397159099578857}}
    - before_scoring: {'attack_name': 'pgd', 'vulnerability_score': 48.84, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 69.78, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:03:28', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}}
    - after_scoring: {'attack_name': 'pgd', 'vulnerability_score': 49.25, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 70.36, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:04:38', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}}
    - delta_attack_success_rate: None
    - delta_accuracy_drop: 0.0
    - delta_f1_drop: 0.0
    - delta_confidence_drop: 0.0176
    - delta_model_degradation: 0.0059
    - delta_vulnerability_score: 0.41
    - delta_clean_accuracy: 0.0
    - delta_adversarial_accuracy: 0.0
    - delta_perturbation: {'l0_mean': 0.0002, 'linf_mean': 0.0, 'l2_mean': -0.0001, 'is_estimated': 0.0}
    - before_risk_level: MEDIUM
    - after_risk_level: MEDIUM
    - risk_level_changed: False
    - is_improved: False
    - summary_notes: ['Vulnerability score increased by 0.41 points.']
    - timestamp: 2026-08-27 17:04:38
    - extra_metadata: {}

  ▶ Vector: fgsm
    - attack_name: fgsm
    - before_assessment: {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568545, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5967, 'f1_drop': 0.6361178561878691, 'confidence_drop': 0.501677930355072, 'model_degradation': 0.5781652621809804, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.4920240044593811, 'timestamp': '2026-08-27 17:03:28', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.41286802291870117}}
    - after_assessment: {'attack_name': 'fgsm', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.9169117696583271, 'l2_mean': 201.17694619568556, 'l0_mean': 0.9860084170386905, 'is_estimated': False}, 'accuracy_drop': 0.5967, 'f1_drop': 0.6361178561878691, 'confidence_drop': 0.501675546169281, 'model_degradation': 0.5781644674523834, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.375, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.2967965367965368, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.4920263886451721, 'timestamp': '2026-08-27 17:04:37', 'extra_metadata': {'attack_class': 'FGSM', 'execution_time_seconds': 0.02091050148010254}}
    - before_scoring: {'attack_name': 'fgsm', 'vulnerability_score': 40.47, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.82, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:03:28', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}}
    - after_scoring: {'attack_name': 'fgsm', 'vulnerability_score': 40.47, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 57.82, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:04:37', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}}
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
    - timestamp: 2026-08-27 17:04:38
    - extra_metadata: {}

  ▶ Vector: deepfool
    - attack_name: deepfool
    - before_assessment: {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4092, 'f1_drop': 0.4853945517145645, 'confidence_drop': 0.5500082969665527, 'model_degradation': 0.48153428289370576, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 17:03:28', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.38034749031067}}
    - after_assessment: {'attack_name': 'deepfool', 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000, 'attack_success_rate': None, 'perturbation': {'linf_mean': 0.8856617696583271, 'l2_mean': 196.76040844030746, 'l0_mean': 0.9611768657658375, 'is_estimated': False}, 'accuracy_drop': 0.4092, 'f1_drop': 0.4853945517145645, 'confidence_drop': 0.5500082969665527, 'model_degradation': 0.48153428289370576, 'clean_accuracy': 0.9717, 'adversarial_accuracy': 0.5625, 'clean_f1': 0.9329143929844058, 'adversarial_f1': 0.4475198412698413, 'clean_confidence': 0.9937019348144531, 'adversarial_confidence': 0.4436936378479004, 'timestamp': '2026-08-27 17:04:38', 'extra_metadata': {'attack_class': 'DeepFool', 'execution_time_seconds': 22.451696157455444}}
    - before_scoring: {'attack_name': 'deepfool', 'vulnerability_score': 33.71, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.15, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:03:28', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}}
    - after_scoring: {'attack_name': 'deepfool', 'vulnerability_score': 33.71, 'risk_level': 'MEDIUM', 'sub_scores': {'asr_score': None, 'degradation_score': 48.15, 'stealth_score': 0.0}, 'timestamp': '2026-08-27 17:04:38', 'metadata': {'weights': {'weight_asr': 0.5, 'weight_degradation': 0.35, 'weight_stealth': 0.15}, 'dataset_name': 'bazyl/GTSRB', 'num_samples': 10000}}
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
    - timestamp: 2026-08-27 17:04:38
    - extra_metadata: {}

13. EXECUTION PERFORMANCE
------------------------------------------------------------------------
  Run Label    : AdverScan [full]
  Started At   : 2026-08-27 17:02:20
  Total Time   : 0.00s
  Overall      : UNKNOWN

  MODULE                         STATUS          TIME
  ········································································

14. RECOMMENDATIONS
------------------------------------------------------------------------
  [01] [MEDIUM] Vulnerability score 33.71 warrants attention. Implement input sanitization and monitor inference traffic for anomalies.
  [02] XAI attribution maps are available. Review highlighted input regions disproportionately targeted by adversarial perturbations to guide robustness patches.
  [03] Adversarial defense was applied. Validate post-hardening accuracy retention and conduct periodic re-tests to ensure defense durability.
  [04] Re-test indicates persistent vulnerability for vector 'pgd'. Increase adversarial training epochs or broaden epsilon schedules.
  [05] Re-test indicates persistent vulnerability for vector 'fgsm'. Increase adversarial training epochs or broaden epsilon schedules.
  [06] Re-test indicates persistent vulnerability for vector 'deepfool'. Increase adversarial training epochs or broaden epsilon schedules.

15. FINAL SECURITY SUMMARY
------------------------------------------------------------------------
  - risk_level: MEDIUM
  - vulnerability_score: 33.71
  - baseline_accuracy: 97.17%
  - mean_adversarial_accuracy: 46.88%
  - attacks_evaluated: ['fgsm', 'pgd', 'deepfool']
  - hardening_applied: True
  - retest_conducted: True
  - total_recommendations: 6
  - primary_recommendation: [MEDIUM] Vulnerability score 33.71 warrants attention. Implement input sanitization and monitor inference traffic for anomalies.

========================================================================
  Generated by AdverScan — 2026-08-27 17:04:38
========================================================================