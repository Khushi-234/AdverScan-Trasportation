# AdverScan — Technical Handover & Repository Status Report

**Project Title:** AdverScan — Adversarial Hardening and Security Testing Platform Across Generalized Domains  
**Repository Branch:** `feature/model-ingestion`  
**Date of Inspection:** August 17, 2026  
**Target Audience:** AI Assistants, Security Researchers, Software Engineers  
**Report Document Path:** [docs/ADVERSCAN_CURRENT_PROJECT_HANDOVER.md](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/docs/ADVERSCAN_CURRENT_PROJECT_HANDOVER.md)  

---

## 1. Project Structure

The complete repository structure of AdverScan as inspected from the file system is presented below.

```
AdverScan/
├── app/                                 # Main application package
│   ├── api/                             # FastAPI REST application
│   │   ├── __init__.py                  # Package init (empty)
│   │   └── main.py                      # Basic FastAPI routes (Root & Health check)
│   ├── attack_engine/                   # Module 3 (M3) — Attack Engine (COMPLETED)
│   │   ├── __init__.py                  # Package exports (__all__)
│   │   ├── attack_discovery.py          # Dynamic module discovery (pkgutil/importlib)
│   │   ├── attack_engine.py             # AttackEngine orchestrator & functional pipeline
│   │   ├── attack_executor.py           # Attack execution wrapper with timing & metadata
│   │   ├── attack_registry.py           # Global attack class catalog & lookup
│   │   ├── attack_selector.py           # Attack validation & compatibility selector
│   │   ├── config.py                    # AttackConfig dataclass
│   │   ├── exceptions.py                # Attack module exception hierarchy
│   │   ├── models.py                    # AttackMetadata, AttackResult, AttackResults dataclasses
│   │   ├── attacks/                     # Implemented attack algorithms
│   │   │   ├── __init__.py              # Package init
│   │   │   ├── deepfool.py              # DeepFool attack implementation
│   │   │   ├── fgsm.py                  # Fast Gradient Sign Method (FGSM) implementation
│   │   │   └── pgd.py                   # Projected Gradient Descent (PGD) implementation
│   │   └── base/                        # Abstract attack base class
│   │       ├── __init__.py              # Package init
│   │       └── base_attack.py           # BaseAttack abstract interface
│   ├── common/                          # Shared utilities & constants (Placeholder)
│   │   └── __init__.py                  # 0-byte file
│   ├── config/                          # Application-wide configuration (Placeholder)
│   │   └── __init__.py                  # 0-byte file
│   ├── dashboard/                       # Web dashboard / UI module (Placeholder)
│   │   └── __init__.py                  # 0-byte file
│   ├── evaluation/                      # Module 2 (M2) — Baseline Evaluation Engine (COMPLETED)
│   │   ├── __init__.py                  # Package exports
│   │   ├── dataset_loader.py            # BaseDatasetLoader (ABC) & GTSRBDatasetLoader
│   │   ├── evaluator.py                 # BaselineEvaluator engine & evaluate_baseline()
│   │   ├── metrics.py                   # MetricsCalculator (Accuracy, Macro/Weighted F1, Entropy, CM)
│   │   └── results.py                   # EvaluationResult dataclass & MLflow logging
│   ├── explainability/                  # Module 6 (M6) — Explainability (Planned/Not Implemented)
│   │   └── __init__.py                  # 0-byte file
│   ├── hardening/                       # Module 5 (M5) — Hardening & Defense (Planned/Not Implemented)
│   │   └── __init__.py                  # 0-byte file
│   ├── ingestion/                       # Module 1 (M1) — Model Ingestion Engine (COMPLETED/FROZEN)
│   │   ├── __init__.py                  # Package exports
│   │   ├── exceptions.py                # Ingestion exception hierarchy
│   │   ├── pipeline.py                  # ingest_model() standard ingestion entry point
│   │   ├── adapters/                    # Model standardization wrappers
│   │   │   ├── __init__.py              # Package init
│   │   │   ├── base_adapter.py          # BaseModelAdapter abstract base class
│   │   │   └── pytorch_adapter.py       # PyTorchAdapter standardized wrapper
│   │   ├── loader/                      # Model loaders
│   │   │   ├── __init__.py              # Package init
│   │   │   ├── base_loader.py           # BaseModelLoader abstract base class
│   │   │   └── pytorch_loader.py        # PyTorchLoader (.pt, .pth, TorchScript, state_dict)
│   │   ├── metadata/                    # Metadata extractions
│   │   │   ├── __init__.py              # Package init
│   │   │   └── model_metadata.py        # ModelMetadata dataclass
│   │   ├── runtime/                     # Execution runtime helpers
│   │   │   ├── __init__.py              # Package init
│   │   │   └── device_manager.py        # DeviceManager (CPU/CUDA mapping)
│   │   └── validation/                  # Model safety validation
│   │       ├── __init__.py              # Package init
│   │       └── validator.py             # ModelValidator (forward check, NaN check)
│   ├── orchestration/                   # Full pipeline orchestrator (Planned/Not Implemented)
│   │   └── __init__.py                  # 0-byte file
│   ├── reporting/                       # Module 7 (M7) — Report Generator (Partial/Not Implemented)
│   │   └── __init__.py                  # 0-byte file
│   ├── scoring/                         # Module 4 (M4) — Risk & Vulnerability Scoring (Planned)
│   │   └── __init__.py                  # 0-byte file
│   ├── utils/                           # General helper functions (Placeholder)
│   │   └── __init__.py                  # 0-byte file
│   └── vulnerability/                   # Module 4 (M4) — Vulnerability Assessment (Planned)
│       └── __init__.py                  # 0-byte file
├── assets/                              # Static visual assets (.gitkeep)
├── configs/                             # Configuration YAML/JSON files (.gitkeep)
├── datasets/                            # Dataset storage directory
│   ├── gtsrb/                           # Local GTSRB placeholder (.gitkeep)
│   └── .gitkeep                         # Git keep placeholder
├── docs/                                # Project documentation & technical reports (.gitkeep)
├── experiments/                         # Experiment logs & configs (.gitkeep)
├── models/                              # Local model checkpoints directory (.gitkeep)
├── notebooks/                           # Jupyter notebooks (.gitkeep)
├── reports/                             # Generated PDF/HTML reports (.gitkeep)
├── results/                             # Evaluation JSON artifacts (.gitkeep)
├── scripts/                             # Evaluation & validation runner scripts
│   ├── run_baseline_evaluation.py       # Full baseline evaluation runner
│   ├── test_gtsrb_10_samples.py         # Multi-attack 10-sample integration test
│   ├── test_gtsrb_50_samples.py         # 50-sample clean baseline integration test
│   ├── test_gtsrb_5000_samples.py       # 5,000-sample GPU baseline test script
│   ├── test_gtsrb_full_dataset.py       # Full 12,630-sample GPU baseline test script
│   ├── test_gtsrb_adversarial_attacks.py# FGSM adversarial evaluation script
│   └── validate_gtsrb_model.py          # Hugging Face ViT model & dataset pipeline validator
├── tests/                               # Comprehensive unit & integration test suite
│   ├── __init__.py                      # Package init
│   ├── test_api.py                      # FastAPI endpoint unit test (Commented out)
│   ├── attack_engine/                   # Module 3 unit tests
│   │   ├── __init__.py                  # Package init
│   │   ├── test_attack_engine.py        # Tests AttackEngine orchestrator
│   │   ├── test_deepfool.py             # Tests DeepFool attack generation
│   │   ├── test_discovery.py            # Tests dynamic attack discovery
│   │   ├── test_executor.py             # Tests attack executor
│   │   ├── test_fgsm.py                 # Tests FGSM attack generation
│   │   ├── test_pgd.py                  # Tests PGD attack generation (1 test fails)
│   │   ├── test_registry.py             # Tests Attack Registry
│   │   └── test_selector.py             # Tests Attack Selector
│   ├── evaluation/                      # Module 2 unit tests
│   │   ├── __init__.py                  # Package init
│   │   ├── test_dataset_loader.py       # Tests GTSRB dataset decoding & preprocessing
│   │   ├── test_evaluator.py            # Tests BaselineEvaluator & logit slicing
│   │   ├── test_metrics.py              # Tests Shannon entropy & classification metrics
│   │   └── test_results.py              # Tests EvaluationResult JSON serialization
│   └── ingestion/                       # Module 1 unit tests
│       ├── __init__.py                  # Package init
│       ├── test_adapter.py              # Tests PyTorchAdapter prediction and input conversion
│       ├── test_device_manager.py       # Tests DeviceManager device selection
│       ├── test_exceptions.py           # Tests exception hierarchy
│       ├── test_integration.py          # Tests end-to-end ingest_model() pipeline
│       ├── test_loader.py               # Tests PyTorchLoader state_dict & TorchScript loading
│       ├── test_metadata.py             # Tests ModelMetadata extraction
│       └── test_validator.py            # Tests ModelValidator forward & NaN checks
├── docker-compose.yml                   # Docker compose config
├── Dockerfile                           # Docker build configuration
├── LICENSE                              # Project license file
├── Makefile                             # Build & test Makefile targets
├── pyproject.toml                       # Build system & package metadata (PEP 621)
├── README.md                            # High-level project documentation
└── requirements.txt                     # Core dependencies & version requirements
```

### Key Python File Breakdown

| File Path | Purpose | Status | Key Classes / Functions | Module Dependencies |
| :--- | :--- | :--- | :--- | :--- |
| `app/ingestion/pipeline.py` | Top-level model ingestion entry point | ✅ Implemented | `ingest_model()` | `adapters`, `loader`, `metadata`, `runtime`, `validation` |
| `app/ingestion/adapters/pytorch_adapter.py` | Standardized PyTorch model wrapper | ✅ Implemented | `PyTorchAdapter` | `base_adapter`, `torch`, `numpy` |
| `app/ingestion/loader/pytorch_loader.py` | Loads PyTorch weights/TorchScript/instance | ✅ Implemented | `PyTorchLoader` | `base_loader`, `exceptions`, `torch` |
| `app/ingestion/metadata/model_metadata.py` | Container for extracted model metadata | ✅ Implemented | `ModelMetadata` | `dataclasses` |
| `app/ingestion/runtime/device_manager.py` | Hardware device (CPU/CUDA) management | ✅ Implemented | `DeviceManager` | `torch` |
| `app/ingestion/validation/validator.py` | Ingestion safety and sanity checker | ✅ Implemented | `ModelValidator` | `exceptions`, `torch` |
| `app/evaluation/dataset_loader.py` | HF GTSRB dataset loader & decoder | ✅ Implemented | `BaseDatasetLoader`, `GTSRBDatasetLoader` | `datasets`, `transformers`, `torch`, `PIL` |
| `app/evaluation/evaluator.py` | Clean baseline evaluation engine | ✅ Implemented | `BaselineEvaluator`, `evaluate_baseline()` | `adapters`, `dataset_loader`, `metrics`, `results` |
| `app/evaluation/metrics.py` | Metric computation (Accuracy, F1, Entropy, CM) | ✅ Implemented | `MetricsCalculator` | `sklearn.metrics`, `numpy` |
| `app/evaluation/results.py` | Evaluation metrics result container | ✅ Implemented | `EvaluationResult` | `dataclasses`, `json`, `mlflow` |
| `app/attack_engine/base/base_attack.py` | Standardized base interface for attacks | ✅ Implemented | `BaseAttack` | `adapters` |
| `app/attack_engine/config.py` | Attack parameters container | ✅ Implemented | `AttackConfig` | `exceptions` |
| `app/attack_engine/models.py` | Attack results and metadata dataclasses | ✅ Implemented | `AttackMetadata`, `AttackResult`, `AttackResults` | `dataclasses` |
| `app/attack_engine/attack_registry.py` | Global registry for attack classes | ✅ Implemented | `register_attack()`, `get_attack()`, `list_attacks()`, `clear_registry()` | `base_attack`, `exceptions` |
| `app/attack_engine/attack_discovery.py` | Dynamic attack module auto-discovery | ✅ Implemented | `discover_attacks()`, `reset_discovery_state()` | `pkgutil`, `importlib` |
| `app/attack_engine/attack_selector.py` | Attack class lookup and compatibility | ✅ Implemented | `select_attacks()`, `select_compatible_attacks()` | `attack_registry`, `attack_discovery` |
| `app/attack_engine/attack_executor.py` | Executes attacks & measures execution time | ✅ Implemented | `execute_attack()` | `base_attack`, `config`, `models`, `exceptions` |
| `app/attack_engine/attack_engine.py` | Multi-attack pipeline orchestrator | ✅ Implemented | `AttackEngine`, `run_attack_pipeline()` | `config`, `models`, `attack_selector`, `attack_executor` |
| `app/attack_engine/attacks/fgsm.py` | Fast Gradient Sign Method attack | ✅ Implemented | `FGSM` | `base_attack`, `config`, `exceptions`, `attack_registry` |
| `app/attack_engine/attacks/pgd.py` | Projected Gradient Descent attack | ✅ Implemented | `PGD` | `base_attack`, `config`, `exceptions`, `attack_registry` |
| `app/attack_engine/attacks/deepfool.py` | Minimal perturbation boundary attack | ✅ Implemented | `DeepFool` | `base_attack`, `config`, `exceptions`, `attack_registry` |
| `app/api/main.py` | Basic REST API entry point | ⚠️ Partial | `read_root()`, `health_check()` | `fastapi` |
| `app/vulnerability/__init__.py` | Vulnerability assessment module | ⏳ Planned | None (0 bytes) | None |
| `app/scoring/__init__.py` | Vulnerability scoring module | ⏳ Planned | None (0 bytes) | None |
| `app/hardening/__init__.py` | Model hardening & defense module | ⏳ Planned | None (0 bytes) | None |
| `app/explainability/__init__.py` | Model explainability & visualization | ⏳ Planned | None (0 bytes) | None |
| `app/reporting/__init__.py` | PDF/HTML report generation module | ⏳ Planned | None (0 bytes) | None |
| `app/orchestration/__init__.py` | Full-system pipeline orchestrator | ⏳ Planned | None (0 bytes) | None |
| `app/dashboard/__init__.py` | Web UI dashboard frontend/backend | ⏳ Planned | None (0 bytes) | None |

---

## 2. Current Architecture

The AdverScan framework architecture is designed around a modular pipeline architecture for security scanning of machine learning models.

```
+------------------+     +-----------------------+     +---------------------------+
| Target ML Model  | --> | M1: Model Ingestion   | --> | M2: Baseline Evaluation   |
| (PyTorch/HF ViT) |     | (Loader/Adapter/Check)|     | (Clean Acc, Conf, Entropy)|
+------------------+     +-----------------------+     +---------------------------+
                                                                     |
                                                                     v
+------------------+     +-----------------------+     +---------------------------+
| M5: Hardening    | <-- | M4: Vulnerability     | <-- | M3: Attack Engine         |
| (Defenses/AT)    |     | Scoring (ASR/Drop)    |     | (FGSM, PGD, DeepFool)     |
| [⏳ Planned]     |     | [⏳ Planned]          |     | [✅ Implemented]          |
+------------------+     +-----------------------+     +---------------------------+
         |                                                           |
         v                                                           v
+------------------+                                   +---------------------------+
| M7: Reporting    | <-------------------------------- | M6: Explainability        |
| (JSON/MLflow/PDF)|                                   | (Grad-CAM/Perturbations)  |
| [⚠️ Partial]     |                                   | [⏳ Planned]              |
+------------------+                                   +---------------------------+
```

### Module Status Summary

*   ✅ **M1 — Model Ingestion Engine**: **Fully Implemented & Frozen**. Standardizes arbitrary PyTorch models into unified `BaseModelAdapter` instances.
*   ✅ **M2 — Baseline Evaluation Engine**: **Fully Implemented**. Evaluates clean accuracy, macro/weighted F1, confidence, Shannon entropy, and confusion matrix.
*   ✅ **M3 — Attack Engine**: **Fully Implemented**. Provides modular dynamic attack discovery, registry, selection, execution, and 3 attacks (FGSM, PGD, DeepFool).
*   ⏳ **M4 — Vulnerability Scoring**: **Not Implemented**. Directory structures (`app/vulnerability/`, `app/scoring/`) exist as empty placeholders.
*   ⏳ **M5 — Hardening**: **Not Implemented**. Directory `app/hardening/` exists as an empty placeholder.
*   ⏳ **M6 — Explainability**: **Not Implemented**. Directory `app/explainability/` exists as an empty placeholder.
*   ⚠️ **M7 — Reporting**: **Partially Implemented**. `EvaluationResult` includes JSON output and basic MLflow logging. `app/reporting/` package is empty.
*   ⚠️ **API / Dashboard / Orchestration**: **Partially Implemented / Planned**. `app/api/main.py` contains basic FastAPI health endpoints; `app/dashboard/` and `app/orchestration/` are empty placeholders.

---

## 3. M1 — Model Ingestion Engine

### Architecture & Mechanics
Module 1 serves as the unified entry point for wrapping external PyTorch models into standardized internal representations used by downstream evaluation and attack engines.

```
Model Source (.pt/.pth/Module) -> PyTorchLoader -> PyTorchAdapter -> ModelValidator -> ModelMetadata
```

1.  **Device Manager** (`app/ingestion/runtime/device_manager.py`): Automatically selects hardware device (`cuda` if available, otherwise `cpu`).
2.  **PyTorch Loader** (`app/ingestion/loader/pytorch_loader.py`): Accepts file paths (`.pt`, `.pth`), state dictionaries, TorchScript modules, or existing `nn.Module` instances. Handles PyTorch 2.6+ `weights_only` loading behavior.
3.  **PyTorch Adapter** (`app/ingestion/adapters/pytorch_adapter.py`): Wraps the PyTorch `nn.Module` inside `BaseModelAdapter`. Provides unified `.predict(inputs)` and callable `__call__()` methods, automatic NumPy/Tensor input conversion, device movement, and `eval()`/`train()` controls.
4.  **Model Validator** (`app/ingestion/validation/validator.py`): Conducts sanity checks by feeding optional `sample_input` through the model to verify output existence and check for `NaN` or infinite output values.
5.  **Model Metadata** (`app/ingestion/metadata/model_metadata.py`): Captures model name, framework, task type, target device, input shape, output shape, and active class count.
6.  **Pipeline Entrypoint** (`app/ingestion/pipeline.py`): Provides `ingest_model()` function that returns `Tuple[PyTorchAdapter, ModelMetadata]`.

### Public APIs & Contracts (M1 → M2 / M3)
*   **Primary Function**: `ingest_model(model_path, sample_input=None, device=None, model_class=None, model_name=None, task_type='classification', **kwargs)`
*   **Adapter Contract**: All downstream modules consume `BaseModelAdapter` through `predict(inputs, return_numpy=False)` or direct calling `adapter(inputs)`.

### Test Suite Status
*   **Test Location**: `tests/ingestion/` (8 test modules)
*   **Total Tests**: **32 unit and integration tests**
*   **Status**: **32 / 32 PASSED (100% Pass Rate)**

> [!IMPORTANT]
> **Module Freeze Notice:** M1 is stable, completely tested, and frozen. No changes should be made to M1 unless critical architectural breaking changes require it.

---

## 4. M2 — Baseline Evaluation Engine

### Architecture & Mechanics
Module 2 executes clean baseline evaluation over datasets (specifically GTSRB for Intelligent Transportation Systems).

1.  **Dataset Loader** (`app/evaluation/dataset_loader.py`): Defines `BaseDatasetLoader` (ABC) and `GTSRBDatasetLoader` (Hugging Face `bazyl/GTSRB` split loader). Reads raw byte payloads or PIL images, decodes RGB format, applies `AutoImageProcessor` (`bazyl/gtsrb-model`), and yields mini-batches of preprocessed `pixel_values` (shape `(B, 3, 224, 224)`) and ground-truth targets.
2.  **Metrics Calculator** (`app/evaluation/metrics.py`): Calculates classification metrics via `scikit-learn`: Overall Accuracy, Macro Precision/Recall/F1, Weighted Precision/Recall/F1, Per-Class Precision/Recall/F1/Support, Confusion Matrix ($43 \times 43$), Average Prediction Confidence, and Shannon Entropy ($H(p) = -\sum p_i \log_2(p_i)$).
3.  **Baseline Evaluator** (`app/evaluation/evaluator.py`): `BaselineEvaluator` iterates over batches yielded by dataset loader, passes inputs to M1 adapter, handles HuggingFace output objects (`ImageClassifierOutput`), slices logits to active target classes, computes probabilities/predictions, accumulates metrics, and returns `EvaluationResult`.
4.  **Result Persistence** (`app/evaluation/results.py`): `EvaluationResult` dataclass supports `.to_dict()`, `.save_json(output_path)`, and `.log_to_mlflow()`.

### GTSRB Specifics & The 44 → 43 Logits Issue
*   **Dataset**: `bazyl/GTSRB` (German Traffic Sign Recognition Benchmark).
*   **Classes**: 43 active German traffic sign classes (class IDs `0` through `42`).
*   **Model**: Vision Transformer (`bazyl/gtsrb-model` / `AutoModelForImageClassification`).
*   **Upstream Config Anomaly**: The upstream model configuration (`config.json`) defines `num_labels: 44`. Index `43` in `id2label` is `null` (unused label).
*   **Logit Slicing Mechanism**: The raw model output logits tensor has shape `(B, 44)`. In `BaselineEvaluator` (and evaluation scripts), the logits are explicitly sliced to active GTSRB classes:
    ```python
    if logits_tensor.shape[-1] > self.num_classes:
        logits_tensor = logits_tensor[:, : self.num_classes]  # Slices (B, 44) -> (B, 43)
    ```

### Test Suite Status
*   **Test Location**: `tests/evaluation/` (4 test modules)
*   **Total Tests**: **11 unit & integration tests**
*   **Status**: **11 / 11 PASSED (100% Pass Rate)**

---

## 5. M3 — Attack Engine

Module 3 is the core adversarial attack generation engine. It dynamically cataloging, selects, and executes attacks against M1 model adapters.

### M3 Pipeline Architecture & Dataflow

```
User / Integration Script
        │
        ▼
   AttackEngine / run_attack_pipeline()
        │
        ├──> 1. discover_attacks() ──> Scans app/attack_engine/attacks/ via pkgutil
        │                              Imports modules & triggers register_attack()
        │
        ├──> 2. select_attacks()   ──> Validates attack names against AttackRegistry
        │                              Returns attack classes (e.g. FGSM, PGD, DeepFool)
        │
        └──> 3. execute_attack()   ──> Instantiates attack class with model
                                       Runs attack.generate(inputs, labels, config)
                                       Measures runtime & returns AttackResult
```

### Implemented Attack Algorithms

AdverScan currently implements **THREE** white-box adversarial attacks:

#### 1. Fast Gradient Sign Method (FGSM)
*   **File**: `app/attack_engine/attacks/fgsm.py`
*   **Mathematical Formulation**:
    $$\boldsymbol{x}_{\text{adv}} = \boldsymbol{x} + \epsilon \cdot \text{sign}\left(\nabla_{\boldsymbol{x}} \mathcal{L}(\theta, \boldsymbol{x}, y)\right)$$
*   **Algorithm**: Single-step gradient ascent along sign of loss gradient w.r.t input.
*   **Parameters & Defaults**: `epsilon=0.1` (budget), `clip_min=0.0`, `clip_max=1.0`, `loss_fn=CrossEntropyLoss()`.
*   **Targeted / Untargeted**: Untargeted (maximizes loss for true label $y$).
*   **Gradient Requirements**: Requires first-order input gradient (`inputs.requires_grad = True`).
*   **Clipping**: Applies `torch.clamp(adv_inputs, clip_min, clip_max)` if bounds specified.

#### 2. Projected Gradient Descent (PGD)
*   **File**: `app/attack_engine/attacks/pgd.py`
*   **Mathematical Formulation**:
    $$\boldsymbol{x}_0 = \boldsymbol{x} + U(-\epsilon, \epsilon)$$
    $$\boldsymbol{x}_{t+1} = \Pi_{\boldsymbol{x} + \mathcal{S}}\left( \text{Clip}\left(\boldsymbol{x}_t + \alpha \cdot \text{sign}\left(\nabla_{\boldsymbol{x}_t} \mathcal{L}(\theta, \boldsymbol{x}_t, y)\right)\right) \right)$$
*   **Algorithm**: Multi-step iterative gradient ascent with uniform random start inside $\epsilon$-ball and projection $\Pi$ back into $\epsilon$-ball around original input.
*   **Parameters & Defaults**: `epsilon=0.1`, `num_steps=10` (or `steps`/`iters`), `random_start=True`, `alpha=2.0 * epsilon / num_steps` (step size), `clip_min=0.0`, `clip_max=1.0`, `loss_fn=CrossEntropyLoss()`.
*   **Targeted / Untargeted**: Untargeted.
*   **Gradient Requirements**: Requires gradient backpropagation at each iteration via `torch.autograd.grad`.

#### 3. DeepFool
*   **File**: `app/attack_engine/attacks/deepfool.py`
*   **Mathematical Formulation**: Computes minimal $L_2$ perturbation vector to push sample across closest decision boundary:
    $$\arg\min_{\boldsymbol{r}_i} \|\boldsymbol{r}_i\|_2 \quad \text{s.t.} \quad f(\boldsymbol{x} + \boldsymbol{r}_i) \neq \hat{k}(\boldsymbol{x})$$
    $$\boldsymbol{r}_i^* = \frac{|f_k(\boldsymbol{x}_i) - f_{\hat{k}}(\boldsymbol{x}_i)|}{\|\boldsymbol{w}_k^\prime\|_2^2} \boldsymbol{w}_k^\prime \quad \text{where} \quad \boldsymbol{w}_k^\prime = \nabla f_k(\boldsymbol{x}_i) - \nabla f_{\hat{k}}(\boldsymbol{x}_i)$$
*   **Algorithm**: Iterative linear approximation of decision boundary. Evaluates candidate classes (Top-$K$), finds nearest hyper-plane, steps with overshoot factor $(1 + \eta)$, clips to bounds and optional $\epsilon$-ball.
*   **Parameters & Defaults**: `max_iter=10`, `overshoot=0.02`, `top_k=10`, `epsilon=inf`, `clip_min=0.0`, `clip_max=1.0`.
*   **Gradient Requirements**: Computes per-class gradient vectors w.r.t input logits.

### M3 Component Details
*   **BaseAttack** (`app/attack_engine/base/base_attack.py`): Abstract base class defining `generate(inputs, labels, config)`. Includes `_get_raw_model()` helper to extract underlying `nn.Module` from `BaseModelAdapter`.
*   **AttackConfig** (`app/attack_engine/config.py`): Configuration dataclass containing `epsilon`, `clip_min`, `clip_max`, `loss_fn`, and `params` dictionary.
*   **AttackRegistry** (`app/attack_engine/attack_registry.py`): Global catalog mapping attack names to attack classes. Provides `register_attack()`, `get_attack()`, `list_attacks()`, and `clear_registry()`.
*   **AttackDiscovery** (`app/attack_engine/attack_discovery.py`): Dynamically inspects `app/attack_engine/attacks/` package using `pkgutil.iter_modules()` and `importlib.import_module()`.
*   **AttackSelector** (`app/attack_engine/attack_selector.py`): Validates requested attack names and returns matching attack classes.
*   **AttackExecutor** (`app/attack_engine/attack_executor.py`): Instantiates selected attack class, runs generation, measures execution time, and constructs `AttackResult`.
*   **AttackEngine** (`app/attack_engine/attack_engine.py`): High-level orchestrator class providing `.run_attack()` and `.run_pipeline()` methods, as well as functional helper `run_attack_pipeline()`.
*   **Data Models** (`app/attack_engine/models.py`): `AttackMetadata`, `AttackResult`, and `AttackResults` (dictionary collection mapping attack names to `AttackResult`).

---

## 6. M3 Testing & Verification Status

### Unit & Integration Test Breakdown (`tests/attack_engine/`)

| Test File | Tests | Purpose | Verification Scope | Status |
| :--- | :---: | :--- | :--- | :---: |
| `test_attack_engine.py` | 4 | Orchestrator pipeline & helper exports | Synthetic Tensors / Mock Model | **PASSED** |
| `test_deepfool.py` | 3 | DeepFool generation, registration, invalid inputs | Synthetic Tensors / ConvNet | **PASSED** |
| `test_discovery.py` | 1 | Dynamic package discovery | Module Import Checks | **PASSED** |
| `test_executor.py` | 3 | Execution on raw models vs adapters | Synthetic Tensors / ConvNet | **PASSED** |
| `test_fgsm.py` | 3 | FGSM generation, clipping, input validation | Synthetic Tensors / ConvNet | **PASSED** |
| `test_pgd.py` | 4 | PGD generation, clipping, registration | Synthetic Tensors / ConvNet | ⚠️ **1 FAILED** |
| `test_registry.py` | 6 | Registry lookup, case insensitivity, error handling | Registry Internal State | **PASSED** |
| `test_selector.py` | 5 | Selection validation & compatibility | Registry & Selector | **PASSED** |

> [!WARNING]
> **Known Unit Test Failure:** `test_pgd.py::test_pgd_registration` failed because line 32 of `test_pgd.py` asserts `"pgm"` in registered attacks (`assert "pgm" in attacks`), but `app/attack_engine/attacks/pgd.py` only calls `register_attack("pgd", PGD)`. This is a minor typo bug in `pgd.py` registration logic.

### Verification Categories

1.  **Unit-Test Verification**: 29 tests total (28 PASSED, 1 FAILED). Uses small synthetic PyTorch tensors (`torch.randn`) and a simple mock convolutional model (`SimpleConvNet`).
2.  **Integration-Test Verification**: `test_attack_engine.py` and `test_executor.py` verify integration between `BaseModelAdapter` (M1) and `AttackEngine` (M3). All PASSED.
3.  **Real-Data Attack Verification**: Executed via integration scripts `scripts/test_gtsrb_10_samples.py` and `scripts/test_gtsrb_adversarial_attacks.py` on real GTSRB traffic sign images from `bazyl/GTSRB` and `bazyl/gtsrb-model`. Verified FGSM, PGD, and DeepFool attack generations on real vision transformer images.
4.  **Full Benchmark Verification**: **NOT EXECUTED / NO SAVED DISK ARTIFACT**. Full-dataset adversarial attack benchmarking over all 12,630 GTSRB samples has not been executed due to compute limitations.

---

## 7. M4 — Vulnerability Scoring

### Current State
*   **Status**: ⏳ **Planned / Not Implemented**.
*   **Repository Footprint**: Directories `app/vulnerability/` and `app/scoring/` exist but contain only empty `__init__.py` files (0 bytes).

### Existing Interfaces for Future M4 Consumption
M3 and M2 provide complete output interfaces that M4 can consume directly:
*   `AttackResult` (`app/attack_engine/models.py`): Contains clean inputs, ground truth labels, generated adversarial examples, and execution metadata (`epsilon`, runtime, parameters).
*   `MetricsCalculator` (`app/evaluation/metrics.py`): Computes baseline vs adversarial metrics.
*   **Available Metrics to Compute Risk Score**:
    1.  **Attack Success Rate (ASR)**: $\text{ASR} = \frac{\sum \mathbb{I}(y_{\text{clean}} = y \land y_{\text{adv}} \neq y)}{\sum \mathbb{I}(y_{\text{clean}} = y)}$
    2.  **Accuracy Drop**: $\Delta \text{Acc} = \text{Acc}_{\text{clean}} - \text{Acc}_{\text{adv}}$
    3.  **Confidence Degradation**: $\Delta \text{Conf} = \text{Conf}_{\text{clean}} - \text{Conf}_{\text{adv}}$
    4.  **Entropy Elevation**: $\Delta H = H_{\text{adv}} - H_{\text{clean}}$

---

## 8. M5 — Hardening

### Current State
*   **Status**: ⏳ **Planned / Not Implemented**.
*   **Repository Footprint**: Directory `app/hardening/` exists but contains only an empty `__init__.py` file (0 bytes).
*   **Planned Functionality**: Adversarial training (PGD adversarial training), input preprocessing defenses (spatial smoothing, JPEG compression, bit-depth reduction), and perturbation detection.

---

## 9. M6 — Explainability

### Current State
*   **Status**: ⏳ **Planned / Not Implemented**.
*   **Repository Footprint**: Directory `app/explainability/` exists but contains only an empty `__init__.py` file (0 bytes).
*   **Data Dependencies Ready for M6 Integration**:
    *   Clean image tensors $\boldsymbol{x}$ and adversarial image tensors $\boldsymbol{x}_{\text{adv}}$.
    *   Perturbation noise map: $\boldsymbol{\eta} = \boldsymbol{x}_{\text{adv}} - \boldsymbol{x}$.
    *   Input gradient tensors $\nabla_{\boldsymbol{x}} \mathcal{L}$ for Grad-CAM / Saliency map generation.

---

## 10. M7 — Reporting

### Current State
*   **Status**: ⚠️ **Partially Implemented**.
*   **Implemented Artifact Serialization**:
    *   `EvaluationResult.save_json(path)` serializes clean baseline metrics to JSON format.
    *   `scripts/test_gtsrb_adversarial_attacks.py` serializes adversarial attack results, ASR, accuracy drop, and confidence statistics to JSON artifacts in `results/adversarial/`.
    *   `EvaluationResult.log_to_mlflow()` provides basic logging of baseline parameters and metrics to MLflow runs.
*   **Unimplemented**: `app/reporting/` package is empty (0 bytes). Automated PDF, HTML, or executive summary report generation is planned for future work.

---

## 11. API / Dashboard / Orchestration

### Current State
*   **`app/api/main.py`**: ⚠️ **Partial Endpoint Stub**. Defines a minimal FastAPI application with two routes:
    *   `GET /`: Returns `{"name": "AdverScan API", "status": "operational", "version": "0.1.0"}`
    *   `GET /health`: Returns `{"status": "healthy"}`
    *   *Note*: `tests/test_api.py` is currently commented out.
*   **`app/dashboard/`**: ⏳ **Planned / Not Implemented** (`__init__.py` is 0 bytes). No web interface exists.
*   **`app/orchestration/`**: ⏳ **Planned / Not Implemented** (`__init__.py` is 0 bytes). Pipeline orchestration is currently executed via python scripts in `scripts/`.

---

## 12. Configuration & Dependencies

### Key Dependencies (`requirements.txt` & `pyproject.toml`)
*   **Python Target**: `>=3.10` (Virtualenv running Python `3.12.3` at `.venv`)
*   **Core Machine Learning**: `torch>=2.0.0`, `torchvision>=0.15.0`, `transformers>=4.30.0`, `datasets>=2.0.0`, `scikit-learn>=1.2.0`, `numpy>=1.24.0`, `pandas>=2.0.0`
*   **Adversarial Security Tools**: `art>=1.14.0`, `foolbox>=3.3.0`
*   **API & Backend**: `fastapi>=0.100.0`, `uvicorn>=0.22.0`, `pydantic>=2.0.0`
*   **Tracking & Formatting**: `mlflow>=2.0.0`, `pyyaml>=6.0`, `jinja2>=3.1.0`
*   **Testing & Quality**: `pytest>=7.0.0`, `pytest-cov>=4.0.0`, `black>=23.0.0`, `isort>=5.12.0`, `flake8>=6.0.0`, `mypy>=1.0.0`

---

## 13. Test Suite — Complete Status Summary

| Module | Subpackage Path | Test Files | Total Tests | Status |
| :--- | :--- | :---: | :---: | :---: |
| **M1: Model Ingestion** | `tests/ingestion/` | 8 | 32 | **32 / 32 PASSED (100%)** |
| **M2: Baseline Evaluation** | `tests/evaluation/` | 4 | 11 | **11 / 11 PASSED (100%)** |
| **M3: Attack Engine** | `tests/attack_engine/` | 8 | 29 | ⚠️ **28 PASSED, 1 FAILED** |
| **M4: Vulnerability Scoring**| N/A | 0 | 0 | ⏳ Not Implemented |
| **M5: Hardening** | N/A | 0 | 0 | ⏳ Not Implemented |
| **M6: Explainability** | N/A | 0 | 0 | ⏳ Not Implemented |
| **M7: Reporting** | N/A | 0 | 0 | ⏳ Not Implemented |
| **API** | `tests/test_api.py` | 1 | 0 | ⚠️ Commented Out |
| **TOTAL** | | **21** | **72** | **71 PASSED, 1 FAILED (98.6%)** |

---

## 14. Real Data Validation

### Actually Executed Real-Data Experiments

1.  **Single Image Model & Gradient Validation**:
    *   **Script**: `scripts/validate_gtsrb_model.py`
    *   **Dataset**: `bazyl/GTSRB` (1 sample)
    *   **Model**: `bazyl/gtsrb-model` (ViT)
    *   **Execution Device**: CPU
    *   **Results**: Verified image decoding, logit output shape `(1, 44)` sliced to `(1, 43)`, single-sample accuracy 100%, and non-zero gradient backpropagation norm (`0.021045`).
2.  **10-Sample GTSRB Baseline & Multi-Attack Integration Test**:
    *   **Script**: `scripts/test_gtsrb_10_samples.py`
    *   **Dataset**: `bazyl/GTSRB` (10 samples, `test[:10]`)
    *   **Model**: `bazyl/gtsrb-model` (ViT)
    *   **Attacks Tested**: FGSM, PGD, DeepFool ($\epsilon = 0.03137$)
    *   **Execution Device**: CPU / GPU
    *   **Results**:
        *   Clean Baseline Accuracy: **100.00%** (10/10 correct, Avg Conf: 98.92%, Avg Entropy: 0.0926 bits)
        *   FGSM ($\epsilon=0.03137$): Adv Acc **10.00%**, ASR **90.00%**, Acc Drop **90.00%**
        *   PGD ($\epsilon=0.03137$, 10 steps): Adv Acc **0.00%**, ASR **100.00%**, Acc Drop **100.00%**
        *   DeepFool (10 steps): Adv Acc **0.00%**, ASR **100.00%**, Acc Drop **100.00%**
        *   Worst-Case Combined Adv Acc: **0.00%**, Combined ASR: **100.00%**
3.  **50-Sample GTSRB Baseline Evaluation**:
    *   **Script**: `scripts/test_gtsrb_50_samples.py`
    *   **Dataset**: `bazyl/GTSRB` (50 samples, `test[:50]`)
    *   **Model**: `bazyl/gtsrb-model` (ViT)
    *   **Execution Device**: CPU
    *   **Results**: Clean Baseline Accuracy: **98.00%** (49/50 correct, Avg Conf: 98.65%, Avg Entropy: 0.1042 bits).

### Implemented Scripts NOT Executed / NO Saved Artifacts on Disk

1.  **5,000-Sample GTSRB Baseline Integration Script**:
    *   **Script**: `scripts/test_gtsrb_5000_samples.py`
    *   **Status**: ⚠️ **IMPLEMENTED BUT NOT EXECUTED ON DISK**. No saved output file exists in `results/baseline/`.
2.  **Full-Dataset GTSRB Baseline Integration Script**:
    *   **Script**: `scripts/test_gtsrb_full_dataset.py`
    *   **Status**: ⚠️ **IMPLEMENTED BUT NOT EXECUTED ON DISK**. No saved output file exists in `results/baseline/`.
3.  **Full Adversarial Benchmark**:
    *   **Script**: `scripts/test_gtsrb_adversarial_attacks.py`
    *   **Status**: ⚠️ **IMPLEMENTED BUT NOT EXECUTED FOR FULL DATASET**.

---

## 15. Current Git & Working Tree State

*   **Current Branch**: `feature/model-ingestion`
*   **Branch Status**: Ahead of `origin/feature/model-ingestion` by 8 commits. Working tree clean.
*   **Recent Commit Log**:
    *   `bc41ec0`: PGD and DEEPFOOL attack implementation
    *   `a256ae8`: Model file in M3
    *   `a5c8ed5`: Model file in M3
    *   `1d3aea7`: Attack engine implementation in 10 sample test
    *   `b17d8d4`: Attack discovery file
    *   `1e4688c`: M3-Attack_engine_executing test files
    *   `349f142`: Merge attack engine changes into model ingestion
    *   `1a1d017`: Testing_file for_model_implementation

---

## 16. Security & Research Alignment

1.  **Threat Model**: White-box evasion attacks targeting computer vision classifiers (specifically ViT architectures in Intelligent Transportation Systems). The adversary has full access to model architecture and parameters to compute input gradients.
2.  **Attack Surface**: Input pixel space ($\boldsymbol{x} \in \mathbb{R}^{3 \times 224 \times 224}$).
3.  **Attack Techniques**: Fast single-step perturbation (FGSM), iterative projected perturbation (PGD), and decision boundary minimal distance perturbation (DeepFool).
4.  **Security Evaluation Metrics**: Attack Success Rate (ASR), Accuracy Drop ($\Delta \text{Acc}$), Confidence Degradation, and Shannon Entropy Elevation.
5.  **Research Gaps**:
    *   Lack of black-box / transferability attacks (e.g. Square Attack, Boundary Attack).
    *   Lack of defensive transformation benchmarking (M5).
    *   Lack of vulnerability risk score quantization / CVSS mapping (M4).

---

## 17. Important Design Contracts

### M1 → M2 / M3 Contract
*   **Interface**: `BaseModelAdapter`
*   **Methods**: `predict(inputs, return_numpy=False)` returns `torch.Tensor` or `np.ndarray`.
*   **Device Handling**: Model resides on `adapter.device`. Inputs passed into `adapter` are automatically moved to target device.

### M2 → M3 Contract
*   **Logit Alignment**: Slices raw logits to active classes count: `logits[:, :num_classes]`.

### M3 → M4 Contract (Data Contracts)
*   **`AttackResult` Dataclass**:
    ```python
    @dataclass
    class AttackResult:
        adversarial_examples: torch.Tensor
        metadata: AttackMetadata
        original_inputs: Optional[torch.Tensor]
        labels: Optional[torch.Tensor]
    ```
*   **`AttackResults` Dataclass**: Key-value map of registered attack names to `AttackResult`.

---

## 18. Current Project Roadmap

### COMPLETED
- [x] **M1 (Model Ingestion Engine)**: PyTorch loader, PyTorch adapter, device manager, model validator, model metadata.
- [x] **M2 (Baseline Evaluation Engine)**: GTSRB dataset loader, metrics calculator, baseline evaluator engine, `EvaluationResult` model.
- [x] **M3 (Attack Engine)**: Base attack interface, attack config, attack registry, dynamic attack discovery, attack selector, attack executor, `AttackEngine` orchestrator, FGSM, PGD, DeepFool.

### REMAINING
- [ ] **M4 (Vulnerability Scoring)**: Risk score calculation, ASR aggregation, severity rating, vulnerability index.
- [ ] **M5 (Hardening & Defense)**: PGD adversarial training, defensive input preprocessing, perturbation detection.
- [ ] **M6 (Explainability)**: Grad-CAM heatmap visualization, saliency map generation, perturbation visualizers.
- [ ] **M7 (Reporting Engine)**: Automated PDF report generation, HTML dashboards, executive security summary generator.
- [ ] **API & Dashboard**: Full REST API endpoints connecting M1–M7 and Web UI front-end interface.

---

## 19. Critical Issues & Technical Debt

### HIGH SEVERITY
1.  **PGD Registration Test Failure**:
    *   *Issue*: `tests/attack_engine/test_pgd.py` fails on `test_pgd_registration` because line 32 asserts `"pgm"` in registered attacks.
    *   *Root Cause*: `app/attack_engine/attacks/pgd.py` line 142 only registers `"pgd"`. The comment claims it registers both `"pgd"` and `"pgm"`, but `register_attack("pgm", PGD)` was omitted.
2.  **Upstream Hugging Face Config Null Label Bug**:
    *   *Issue*: `bazyl/gtsrb-model` contains `id2label["43"]: null` in its remote `config.json`. Calling `AutoModelForImageClassification.from_pretrained("bazyl/gtsrb-model")` directly without patching fails during label initialization.
    *   *Workaround in Scripts*: Scripts manually download `config.json` via `hf_hub_download`, replace `null` with `"Unused"`, and load `ViTConfig.from_dict()`.
3.  **CPU Heavy Workload Instability**:
    *   *Issue*: Running multi-thousand sample evaluations or iterative attacks (PGD/DeepFool) over full GTSRB datasets (12,630 samples) on CPU causes high CPU load, thread contention, and potential system hangs/shutdowns.
    *   *Rule*: All large evaluations MUST specify `--device cuda` or restrict sample count via `--samples N` when running on CPU.

### MEDIUM SEVERITY
4.  **44 → 43 Model Logits Mismatch**:
    *   *Issue*: `bazyl/gtsrb-model` outputs logits of shape `(B, 44)`, but GTSRB has only 43 active classes. Slicing `logits[:, :43]` is currently performed manually in `BaselineEvaluator` and scripts.
5.  **Commented-Out API Tests**:
    *   *Issue*: `tests/test_api.py` is entirely commented out. The FastAPI application in `app/api/main.py` is not covered by active unit tests.
6.  **Empty Artifact Persistence in Repository**:
    *   *Issue*: Directory `results/` is completely empty (except `.gitkeep`). No baseline or adversarial evaluation JSON artifacts from past runs are checked into git.

### LOW SEVERITY
7.  **PyTorch JIT Loader Deprecation Warning**:
    *   *Issue*: PyTorch 2.6 emits a `DeprecationWarning: torch.jit.load is deprecated` during `PyTorchLoader` execution.
    *   *Action*: Future refactoring should adopt `torch.export` when supported.

---

## 20. Final Handover Summary

### A. What is Definitely COMPLETE
*   **M1 Model Ingestion Engine**: Fully functional PyTorch loader, PyTorch adapter, device manager, model validator, and model metadata. 100% test pass rate (32/32 tests).
*   **M2 Baseline Evaluation Engine**: Full GTSRB dataset loader, metrics calculation (accuracy, macro/weighted F1, average confidence, Shannon entropy, confusion matrix), and baseline evaluator engine. 100% test pass rate (11/11 tests).
*   **M3 Attack Engine Framework & Algorithms**: Dynamic attack discovery, attack registry, attack selector, attack executor, and high-level `AttackEngine` orchestrator. Three fully implemented attacks: FGSM, PGD, and DeepFool. 28 out of 29 tests pass.

### B. What is PARTIAL
*   **M7 Reporting**: Basic JSON serialization and MLflow helper in `EvaluationResult`. Package `app/reporting/` is empty.
*   **REST API**: Minimal FastAPI stub (`app/api/main.py`) with root and health routes. `tests/test_api.py` is commented out.

### C. What is NOT IMPLEMENTED
*   **M4 Vulnerability Scoring** (`app/vulnerability/`, `app/scoring/`)
*   **M5 Hardening & Defense** (`app/hardening/`)
*   **M6 Explainability & Visualization** (`app/explainability/`)
*   **Dashboard & Web UI** (`app/dashboard/`)
*   **System Orchestrator** (`app/orchestration/`)

### D. What Has Actually Been Tested
*   72 pytest unit/integration tests across M1, M2, and M3 (71 passed, 1 failed due to `pgm` alias registration typo).
*   Real-data single-image validation (`scripts/validate_gtsrb_model.py`).
*   Real-data 10-sample multi-attack integration test (`scripts/test_gtsrb_10_samples.py`) verifying FGSM, PGD, and DeepFool on Hugging Face ViT GTSRB model.
*   Real-data 50-sample clean baseline integration test (`scripts/test_gtsrb_50_samples.py`).

### E. What Has NOT Been Tested / Executed
*   Full 12,630-sample GTSRB dataset clean baseline evaluation script (`scripts/test_gtsrb_full_dataset.py` exists but no saved artifact exists on disk).
*   Full 12,630-sample GTSRB dataset adversarial attack benchmark.

### F. Current Architecture & Design Contracts
*   M1 provides `BaseModelAdapter` wrapper with unified `.predict()` and `__call__()` methods.
*   M2 iterates over dataset batches, feeds `pixel_values` to adapter, slices logits `(B, 44) -> (B, 43)`, and computes metrics.
*   M3 consumes M1 `BaseModelAdapter`, extracts raw model via `_get_raw_model()`, generates adversarial perturbations, and returns `AttackResult` dataclass.

### G. Remaining Work for Next AI Assistant
1.  **Fix PGD Registration Typo**: Add `register_attack("pgm", PGD)` in `app/attack_engine/attacks/pgd.py` so `test_pgd.py::test_pgd_registration` passes.
2.  **Implement M4 Vulnerability Scoring**: Create scoring module in `app/scoring/` or `app/vulnerability/` to consume `AttackResult` / `AttackResults` and compute unified risk scores, vulnerability ratings, and ASR degradation indexes.
3.  **Implement M5 Hardening**: Implement PGD adversarial training and defensive input transformations in `app/hardening/`.
4.  **Implement M6 Explainability**: Implement Grad-CAM, saliency maps, and perturbation visualizers in `app/explainability/`.
5.  **Implement M7 Reporting**: Build automated PDF/HTML report generators in `app/reporting/`.
6.  **Build API / Dashboard**: Complete FastAPI endpoints in `app/api/` and dashboard UI in `app/dashboard/`.

---
*Report compiled automatically by Antigravity AI assistant.*
