# AdverScan — Model Ingestion & Standardization Module

## 1. Overview & Objective

The **Model Ingestion Module** (`app/ingestion/`) is **Module 1** of the AdverScan framework. Its primary objective is to ingest, adapt, validate, and standardize heterogeneous machine learning models from diverse sources into a unified, framework-agnostic interface.

By standardizing models at the entry point, downstream security assessment engines—such as adversarial attack generators, vulnerability evaluators, robustness scorers, and defense/hardening mechanisms—can interact with any model via identical calling conventions without worrying about the underlying architecture, weight formats, or device specifics.

```
                      ┌─────────────────────────────────────────┐
                      │              Model Sources              │
                      │  • Local Checkpoints (.pt, .pth, TorchScript)
                      │  • Hugging Face Hub (Vision, NLP, etc.) │
                      │  • In-Memory PyTorch nn.Module Instances│
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │        app/ingestion/ (Module 1)        │
                      │  1. Hardware Detection & Device Mapping │
                      │  2. Format Loading & Weight Injection   │
                      │  3. Standardized Adapter Encapsulation  │
                      │  4. Validation & Numerical Sanity Check │
                      │  5. Metadata Extraction & Audit Logging │
                      └────────────────────┬────────────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
            Standardized Model Adapter                  Extracted Metadata
            (app.ingestion.PyTorchAdapter)         (app.ingestion.ModelMetadata)
                      │                                         │
                      └────────────────────┬────────────────────┘
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │           Downstream Modules            │
                      │  • app/attacks/       (FGSM, PGD, C&W)  │
                      │  • app/evaluation/    (Baseline, Robustness)
                      │  • app/vulnerability/ (Threat Scans)    │
                      │  • app/hardening/     (Adversarial Train)
                      └─────────────────────────────────────────┘
```

---

## 2. Directory & Component Structure

The module is organized under `app/ingestion/` as follows:

```
app/ingestion/
├── __init__.py                 # Public API exports and unified module imports
├── pipeline.py                 # Core orchestration function: ingest_model()
├── model_registry.py           # Model presets, Kaggle/HF resolvers, environment setup & logging
├── exceptions.py               # Ingestion-specific exception hierarchy
├── loader/                     # Model loading implementations
│   ├── __init__.py
│   ├── base_loader.py          # BaseModelLoader abstract base class
│   ├── pytorch_loader.py       # PyTorchLoader (.pt, .pth, state_dict, ScriptModule)
│   └── huggingface_loader.py   # HuggingFaceLoader (Hub repositories, AutoModel fallbacks)
├── adapters/                   # Standardized wrapper adapters
│   ├── __init__.py
│   ├── base_adapter.py         # BaseModelAdapter abstract base class
│   └── pytorch_adapter.py      # PyTorchAdapter (predict, to, eval, train, __call__)
├── runtime/                    # Hardware execution and device routing
│   ├── __init__.py
│   └── device_manager.py       # DeviceManager (CUDA, MPS, CPU detection & allocation)
├── validation/                 # Integrity and sanity verification
│   ├── __init__.py
│   └── validator.py            # ModelValidator (interface integrity, NaN/Inf checks)
├── metadata/                   # Metadata representation
│   ├── __init__.py
│   └── model_metadata.py       # ModelMetadata dataclass (shapes, parameters, domain, task)
└── utils/                      # Ingestion utilities and logging
    ├── __init__.py
    └── logger.py               # log_ingestion_event() structured JSON audit logging
```

---

## 3. What Each Component Does

### 3.1. Loaders (`app/ingestion/loader/`)

The loader subsystem is responsible for retrieving and unpacking raw models and weights into memory.

* **`BaseModelLoader`** (`base_loader.py`):
  * Abstract base class defining the contract: `load(model_path, **kwargs) -> Any`.
* **`PyTorchLoader`** (`pytorch_loader.py`):
  * Loads native PyTorch models and serialized checkpoints.
  * **Supported Inputs**:
    1. Direct `nn.Module` or `torch.jit.ScriptModule` instances in memory.
    2. TorchScript exported archives via `torch.jit.load()`.
    3. Serialized checkpoint files (`.pt`, `.pth`), handling PyTorch 2.6+ `weights_only` defaults.
    4. PyTorch state dictionaries (`state_dict`), reconstructing weights into a provided `model_class` factory or blueprint.
  * Automatically sets the loaded model to `eval()` mode.
* **`HuggingFaceLoader`** (`huggingface_loader.py`):
  * Fetches pretrained models from Hugging Face Hub or local directory caches.
  * Handles domain-specific model heads (`AutoModelForImageClassification`, `AutoModelForSequenceClassification`, `AutoModel`).
  * Patches Hub configurations (e.g., `id2label` mapping normalization for Vision Transformers).

---

### 3.2. Adapters (`app/ingestion/adapters/`)

The adapter subsystem wraps framework-specific models in a uniform execution harness.

* **`BaseModelAdapter`** (`base_adapter.py`):
  * Defines standard methods required by AdverScan engines:
    * `predict(inputs)`: Generates predictions / logits.
    * `get_model()`: Exposes the underlying framework model when gradient access is required.
    * `to(device)`: Transfers the model across hardware devices.
    * `eval()` / `train(mode)`: Toggles training/inference state.
    * `__call__(*args, **kwargs)`: Callable syntax allowing direct model execution.
* **`PyTorchAdapter`** (`pytorch_adapter.py`):
  * Concrete implementation wrapping `torch.nn.Module`.
  * **Input Preprocessing (`_prepare_tensor_input`)**: Accepts `torch.Tensor`, `numpy.ndarray`, or Python `list`/`tuple` and standardizes them to a PyTorch tensor on the target device.
  * **Inference (`predict`)**: Executes `torch.no_grad()` inference and supports optional output formatting (`return_numpy=True`).
  * **Gradient Preservation (`__call__`)**: Passes inputs through the model while preserving computation graphs, enabling white-box adversarial attacks (e.g., FGSM, PGD) to compute input gradients.

---

### 3.3. Runtime Management (`app/ingestion/runtime/device_manager.py`)

* **`DeviceManager`**:
  * Resolves hardware acceleration targets:
    * `cuda` / `cuda:N`: Checks `torch.cuda.is_available()`, falling back to CPU if unavailable.
    * `mps`: Checks Apple Silicon MPS availability, falling back to CPU if unavailable.
    * `cpu`: Standard CPU execution fallback.
  * Provides `to_device(obj, device)` for relocating tensors or models safely.

---

### 3.4. Validation Engine (`app/ingestion/validation/validator.py`)

* **`ModelValidator`**:
  * Enforces model health before security scans begin.
  * Verifies the model adapter inherits from `BaseModelAdapter`.
  * Verifies the underlying neural network exists (`get_model() is not None`).
  * If a `sample_input` is supplied:
    * Executes a forward inference pass (`adapter.predict(sample_input)`).
    * Validates that outputs are non-empty.
    * Scans output tensors/arrays for numerical corruption (`NaN` or `Inf` values).
  * Raises descriptive `ModelValidationError` exceptions upon failure.

---

### 3.5. Metadata Representation (`app/ingestion/metadata/model_metadata.py`)

* **`ModelMetadata`**:
  * Standardized dataclass capturing essential specifications:
    * `framework`: Model source framework (e.g., `"pytorch"`).
    * `model_name`: Descriptive identifier.
    * `input_shape`: Dimensions expected by the network (e.g., `(1, 3, 224, 224)`).
    * `output_shape`: Prediction output dimensions (e.g., `(1, 43)`).
    * `num_classes`: Number of target output classes inferred from shape.
    * `task_type`: Task nature (e.g., `"classification"`).
    * `device`: Allocated hardware device (e.g., `"cuda:0"`, `"cpu"`).
    * `domain`: Application domain (`"image"`, `"text"`, `"time_series"`, `"tabular"`).
    * `total_parameters` & `trainable_parameters`: Parameter counts.
    * `model_architecture`: Class name of the underlying model.
    * `extra_info`: Extensible key-value metadata dictionary.
  * Offers `.to_dict()` for JSON serialization and audit logging.

---

### 3.6. Model Registry & Environment Setup (`app/ingestion/model_registry.py`)

* **`MODEL_PRESETS`**:
  * Curated catalog of benchmark models (GTSRB ViT, Intel Image Classification ViT, CIFAR-10 ViT, ImageNet-1k ViT).
* **`DATASET_TO_RECOMMENDED_MODEL`**:
  * Disambiguates Kaggle URLs and dataset identifiers from model checkpoints (e.g., resolves Kaggle Intel dataset URL to the corresponding Hugging Face model).
* **`ModelPresetRegistry`**:
  * Resolves arbitrary inputs (preset IDs, Hugging Face Hub IDs, Kaggle links, local file paths).
  * Inspects model architecture configurations dynamically to distinguish Computer Vision vs NLP/Text models.
* **`setup_model_environment()`**:
  * Resolves target execution device, dataset name, preprocessor/tokenizer name, expected resolution, and expected number of classes.
* **`log_ingestion_event()`**:
  * Appends structured, timestamped audit events to `results/ingestion_log.json`.

---

### 3.7. Exception Hierarchy (`app/ingestion/exceptions.py`)

```
ModelIngestionError (Base Exception)
├── ModelLoadError        # Triggered when model file is missing, corrupt, or fails to deserialize
├── ModelValidationError # Triggered when adapter is invalid, forward pass fails, or outputs contain NaN/Inf
└── UnsupportedModelError# Triggered when an unrecognized framework or payload type is provided
```

---

## 4. End-to-End Workflow & Step-by-Step Execution

When `ingest_model()` or the CLI/script runner executes, data flows through five sequential phases:

```
[User / CLI / Script Input]
      │ (model_path, sample_input, device, model_class, domain)
      ▼
┌───────────────────────────────────────────────────────────────┐
│ Phase 1: Hardware & Device Detection                          │
│   DeviceManager.get_device(preferred_device)                  │
│   --> Resolves CUDA / MPS / CPU with graceful fallback        │
└───────────────────────────────┬───────────────────────────────┘
                                │ target_device
                                ▼
┌───────────────────────────────────────────────────────────────┐
│ Phase 2: Model Loading & Deserialization                      │
│   PyTorchLoader.load() / HuggingFaceLoader.load()             │
│   --> Unpacks file, TorchScript, state_dict, or HF Hub weights│
│   --> Applies model.eval()                                    │
└───────────────────────────────┬───────────────────────────────┘
                                │ raw_model (nn.Module)
                                ▼
┌───────────────────────────────────────────────────────────────┐
│ Phase 3: Standardized Adaptation                              │
│   PyTorchAdapter(raw_model, device=target_device)             │
│   --> Wraps model with uniform predict(), eval(), __call__()  │
│   --> Moves model parameters to target_device                 │
└───────────────────────────────┬───────────────────────────────┘
                                │ adapter (PyTorchAdapter)
                                ▼
┌───────────────────────────────────────────────────────────────┐
│ Phase 4: Model Verification & Health Sanity Check             │
│   ModelValidator.validate(adapter, sample_input)              │
│   --> Tests BaseModelAdapter compliance                       │
│   --> Executes forward inference pass on sample_input         │
│   --> Checks outputs for NaN / Inf corruption                 │
└───────────────────────────────┬───────────────────────────────┘
                                │ validation passed
                                ▼
┌───────────────────────────────────────────────────────────────┐
│ Phase 5: Shape Inference, Metadata Extraction & Audit Logging │
│   ModelMetadata(...)                                          │
│   --> Extracts input/output shapes & class count              │
│   --> Tallies total & trainable parameter counts              │
│   --> Optionally records entry via log_ingestion_event()      │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
                  Returns: (adapter, metadata)
```

---

## 5. Summary Table: Component Inter-relationships

| Component | File Path | Input | Output | Primary Collaborators |
| :--- | :--- | :--- | :--- | :--- |
| **`ingest_model`** | `pipeline.py` | Model path/instance, sample input, config | `(PyTorchAdapter, ModelMetadata)` | `DeviceManager`, `PyTorchLoader`, `PyTorchAdapter`, `ModelValidator`, `ModelMetadata` |
| **`PyTorchLoader`** | `loader/pytorch_loader.py` | Path (`.pt`, `.pth`), state_dict, or module | `torch.nn.Module` (in `eval` mode) | `BaseModelLoader`, `ModelLoadError` |
| **`HuggingFaceLoader`**| `loader/huggingface_loader.py`| HF Repo ID / directory path | `torch.nn.Module` (in `eval` mode) | `BaseModelLoader`, `patch_hf_config` |
| **`PyTorchAdapter`** | `adapters/pytorch_adapter.py` | `torch.nn.Module`, target device | Standardized adapter instance | `BaseModelAdapter`, downstream attack engines |
| **`DeviceManager`** | `runtime/device_manager.py` | Preferred device string/object | `torch.device` | `torch.cuda`, `torch.backends.mps` |
| **`ModelValidator`** | `validation/validator.py` | `BaseModelAdapter`, sample input | `bool` (or raises error) | `BaseModelAdapter`, `ModelValidationError` |
| **`ModelMetadata`** | `metadata/model_metadata.py`| Architecture, shape, and count specs | `ModelMetadata` object & dict | `pipeline.py`, reporting/logging |
| **`ModelPresetRegistry`** | `model_registry.py` | Model ID, Kaggle URL, or path | Metadata dict & resolved IDs | `setup_model_environment()`, CLI scripts |
| **`log_ingestion_event`** | `utils/logger.py` | Event dictionary, file destination | Log file path string | `results/ingestion_log.json` |

---

## 6. Usage Examples

### Example 1: Direct Ingestion via Public API

```python
import torch
from app.ingestion import ingest_model

# 1. Define sample input for shape inference and forward validation
sample_input = torch.randn(1, 3, 224, 224)

# 2. Ingest a local PyTorch checkpoint or in-memory model
adapter, metadata = ingest_model(
    model_path="models/gtsrb_classifier.pt",
    sample_input=sample_input,
    device="cuda",  # Gracefully falls back to CPU if CUDA unavailable
    model_name="GTSRB-Classifier",
    task_type="classification",
    domain="image",
    log_ingestion=True,
)

# 3. Access standardized capabilities
print(f"Model: {metadata.model_name}")
print(f"Total Parameters: {metadata.total_parameters:,}")
print(f"Inferred Classes: {metadata.num_classes}")

# 4. Perform standardized prediction
predictions = adapter.predict(sample_input, return_numpy=True)
```

### Example 2: Interactive Registry & Environment Configuration

```python
from app.ingestion.model_registry import ModelPresetRegistry, setup_model_environment

# Resolve presets or custom Hugging Face models
model_info = ModelPresetRegistry.resolve_model_info("bazyl/gtsrb-model")

# Configure runtime execution environment and log event
env_config = setup_model_environment(
    model_id_or_path="bazyl/gtsrb-model",
    device="cuda",
    log_log_event=True,
    log_filepath="results/ingestion_log.json",
)

print(f"Configured device: {env_config['target_device']}")
print(f"Recommended dataset: {env_config['recommended_dataset']}")
```
