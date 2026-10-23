Ran command: `git log -n 5 --oneline`
Ran command: `git log -p -n 1 cf88368 -- app/ingestion/`
Ran command: `git diff cf88368^! -- app/ingestion/model_registry.py`
Viewed run_baseline_evaluation.py:175-210
Ran command: `git status`
Ran command: `git log -n 3 --oneline`
Ran command: `.venv/bin/python -c '
from app.ingestion.utils.model_utils import detect_model_source, detect_model_framework

for model_id in ["bhadresh-savani/distilbert-base-uncased-emotion", "Gladiator/microsoft-deberta-v3-large_cls_subj"]:
    src = detect_model_source(model_id)
    fw = detect_model_framework(model_id)
    print(f"{model_id} -> source: {src}, fw: {fw}")
'`
Ran command: `.venv/bin/python -c '
from app.ingestion.model_registry import ModelPresetRegistry
from app.ingestion.utils.model_utils import detect_model_source, detect_model_framework
from app.ingestion.pipeline import ingest_model

# Monkey patch build_runtime_config to verify the fix
original_build = ModelPresetRegistry.build_runtime_config

def fixed_build_runtime_config(cls, model_id, **kwargs):
    cfg = original_build(model_id, **kwargs)
    clean_id = str(model_id).strip()
    if not cls.has_preset(clean_id):
        src = detect_model_source(clean_id)
        fw = detect_model_framework(clean_id)
        if src == "huggingface" and fw != "sklearn":
            cfg["framework"] = "huggingface"
    return cfg

ModelPresetRegistry.build_runtime_config = classmethod(fixed_build_runtime_config)

print("Testing bhadresh-savani/distilbert-base-uncased-emotion...")
adapter, meta = ingest_model(
    model_path="bhadresh-savani/distilbert-base-uncased-emotion",
    sample_input="Testing sentence",
    device="cpu",
)
print("Success! Adapter:", type(adapter), "Framework:", meta.framework)
'`
Ran command: `.venv/bin/python -c '
from app.ingestion.model_registry import ModelPresetRegistry
from app.ingestion.utils.model_utils import detect_model_source, detect_model_framework
from app.ingestion.pipeline import ingest_model

original_build = ModelPresetRegistry.build_runtime_config

def fixed_build_runtime_config(cls, model_id, **kwargs):
    cfg = original_build(model_id, **kwargs)
    clean_id = str(model_id).strip()
    if not cls.has_preset(clean_id):
        src = detect_model_source(clean_id)
        fw = detect_model_framework(clean_id)
        if src == "huggingface" and fw != "sklearn":
            cfg["framework"] = "huggingface"
    return cfg

ModelPresetRegistry.build_runtime_config = classmethod(fixed_build_runtime_config)

print("Testing Gladiator/microsoft-deberta-v3-large_cls_subj...")
adapter, meta = ingest_model(
    model_path="Gladiator/microsoft-deberta-v3-large_cls_subj",
    sample_input="Testing sentence",
    device="cpu",
)
print("Success! Adapter:", type(adapter), "Framework:", meta.framework)
'`

I will wait for the test to complete.

Used tool: manage_task
Used tool: manage_task
Ran command: `.venv/bin/pytest tests/ingestion/`

I am waiting for the ingestion tests to complete.

Used tool: manage_task
Edited model_registry.py
Viewed run_baseline_evaluation.py:180-210
Edited run_baseline_evaluation.py
Ran command: `ls -la scripts/`
Ran command: `printf "15\nbhadresh-savani/distilbert-base-uncased-emotion\n\ntext\ndair-ai/emotion\n" | .venv/bin/python scripts/run_baseline_evaluation.py`
Used tool: manage_task
Ran command: `git diff --stat`
Ran command: `git diff`
Ran command: `.venv/bin/pytest tests/ingestion/test_model_registry.py tests/ingestion/test_custom_and_registry_models.py`

### Why It Worked Before vs. Why It Broke

In earlier versions of the codebase, dynamic model ingestion from the terminal worked because [ModelPresetRegistry.build_runtime_config](file:///home/bisagn/Documents/AdverScan/adverscan-framework/AdverScan/app/ingestion/model_registry.py#L53) used this rule:
```python
# Older code:
resolved_framework = framework or ("pytorch_local" if is_local_file else "huggingface")
```
If the input path was not a local `.pt` file on disk (`is_local_file == False`), it **automatically resolved to `"huggingface"`**.

---

### The Regression (Commit `cf88368`: "Dynamic model adaption for tabular and TS domain")

When tabular/time-series model support was added, framework resolution in [ModelPresetRegistry.build_runtime_config](file:///home/bisagn/Documents/AdverScan/adverscan-framework/AdverScan/app/ingestion/model_registry.py#L93-L107) was changed to:

```python
detected_fw = detect_model_framework(clean_id)
if framework:
    resolved_framework = framework
elif is_local_file:
    resolved_framework = "sklearn" if p.suffix.lower() in (".joblib", ".pkl", ".pickle") else "pytorch_local"
elif detected_fw in ("sklearn", "pytorch"):
    resolved_framework = detected_fw
elif resolved_domain == "tabular":
    resolved_framework = "sklearn"
else:
    resolved_framework = "huggingface"
```

#### What went wrong:
1. When you entered a Hugging Face repo ID dynamically (such as `bhadresh-savani/distilbert-base-uncased-emotion` or `Gladiator/microsoft-deberta-v3-large_cls_subj`), [detect_model_framework](file:///home/bisagn/Documents/AdverScan/adverscan-framework/AdverScan/app/ingestion/utils/model_utils.py#L145-L196) was called.
2. [detect_model_framework](file:///home/bisagn/Documents/AdverScan/adverscan-framework/AdverScan/app/ingestion/utils/model_utils.py#L145-L196) explicitly states in its docstring:
   > *"Determine model framework ('sklearn', 'pytorch', 'tensorflow'). Never conflates source ('huggingface') with execution framework."*
   
   Because it considers PyTorch the backend execution engine, its fallback return is:
   ```python
   return "pytorch"
   ```
3. Therefore, `detected_fw` returned `"pytorch"`.
4. Then `elif detected_fw in ("sklearn", "pytorch"):` was **always satisfied**, locking `resolved_framework = "pytorch"`.
5. The `else: resolved_framework = "huggingface"` branch became **unreachable dead code**.
6. Back in [ingest_model](file:///home/bisagn/Documents/AdverScan/adverscan-framework/AdverScan/app/ingestion/pipeline.py#L234-L241):
   ```python
   elif framework_clean in ("pytorch", "pytorch_local"):
       loader = PyTorchLoader()
       raw_model = loader.load(target_model_path, ...)
   ```
   It selected [PyTorchLoader](file:///home/bisagn/Documents/AdverScan/adverscan-framework/AdverScan/app/ingestion/loader/pytorch_loader.py#L50), which checked `if not Path(path).is_file()` and raised:
   ```
   ModelLoadError: Model file not found at path: <model_id>
   ```

---

### What Was Fixed

1. In [ModelPresetRegistry.build_runtime_config](file:///home/bisagn/Documents/AdverScan/adverscan-framework/AdverScan/app/ingestion/model_registry.py#L95-L106), added a check for [detect_model_source](file:///home/bisagn/Documents/AdverScan/adverscan-framework/AdverScan/app/ingestion/utils/model_utils.py#L125-L142) before testing `detected_fw in ("sklearn", "pytorch")`:
   ```python
   detected_source = detect_model_source(clean_id)
   detected_fw = detect_model_framework(clean_id)
   if framework:
       resolved_framework = framework
   elif is_local_file:
       resolved_framework = "sklearn" if p.suffix.lower() in (".joblib", ".pkl", ".pickle") else "pytorch_local"
   elif detected_source == "huggingface":
       resolved_framework = "sklearn" if detected_fw == "sklearn" else "huggingface"
   elif detected_fw in ("sklearn", "pytorch"):
       resolved_framework = detected_fw
   ...
   ```

2. In [scripts/run_baseline_evaluation.py](file:///home/bisagn/Documents/AdverScan/adverscan-framework/AdverScan/scripts/run_baseline_evaluation.py#L186-L205), forwarded `framework=env_info.get("framework")` into `ingest_model` so the framework resolved during environment setup is explicitly maintained.

---

### Verification

Tested dynamic terminal ingestion of `bhadresh-savani/distilbert-base-uncased-emotion` directly with [scripts/run_baseline_evaluation.py](file:///home/bisagn/Documents/AdverScan/adverscan-framework/AdverScan/scripts/run_baseline_evaluation.py) without adding it to presets:

```
[1/3] Standardizing & Setting Environment for 'bhadresh-savani/distilbert-base-uncased-emotion'...
  ✓ Target Data Domain: [TEXT]
  ✓ Model Architecture: Transformer NLP (distilbert)
  ✓ Dataset Recommended: dair-ai/emotion
  ✓ Preprocessor Config: bhadresh-savani/distilbert-base-uncased-emotion
  ✓ Execution Device: cpu
• Model Architecture:   DistilBertForSequenceClassification
• Total Parameters:     66,958,086 (~66.96M)
• Trainable Parameters: 66,958,086 (100.0%)

[2/3] Running Clean Baseline Evaluation on 'dair-ai/emotion' (test split, 100 samples)...
======================================================================
              ADVERSCAN — BASELINE EVALUATION SUMMARY             
======================================================================
  CLEAN PERFORMANCE BASELINE
  • Clean Overall Accuracy:  94.00% (0.9400)
  • Macro F1-Score:          90.29% (0.9029)
  • Weighted F1-Score:       93.90% (0.9390)
  • Average Confidence:      96.56%
======================================================================
```

You can now run any dynamic Hugging Face Hub model directly from the terminal without registering it in `MODEL_PRESETS`.