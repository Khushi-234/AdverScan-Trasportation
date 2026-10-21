"""
Scikit-learn model loader implementation with secure deserialization boundaries.
"""

from pathlib import Path
from typing import Any, List, Optional, Union
import pickle

from app.ingestion.exceptions import ModelLoadError, UnsafeModelError
from app.ingestion.loader.base_loader import BaseModelLoader, LoadResult
from app.ingestion.utils.logger import log_ingestion_event


class SklearnLoader(BaseModelLoader):
    """
    Loads scikit-learn models from in-memory estimators or serialized files (.joblib, .pkl, .pickle).
    Enforces security boundaries against arbitrary deserialization attacks.
    """

    def load(
        self,
        model_path: Union[str, Path, Any],
        unsafe_allow_pickle: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Load a scikit-learn model.

        Args:
            model_path: Path to serialized estimator file or an instantiated estimator instance.
            unsafe_allow_pickle: Explicit user consent flag required to deserialize pickle/joblib files.
            **kwargs: Extra loading options.

        Returns:
            The loaded scikit-learn estimator instance.

        Raises:
            UnsafeModelError: If model_path is a pickle/joblib file and unsafe_allow_pickle is False.
            ModelLoadError: If file not found or deserialized object is not a valid estimator.
        """
        # Case 1: In-memory model instance
        if not isinstance(model_path, (str, Path)):
            if hasattr(model_path, "predict") and (hasattr(model_path, "fit") or hasattr(model_path, "classes_")):
                return model_path
            raise ModelLoadError(
                f"Provided object of type {type(model_path)} is not a valid scikit-learn estimator."
            )

        path = Path(model_path)
        if not path.is_file():
            # If not a local file, check if it is a Hugging Face repository
            path_str = str(model_path).strip()
            if "/" in path_str and not path_str.startswith((".", "/")):
                try:
                    from huggingface_hub import hf_hub_download
                    local_f = None
                    for candidate in ("model.joblib", "model.pkl", "model.pickle"):
                        try:
                            local_f = hf_hub_download(path_str, candidate)
                            break
                        except Exception:
                            continue
                    if local_f:
                        path = Path(local_f)
                except Exception:
                    pass

        if not path.is_file():
            raise ModelLoadError(f"Model file not found at path: {path}")

        # Security check: Arbitrary deserialization risk
        if not unsafe_allow_pickle:
            raise UnsafeModelError(
                f"Security risk: Refusing to deserialize model from '{path}' without explicit user consent. "
                f"Loading pickled or joblib files from untrusted sources can lead to arbitrary code execution. "
                f"To load this model, pass `unsafe_allow_pickle=True` after verifying the file source."
            )

        log_ingestion_event(
            {
                "status": "SECURITY_WARNING",
                "message": f"Loading model via unsafe deserialization from {path}.",
                "path": str(path),
                "unsafe_allow_pickle": True,
            }
        )

        try:
            # Try joblib if available
            try:
                import joblib
                loaded_obj = joblib.load(str(path))
            except ImportError:
                with open(path, "rb") as f:
                    loaded_obj = pickle.load(f)

            if not (hasattr(loaded_obj, "predict") and (hasattr(loaded_obj, "fit") or hasattr(loaded_obj, "classes_"))):
                raise ModelLoadError(
                    f"Deserialized object from '{path}' is of type {type(loaded_obj)}, "
                    f"which is not a valid scikit-learn estimator with predict()."
                )

            return loaded_obj

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load scikit-learn model from {path}: {str(e)}") from e

    def inspect_and_load(
        self,
        model_path: Union[str, Path, Any],
        sample_input: Optional[Any] = None,
        unsafe_allow_pickle: bool = False,
        **kwargs: Any,
    ) -> LoadResult:
        """
        Load model and generate structured inspection evidence.
        """
        model = self.load(model_path, unsafe_allow_pickle=unsafe_allow_pickle, **kwargs)
        source = "in_memory" if not isinstance(model_path, (str, Path)) else "local"

        evidence: List[str] = [
            f"Detected scikit-learn estimator ({model.__class__.__name__})",
            "Model exposes predict()",
        ]
        if hasattr(model, "predict_proba"):
            evidence.append("Model exposes predict_proba()")
        if hasattr(model, "decision_function"):
            evidence.append("Model exposes decision_function()")
        if hasattr(model, "n_features_in_"):
            evidence.append(f"Model fitted on n_features_in_ = {model.n_features_in_}")
        if hasattr(model, "classes_"):
            evidence.append(f"Classes detected: {len(model.classes_)}")

        evidence.append("Domain inferred as tabular from estimator interface")
        evidence.append("Preprocessing metadata unavailable; assuming inputs match model space unless configured")

        return LoadResult(
            model=model,
            source=source,
            framework="sklearn",
            evidence=evidence,
            domain_hint="tabular",
            task_hint="classification",
        )
