"""
PyTorch model loader implementation.
"""

from pathlib import Path
from typing import Any, Optional, Union
import torch
import torch.nn as nn

from app.ingestion.exceptions import ModelLoadError
from app.ingestion.loader.base_loader import BaseModelLoader


class PyTorchLoader(BaseModelLoader):
    """
    Loads PyTorch models from file paths (pickled models, state dicts, TorchScript)
    or validates existing PyTorch model instances.
    """

    def load(
        self,
        model_path: Union[str, Path, nn.Module],
        model_class: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any
    ) -> nn.Module:
        """
        Load a PyTorch model.

        Args:
            model_path: File path to model checkpoint (.pt, .pth), TorchScript model,
                        or an existing nn.Module instance.
            model_class: Model class or instantiated model structure when loading state_dict.
            device: Target device to map stored tensors onto during load.
            **kwargs: Extra parameters passed to torch.load.

        Returns:
            nn.Module: Loaded PyTorch model set in evaluation mode.
        """
        map_location = device or "cpu"

        # Case 1: model_path is already a PyTorch nn.Module or ScriptModule instance
        if isinstance(model_path, (nn.Module, torch.jit.ScriptModule)):
            model_path.eval()
            return model_path

        # Validate file existence for path inputs
        path = Path(model_path)
        if not path.is_file():
            raise ModelLoadError(f"Model file not found at path: {path}")

        try:
            # First try loading via TorchScript if file indicates TorchScript format
            try:
                model = torch.jit.load(str(path), map_location=map_location)
                model.eval()
                return model
            except Exception:
                pass  # Fallback to standard torch.load

            # Load using torch.load handling PyTorch 2.6+ weights_only behavior
            if "weights_only" not in kwargs:
                try:
                    loaded_obj = torch.load(str(path), map_location=map_location, weights_only=False, **kwargs)
                except TypeError:
                    loaded_obj = torch.load(str(path), map_location=map_location, **kwargs)
            else:
                loaded_obj = torch.load(str(path), map_location=map_location, **kwargs)

            # Case 2: loaded_obj is an nn.Module instance
            if isinstance(loaded_obj, nn.Module):
                loaded_obj.eval()
                return loaded_obj

            # Case 3: loaded_obj is a state_dict (dictionary of weights)
            if isinstance(loaded_obj, dict):
                if model_class is None:
                    raise ModelLoadError(
                        f"Loaded state_dict from {path}, but no model_class or model structure was provided."
                    )
                
                # Instantiate model if model_class is a class, or use instance if already instantiated
                model = model_class() if isinstance(model_class, type) else model_class
                if not isinstance(model, nn.Module):
                    raise ModelLoadError("Provided model_class is not a torch.nn.Module instance or factory.")
                
                # Check for nested state_dict keys (e.g., 'state_dict' or 'model_state_dict')
                state_dict = loaded_obj
                for key in ["state_dict", "model_state_dict", "model"]:
                    if key in state_dict and isinstance(state_dict[key], dict):
                        state_dict = state_dict[key]
                        break

                model.load_state_dict(state_dict)
                model.eval()
                return model

            raise ModelLoadError(
                f"Unsupported model payload type loaded from {path}: {type(loaded_obj)}"
            )

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load PyTorch model from {path}: {str(e)}") from e
