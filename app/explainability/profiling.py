"""
Profiling and query counting utilities for AdverScan XAI module (Module 5).
Measures explanation runtime (explain_seconds) and target model queries (num_model_queries).
"""

from dataclasses import asdict, dataclass
import functools
import time
import types
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from app.explainability.utils import unwrap_model


@dataclass
class ProfileStats:
    """
    Dataclass holding execution profiling metrics for an XAI explanation.
    """

    explain_seconds: float = 0.0
    num_model_queries: int = 0
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert ProfileStats to standard dictionary."""
        return asdict(self)


class QueryCounter:
    """
    Model query counter and instrumentation context.
    Tracks forward inferences across PyTorch nn.Modules, scikit-learn estimators,
    PyTorchAdapters, and generic callable models while preventing double counting via reentrancy guards.
    """

    def __init__(self, model: Any):
        """
        Initialize QueryCounter for the given target model.

        Args:
            model: PyTorch module, scikit-learn estimator, PyTorchAdapter, or callable.
        """
        self.raw_model = model
        self.query_count: int = 0
        self.sample_count: int = 0
        self._in_call: bool = False
        self._active: bool = False
        self._restorations: List[Tuple[Any, str, Any]] = []
        self.instrumented_model: Any = model

    def reset(self) -> None:
        """Reset query and sample counts to zero."""
        self.query_count = 0
        self.sample_count = 0
        self._in_call = False

    def record_call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute model function while safely counting queries and samples if not in a nested call.
        """
        if not self._active:
            return fn(*args, **kwargs)

        if not self._in_call:
            self._in_call = True
            try:
                self.query_count += 1
                if args:
                    first_arg = args[0]
                    try:
                        if hasattr(first_arg, "shape") and len(first_arg.shape) > 0:
                            self.sample_count += int(first_arg.shape[0])
                        elif isinstance(first_arg, (list, tuple)):
                            self.sample_count += len(first_arg)
                        else:
                            self.sample_count += 1
                    except Exception:
                        self.sample_count += 1
                else:
                    self.sample_count += 1
                return fn(*args, **kwargs)
            finally:
                self._in_call = False
        else:
            return fn(*args, **kwargs)

    def _instrument_object(self, obj: Any) -> None:
        """Attach method interception to target object or module."""
        if obj is None:
            return

        # 1. PyTorch nn.Module (top-level forward pass)
        if isinstance(obj, (nn.Module, getattr(torch.jit, "ScriptModule", nn.Module))):
            if hasattr(obj, "forward") and callable(obj.forward):
                orig_forward = obj.forward

                @functools.wraps(orig_forward)
                def profiled_forward(*args: Any, _orig: Callable[..., Any] = orig_forward, **kwargs: Any) -> Any:
                    return self.record_call(_orig, *args, **kwargs)

                try:
                    obj.forward = profiled_forward
                    self._restorations.append((obj, "forward", orig_forward))
                except Exception:
                    pass
            return

        # 2. Scikit-learn models, Adapters, or objects with predict / predict_proba / decision_function
        intercept_methods = ("predict_proba", "predict", "decision_function", "__call__")
        for method_name in intercept_methods:
            if hasattr(obj, method_name):
                try:
                    orig_method = getattr(obj, method_name)
                    if callable(orig_method) and not isinstance(orig_method, types.BuiltinFunctionType):
                        # Avoid double-wrapping
                        if getattr(orig_method, "__query_profiled__", False):
                            continue

                        def make_profiled_method(fn: Callable[..., Any]) -> Callable[..., Any]:
                            @functools.wraps(fn)
                            def profiled_method(*args: Any, **kwargs: Any) -> Any:
                                return self.record_call(fn, *args, **kwargs)

                            setattr(profiled_method, "__query_profiled__", True)
                            return profiled_method

                        profiled_fn = make_profiled_method(orig_method)
                        setattr(obj, method_name, profiled_fn)
                        self._restorations.append((obj, method_name, orig_method))
                except Exception:
                    continue

    def start(self) -> Any:
        """
        Activate query counting and instrument model instances.

        Returns:
            The model instance to pass to the explainer.
        """
        self._active = True
        self.reset()

        # Collect nested models (e.g. PyTorchAdapter -> nn.Module)
        models_to_instrument: List[Any] = []
        curr = self.raw_model
        visited: Set[int] = set()

        while curr is not None and id(curr) not in visited:
            visited.add(id(curr))
            models_to_instrument.append(curr)
            if hasattr(curr, "get_model") and callable(curr.get_model):
                try:
                    curr = curr.get_model()
                except Exception:
                    break
            elif hasattr(curr, "_model") and curr._model is not None and curr._model is not curr:
                curr = curr._model
            elif hasattr(curr, "model") and curr.model is not None and curr.model is not curr:
                curr = curr.model
            else:
                break

        # Instrument from inner underlying model to outer adapter
        for m in reversed(models_to_instrument):
            self._instrument_object(m)

        # Handle pure function callables that cannot have instance attributes set
        if (
            callable(self.raw_model)
            and not isinstance(self.raw_model, (nn.Module, type))
            and not self._restorations
        ):
            self.instrumented_model = _CallableQueryWrapper(self.raw_model, self)
        else:
            self.instrumented_model = self.raw_model

        return self.instrumented_model

    def stop(self) -> None:
        """Deactivate query counting and restore original model methods."""
        self._active = False
        self._in_call = False

        for obj, attr_name, orig_val in reversed(self._restorations):
            try:
                setattr(obj, attr_name, orig_val)
            except Exception:
                pass
        self._restorations.clear()

    def __enter__(self) -> "QueryCounter":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


class _CallableQueryWrapper:
    """Wrapper for pure functions or immutable callables."""

    def __init__(self, fn: Callable[..., Any], counter: QueryCounter):
        self._fn = fn
        self._counter = counter
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._counter.record_call(self._fn, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fn, name)


class ProfileContext:
    """
    Context manager measuring both execution time and model query count.
    """

    def __init__(self, model: Any):
        """
        Initialize ProfileContext.

        Args:
            model: Target model or prediction callable to monitor.
        """
        self.counter = QueryCounter(model)
        self.explain_seconds: float = 0.0
        self._start_time: Optional[float] = None
        self.model: Any = model

    @property
    def num_model_queries(self) -> int:
        """Number of model inference queries captured."""
        return self.counter.query_count

    @property
    def num_samples(self) -> int:
        """Total sample inputs evaluated by the model."""
        return self.counter.sample_count

    def record_duration(self, seconds: float) -> None:
        """Record total duration in seconds."""
        self.explain_seconds = float(seconds)

    def get_stats(self) -> ProfileStats:
        """Return gathered ProfileStats."""
        return ProfileStats(
            explain_seconds=self.explain_seconds,
            num_model_queries=self.num_model_queries,
            num_samples=self.num_samples,
        )

    def __enter__(self) -> "ProfileContext":
        self.model = self.counter.start()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._start_time is not None:
            self.explain_seconds = time.perf_counter() - self._start_time
        self.counter.stop()


def profile_explainer_call(
    explainer: Any,
    model: Any,
    *args: Any,
    **kwargs: Any,
) -> Tuple[Dict[str, Any], ProfileStats]:
    """
    Wrap an explainer.explain(model, ...) call with execution timing and model query counting.

    Args:
        explainer: BaseExplainer or object implementing an explain(...) method.
        model: Target ML model or callable.
        *args: Positional arguments forwarded to explainer.explain.
        **kwargs: Keyword arguments forwarded to explainer.explain.

    Returns:
        Tuple of (result_dict, ProfileStats) where result_dict is the attribution dictionary
        and ProfileStats contains explain_seconds, num_model_queries, and num_samples.
    """
    with ProfileContext(model) as profiler:
        t0 = time.perf_counter()
        try:
            result = explainer.explain(profiler.model, *args, **kwargs)
        except Exception:
            elapsed = time.perf_counter() - t0
            profiler.record_duration(elapsed)
            raise

        elapsed = time.perf_counter() - t0
        profiler.record_duration(elapsed)
        stats = profiler.get_stats()

        if isinstance(result, dict):
            result["explain_seconds"] = stats.explain_seconds
            result["num_model_queries"] = stats.num_model_queries

        return result, stats
