"""
Carlini & Wagner (C&W) L2 optimization-based adversarial attack implementation.
"""

from typing import Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.models import AttackMetadata
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackExecutionError, AttackConfigurationError
from app.attack_engine.attack_registry import register_attack
from app.attack_engine.utils.tensor_utils import get_device, clip_tensor
from app.attack_engine.utils.validation import validate_inputs, validate_attack_config


class CW(BaseAttack):
    """
    Carlini & Wagner (C&W) L2 optimization attack.

    Formulates adversarial example generation as minimizing:
        ||x_adv - x||_2^2 + c * f(x_adv)
    subject to box constraints in [clip_min, clip_max] via a tanh change-of-variables.

    Note on constraints:
        Unlike PGD or FGSM which enforce a hard epsilon-ball constraint (e.g. Linf or L2),
        C&W directly optimizes the L2 perturbation magnitude via a Lagrangian multiplier c.
        Any global 'epsilon' in AttackConfig is not enforced as a hard bound; C&W minimizes
        L2 distance directly to achieve the desired confidence margin.
    """

    metadata = AttackMetadata(
        name="cw",
        domain="image",
        category="optimization",
        attack_type="white_box",
        requires_gradient=True,
        requires_loss=True,
        requires_model_weights=True,
        iterative=True,
        targeted_supported=True,
        untargeted_supported=True,
        query_based=False,
        supported_norms=["L2"],
        computational_cost="high",
        supported_model_types=["pytorch"],
    )

    def __init__(self, model: Any):
        """
        Initialize C&W attack.

        Args:
            model: PyTorch nn.Module or BaseModelAdapter wrapping PyTorch model.
        """
        super().__init__(model)
        self.raw_model = self._get_raw_model()

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate C&W-specific configuration.
        Ensures box constraints are well-formed and validates parameters.
        """
        validate_attack_config(config)
        if config is not None:
            params = config.params or {}
            c_min = params.get("clip_min", config.clip_min)
            c_max = params.get("clip_max", config.clip_max)
            if c_min is not None and c_max is not None and c_min >= c_max:
                raise AttackConfigurationError(
                    f"clip_min ({c_min}) must be strictly less than clip_max ({c_max})"
                )
            max_iter = params.get("max_iterations", params.get("iters", params.get("num_steps")))
            if max_iter is not None and max_iter <= 0:
                raise AttackConfigurationError("max_iterations must be positive")
            if "binary_search_steps" in params and params["binary_search_steps"] < 1:
                raise AttackConfigurationError("binary_search_steps must be at least 1")
            init_c = params.get("initial_const", params.get("c"))
            if init_c is not None and init_c <= 0:
                raise AttackConfigurationError("initial_const must be positive")
            lr = params.get("learning_rate", params.get("lr"))
            if lr is not None and lr <= 0:
                raise AttackConfigurationError("learning_rate must be positive")
            conf = params.get("confidence", params.get("kappa"))
            if conf is not None and conf < 0:
                raise AttackConfigurationError("confidence must be non-negative")

    def _to_tanh_space(
        self, x: torch.Tensor, clip_min: float, clip_max: float
    ) -> torch.Tensor:
        """
        Map tensor from [clip_min, clip_max] to unconstrained tanh space (-inf, +inf).
        """
        box_mul = (clip_max - clip_min) / 2.0
        box_plus = (clip_min + clip_max) / 2.0
        val = (x - box_plus) / box_mul
        val = torch.clamp(val, -0.999999, 0.999999)
        return torch.atanh(val)

    def _from_tanh_space(
        self, w: torch.Tensor, clip_min: float, clip_max: float
    ) -> torch.Tensor:
        """
        Map tensor from unconstrained tanh space (-inf, +inf) back to [clip_min, clip_max].
        """
        box_mul = (clip_max - clip_min) / 2.0
        box_plus = (clip_min + clip_max) / 2.0
        return torch.tanh(w) * box_mul + box_plus

    def generate(
        self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None
    ) -> Any:
        """
        Generate C&W L2 adversarial examples.

        Carlini & Wagner (C&W) L2 attack finds adversarial perturbations by minimizing
        the L2 distance between the adversarial example and the original input while
        satisfying a classification margin objective f(x_adv).

        Unlike PGD or FGSM, C&W is an unconstrained L2 optimization attack that does NOT
        enforce a hard global epsilon-ball or Linf constraint. Any generic 'epsilon'
        setting in AttackConfig is not enforced as a hard bound.

        Args:
            inputs: Input tensor (e.g. torch.Tensor) of shape (B, C, H, W) or (B, ...).
            labels: True class labels tensor (or target labels if targeted=True).
            config: AttackConfig containing clip_min, clip_max, and optional params:
                - max_iterations / iters / num_steps: Optimization steps per binary search step (default 20).
                - learning_rate / lr: Adam optimizer learning rate (default 0.01).
                - confidence / kappa: Margin parameter kappa (default 0.0).
                - binary_search_steps: Binary search steps for constant c (default 1).
                - initial_const / c: Initial trade-off constant c (default 1.0).
                - targeted: Boolean flag indicating targeted mode (default False).

        Returns:
            Adversarial input tensor with lowest L2 distortion for successful samples,
            preserving original inputs for samples where no adversarial perturbation was found.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        validate_inputs(inputs, labels, attack_name="CW")

        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)

        params = config.params or {}
        max_iter = int(params.get("max_iterations", params.get("iters", params.get("num_steps", 20))))
        lr = float(params.get("learning_rate", params.get("lr", 0.01)))
        confidence = float(params.get("confidence", params.get("kappa", 0.0)))
        binary_search_steps = int(params.get("binary_search_steps", 1))
        initial_const = float(params.get("initial_const", params.get("c", 1.0)))
        targeted = bool(params.get("targeted", False))
        target_label = params.get("target_label", params.get("target", None))

        clip_min = float(params.get("clip_min", config.clip_min if config.clip_min is not None else 0.0))
        clip_max = float(params.get("clip_max", config.clip_max if config.clip_max is not None else 1.0))

        device = get_device(self.raw_model)
        x_orig = inputs.clone().detach().to(device)
        labels = labels.to(device)
        batch_size = x_orig.size(0)

        if targeted and target_label is not None:
            if isinstance(target_label, int):
                loss_labels = torch.full((batch_size,), target_label, dtype=torch.long, device=device)
            else:
                target_tensor = torch.as_tensor(target_label, dtype=torch.long, device=device)
                if target_tensor.dim() == 0:
                    target_tensor = target_tensor.repeat(batch_size)
                loss_labels = target_tensor
        else:
            loss_labels = labels

        was_training = self.raw_model.training
        self.raw_model.eval()

        try:
            # Track the lowest-L2 successful adversarial example for each input independently
            best_adv = x_orig.clone().detach()
            best_l2 = torch.full((batch_size,), float("inf"), device=device)

            # Per-sample binary search bounds for trade-off constant c
            c_lower = torch.zeros(batch_size, device=device)
            c_upper = torch.full((batch_size,), float("inf"), device=device)
            c_val = torch.full((batch_size,), initial_const, device=device)

            for _ in range(binary_search_steps):
                # Map inputs to unconstrained tanh space
                w = self._to_tanh_space(x_orig, clip_min, clip_max).clone().detach()
                w.requires_grad_(True)

                optimizer = optim.Adam([w], lr=lr)
                step_success = torch.zeros(batch_size, dtype=torch.bool, device=device)

                for _ in range(max_iter):
                    optimizer.zero_grad()
                    x_adv = self._from_tanh_space(w, clip_min, clip_max)

                    # L2 distortion in input space: ||x_adv - x_orig||_2
                    delta = (x_adv - x_orig).view(batch_size, -1)
                    l2_sq = torch.sum(delta ** 2, dim=1)
                    l2_dist = torch.norm(delta, p=2, dim=1)

                    # Forward pass
                    outputs = self.raw_model(x_adv)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]

                    pred = logits.argmax(dim=1)
                    if targeted:
                        successful = (pred == loss_labels)
                    else:
                        successful = (pred != loss_labels)

                    # Update best adversarial examples for successful candidates with lower L2
                    with torch.no_grad():
                        improved = successful & (l2_dist < best_l2)
                        if improved.any():
                            best_l2[improved] = l2_dist[improved]
                            best_adv[improved] = x_adv[improved].detach()
                        step_success |= successful

                    # Compute CW margin loss f(x_adv)
                    # real_logits: logit for target class (targeted) or true class (untargeted)
                    real_logits = logits.gather(1, loss_labels.unsqueeze(1)).squeeze(1)

                    # other_logits: max logit among all classes other than label
                    masked_logits = logits.clone()
                    masked_logits.scatter_(1, loss_labels.unsqueeze(1), float("-inf"))
                    other_logits = torch.max(masked_logits, dim=1)[0]

                    if targeted:
                        # Minimize (max_{i!=t} Z_i - Z_t + kappa)
                        loss_f = torch.clamp(other_logits - real_logits + confidence, min=0.0)
                    else:
                        # Minimize (Z_t - max_{i!=t} Z_i + kappa)
                        loss_f = torch.clamp(real_logits - other_logits + confidence, min=0.0)

                    # Joint optimization objective: L2 distortion + c * classification loss
                    loss = torch.sum(l2_sq + c_val * loss_f)

                    # Preserve gradient calculation strictly with respect to w (without accumulating on model)
                    grad_w = torch.autograd.grad(loss, w, retain_graph=False, create_graph=False)[0]
                    w.grad = grad_w
                    optimizer.step()

                # Evaluate the final optimization candidate produced after the last optimizer step
                with torch.no_grad():
                    final_x_adv = self._from_tanh_space(w, clip_min, clip_max)
                    final_delta = (final_x_adv - x_orig).view(batch_size, -1)
                    final_l2_dist = torch.norm(final_delta, p=2, dim=1)
                    final_outputs = self.raw_model(final_x_adv)
                    final_logits = final_outputs.logits if hasattr(final_outputs, "logits") else final_outputs
                    if isinstance(final_logits, (tuple, list)):
                        final_logits = final_logits[0]
                    final_pred = final_logits.argmax(dim=1)
                    if targeted:
                        final_successful = (final_pred == labels)
                    else:
                        final_successful = (final_pred != labels)

                    final_improved = final_successful & (final_l2_dist < best_l2)
                    if final_improved.any():
                        best_l2[final_improved] = final_l2_dist[final_improved]
                        best_adv[final_improved] = final_x_adv[final_improved].detach()
                    step_success |= final_successful

                # Vectorized per-sample binary search updates for trade-off constant c
                has_upper = c_upper < 1e9
                c_upper = torch.where(step_success, torch.minimum(c_upper, c_val), c_upper)
                c_lower = torch.where(~step_success, torch.maximum(c_lower, c_val), c_lower)
                c_val = torch.where(
                    has_upper | step_success,
                    (c_lower + c_upper) / 2.0,
                    c_val * 10.0,
                )

            # Preserve original inputs for samples where no successful adversarial example was found
            with torch.no_grad():
                unsuccessful = torch.isinf(best_l2)
                if unsuccessful.any():
                    best_adv[unsuccessful] = x_orig[unsuccessful]

            # Box constraint enforcement
            best_adv = clip_tensor(best_adv, clip_min, clip_max)
            return best_adv.detach()

        except AttackExecutionError:
            raise
        except Exception as e:
            raise AttackExecutionError(f"Error during C&W attack generation: {str(e)}") from e
        finally:
            self.raw_model.train(was_training)


# Self-registration
register_attack("cw", CW)
register_attack("carlini_wagner", CW)
