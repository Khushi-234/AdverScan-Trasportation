"""
VAE / On-Manifold adversarial attack implementation for tabular data.

Perturbs inputs in the latent space of a trained Variational Autoencoder (VAE) using
gradients from the target classifier, preserving realistic feature correlations on
the data manifold before projecting onto tabular constraints.

Reference:
    Kos et al., "Adversarial Examples on Forecasting Models" (2018)
    Gharib et al., "On-Manifold Adversarial Examples in Tabular Data" (2021)
"""

from typing import Any, Optional
import torch
import torch.nn as nn

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.models import AttackMetadata
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackExecutionError,
)
from app.attack_engine.attack_registry import register_attack
from app.attack_engine.utils.tensor_utils import get_device
from app.attack_engine.attacks.tabular._tabular_utils import (
    get_raw_model,
    parse_tabular_metadata,
    validate_tabular_inputs,
)


class VAEAttack(BaseAttack):
    """
    On-manifold latent adversarial attack using a trained VAE for tabular data.
    """

    metadata = AttackMetadata(
        name="vae_attack",
        domain="tabular",
        category="manifold",
        attack_type="white_box",
        requires_gradient=True,
        requires_loss=True,
        requires_model_weights=True,
        iterative=True,
        targeted_supported=True,
        untargeted_supported=True,
        query_based=False,
        supported_norms=["L2", "Linf"],
        computational_cost="high",
        supported_model_types=["pytorch"],
        parameters={
            "latent_space": True,
            "constraint_aware": True,
            "supports_categorical": True,
            "supports_integer": True,
            "supports_continuous": True,
        },
    )

    def __init__(self, model: Any):
        super().__init__(model)
        self.raw_model = get_raw_model(self.model)

    def _get_vae(self, config_params: Optional[dict] = None) -> Any:
        """
        Locate and return the trained VAE instance from config parameters or model.

        Raises:
            AttackConfigurationError: If no VAE is available.
        """
        params = config_params or {}
        vae = params.get("vae", None)
        if vae is None:
            vae = getattr(self.model, "vae", None)
        if vae is None:
            vae = getattr(self.raw_model, "vae", None)

        if vae is None:
            raise AttackConfigurationError(
                "VAEAttack requires a trained VAE instance provided via config.params['vae'] "
                "or model.vae, with encode() and decode() methods."
            )

        # Validate VAE methods
        has_encode = hasattr(vae, "encode") or hasattr(vae, "encoder")
        has_decode = hasattr(vae, "decode") or hasattr(vae, "decoder")
        if not (has_encode and has_decode):
            raise AttackConfigurationError(
                "Provided VAE must implement encode() and decode() methods."
            )

        return vae

    def validate_config(self, config: Optional[AttackConfig] = None) -> None:
        """
        Validate VAEAttack configuration parameters and VAE availability.
        """
        if config is None:
            return

        params = config.params or {}

        # Validate VAE presence
        self._get_vae(params)

        epsilon = params.get("epsilon", params.get("eps_latent", 0.5))
        if epsilon < 0:
            raise AttackConfigurationError(
                f"epsilon/eps_latent cannot be negative, got {epsilon}"
            )

        alpha = params.get("alpha", params.get("step_size", 0.05))
        if alpha <= 0:
            raise AttackConfigurationError(
                f"step_size (alpha) must be positive, got {alpha}"
            )

        num_steps = params.get(
            "num_steps", params.get("steps", params.get("iterations", 20))
        )
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise AttackConfigurationError(
                f"num_steps must be a positive integer, got {num_steps}"
            )

    def _encode_to_latent(self, vae: Any, x: torch.Tensor) -> torch.Tensor:
        """Helper to encode input to latent vector z."""
        if hasattr(vae, "encode"):
            out = vae.encode(x)
        else:
            out = vae.encoder(x)

        if isinstance(out, tuple):
            # Typically (mu, logvar) or (z, mu, logvar) -> use mu / first element
            return out[0]
        return out

    def _decode_from_latent(self, vae: Any, z: torch.Tensor) -> torch.Tensor:
        """Helper to decode latent vector z to feature space."""
        if hasattr(vae, "decode"):
            out = vae.decode(z)
        else:
            out = vae.decoder(z)

        if isinstance(out, tuple):
            return out[0]
        return out

    def generate(
        self, inputs: Any, labels: Any, config: Optional[AttackConfig] = None
    ) -> torch.Tensor:
        """
        Generate on-manifold adversarial examples via latent VAE optimization.

        Args:
            inputs: Tabular tensor of shape (batch_size, num_features).
            labels: Ground truth labels (or target labels if targeted).
            config: Optional AttackConfig containing 'vae' in params.

        Returns:
            Adversarial tensor satisfying tabular constraints.
        """
        if config is None:
            config = AttackConfig()

        self.validate_config(config)
        params = config.params or {}

        vae = self._get_vae(params)
        eps_latent = float(params.get("epsilon", params.get("eps_latent", 0.5)))
        alpha = float(params.get("alpha", params.get("step_size", 0.05)))
        num_steps = int(
            params.get("num_steps", params.get("steps", params.get("iterations", 20)))
        )
        targeted = bool(params.get("targeted", False))
        target_label = params.get("target_label", None)
        loss_fn = config.loss_fn or nn.CrossEntropyLoss()
        clip_min = params.get(
            "clip_min",
            (
                config.clip_min
                if (
                    "clip_min" in params
                    or config.clip_min != 0.0
                    or config.clip_max != 1.0
                )
                else None
            ),
        )
        clip_max = params.get(
            "clip_max",
            (
                config.clip_max
                if (
                    "clip_max" in params
                    or config.clip_min != 0.0
                    or config.clip_max != 1.0
                )
                else None
            ),
        )

        device = get_device(self.raw_model)
        x_orig, y = validate_tabular_inputs(inputs, labels, attack_name="VAEAttack")
        x_orig = x_orig.to(device)
        y = y.to(device)

        batch_size, num_features = x_orig.shape

        if targeted:
            if target_label is not None:
                if isinstance(target_label, int):
                    y_target = torch.full(
                        (batch_size,), target_label, dtype=torch.long, device=device
                    )
                else:
                    y_target = torch.as_tensor(
                        target_label, dtype=torch.long, device=device
                    )
                    if y_target.dim() == 0:
                        y_target = y_target.repeat(batch_size)
            else:
                y_target = y
            labels_eval = y_target
        else:
            labels_eval = y

        metadata = parse_tabular_metadata(self.model, params, num_features=num_features)

        # Freeze model and VAE weights
        was_training_model = self.raw_model.training
        self.raw_model.eval()

        was_training_vae = vae.training if hasattr(vae, "training") else False
        if hasattr(vae, "eval"):
            vae.eval()

        # Step 1: Encode original sample to latent representation z
        with torch.no_grad():
            z_orig = self._encode_to_latent(vae, x_orig).detach()

        # Step 2: Optimize latent perturbation delta_z
        delta_z = torch.zeros_like(z_orig, requires_grad=True)

        for step in range(num_steps):
            z_perturbed = z_orig + delta_z
            x_recon = self._decode_from_latent(vae, z_perturbed)

            outputs = self.raw_model(x_recon)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            if targeted:
                clf_loss = -loss_fn(logits, labels_eval)
            else:
                clf_loss = loss_fn(logits, labels_eval)

            grad = torch.autograd.grad(
                clf_loss,
                delta_z,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]

            if grad is None:
                raise AttackExecutionError(
                    "VAEAttack failed to compute gradients w.r.t latent vector."
                )

            with torch.no_grad():
                # Gradient ascent step in latent space
                delta_z = delta_z + alpha * grad.sign()
                if eps_latent > 0:
                    delta_z = torch.clamp(delta_z, -eps_latent, eps_latent)

            delta_z.requires_grad_(True)

        # Step 3: Decode perturbed latent vector and project onto tabular constraints
        with torch.no_grad():
            x_adv_raw = self._decode_from_latent(vae, z_orig + delta_z)
            x_adv = metadata.project(
                x_adv=x_adv_raw,
                x_orig=x_orig,
                clip_min=clip_min,
                clip_max=clip_max,
            )

        self.raw_model.train(was_training_model)
        if hasattr(vae, "train"):
            vae.train(was_training_vae)

        return x_adv.detach()


# Self-register
register_attack("vae_attack", VAEAttack)
register_attack("tabular_vae", VAEAttack)
