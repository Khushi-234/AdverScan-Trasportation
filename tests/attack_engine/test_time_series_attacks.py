"""
Comprehensive unit and integration tests for time-series adversarial attacks:
- TimeSeriesPGD (pgd.py)
- SmoothPGD (smooth_pgd.py)
- WaveletAttack (wavelet.py)
- SoftDTWAttack (soft_dtw.py)
- JSMA (jsma.py)
"""

import pytest
import torch
import torch.nn as nn

from app.attack_engine import AttackConfig, AttackEngine
from app.attack_engine.attack_discovery import discover_attacks
from app.attack_engine.attack_registry import get_attack, list_attacks
from app.attack_engine.attacks.time_series.pgd import TimeSeriesPGD
from app.attack_engine.attacks.time_series.smooth_pgd import (
    SmoothPGD,
    compute_temporal_total_variation,
)
from app.attack_engine.attacks.time_series.wavelet import WaveletAttack
from app.attack_engine.attacks.time_series.soft_dtw import SoftDTWAttack
from app.attack_engine.attacks.time_series.jsma import JSMA
from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackExecutionError,
)

# =====================================================================
# Synthetic Time-Series Models (LSTM, GRU, 1D CNN, Transformer)
# =====================================================================


class SyntheticLSTM(nn.Module):
    def __init__(self, in_features: int = 3, hidden_dim: int = 8, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class SyntheticGRU(nn.Module):
    def __init__(self, in_features: int = 3, hidden_dim: int = 8, num_classes: int = 2):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class Synthetic1DCNN(nn.Module):
    def __init__(self, in_features: int = 3, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(in_features, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, T, D) -> Conv1d expects (B, D, T)
        feat = x.permute(0, 2, 1)
        out = self.relu(self.conv(feat))
        out = self.pool(out).squeeze(-1)
        return self.fc(out)


class SyntheticTransformer(nn.Module):
    def __init__(self, in_features: int = 4, num_classes: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=in_features, nhead=2, dim_feedforward=8, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        return self.fc(out[:, -1, :])


# =====================================================================
# Registration & Discovery Tests
# =====================================================================


def test_time_series_attacks_discovery():
    discover_attacks(force_reload=True)
    registered = list_attacks()
    for name in [
        "time_series_pgd",
        "ts_pgd",
        "smooth_pgd",
        "wavelet",
        "soft_dtw",
        "jsma",
    ]:
        assert name in registered
        cls = get_attack(name)
        assert cls.metadata.domain == "time_series"


# =====================================================================
# 1. Time-Series PGD Tests
# =====================================================================


def test_ts_pgd_basic_lstm():
    model = SyntheticLSTM(in_features=3, hidden_dim=8, num_classes=2)
    attack = TimeSeriesPGD(model)
    inputs = torch.empty(4, 12, 3).uniform_(-0.8, 0.8)
    labels = torch.tensor([0, 1, 0, 1])

    config = AttackConfig(
        epsilon=0.2,
        clip_min=-1.0,
        clip_max=1.0,
        params={"num_steps": 5, "alpha": 0.05, "random_start": True, "norm": "Linf"},
    )
    adv = attack.generate(inputs, labels, config)

    assert isinstance(adv, torch.Tensor)
    assert adv.shape == inputs.shape
    assert not torch.equal(adv, inputs)
    diff = (adv - inputs).abs()
    assert (diff <= 0.2001).all()
    assert (adv >= -1.0).all()
    assert (adv <= 1.0).all()


def test_ts_pgd_l2_norm():
    model = SyntheticGRU(in_features=4, hidden_dim=8, num_classes=2)
    attack = TimeSeriesPGD(model)
    inputs = torch.rand(3, 10, 4) * 0.6 + 0.2
    labels = torch.tensor([1, 0, 1])
    eps = 0.5

    config = AttackConfig(
        epsilon=eps,
        clip_min=0.0,
        clip_max=1.0,
        params={"norm": "L2", "num_steps": 5, "random_start": True},
    )
    adv = attack.generate(inputs, labels, config)

    assert adv.shape == inputs.shape
    delta = (adv - inputs).view(inputs.size(0), -1)
    l2_norms = torch.norm(delta, p=2, dim=1)
    assert (l2_norms <= eps + 1e-4).all()


def test_ts_pgd_architectures():
    # Verify execution on 1D CNN and Transformer
    cnn_model = Synthetic1DCNN(in_features=3, num_classes=2)
    tf_model = SyntheticTransformer(in_features=4, num_classes=2)

    attack_cnn = TimeSeriesPGD(cnn_model)
    adv_cnn = attack_cnn.generate(
        torch.randn(2, 8, 3),
        torch.tensor([0, 1]),
        AttackConfig(epsilon=0.1, params={"num_steps": 3}),
    )
    assert adv_cnn.shape == (2, 8, 3)

    attack_tf = TimeSeriesPGD(tf_model)
    adv_tf = attack_tf.generate(
        torch.randn(2, 6, 4),
        torch.tensor([1, 0]),
        AttackConfig(epsilon=0.1, params={"num_steps": 3}),
    )
    assert adv_tf.shape == (2, 6, 4)


def test_ts_pgd_feature_bounds():
    model = SyntheticLSTM(in_features=3, hidden_dim=4, num_classes=2)
    attack = TimeSeriesPGD(model)
    inputs = torch.zeros(2, 6, 3)
    labels = torch.tensor([0, 1])

    f_min = [-0.1, -0.2, -0.3]
    f_max = [0.1, 0.2, 0.3]

    config = AttackConfig(
        epsilon=1.0,
        params={"feature_min": f_min, "feature_max": f_max, "num_steps": 5},
    )
    adv = attack.generate(inputs, labels, config)
    for d in range(3):
        assert (adv[:, :, d] >= f_min[d] - 1e-5).all()
        assert (adv[:, :, d] <= f_max[d] + 1e-5).all()


def test_ts_pgd_targeted():
    model = SyntheticLSTM(in_features=2, hidden_dim=6, num_classes=3)
    attack = TimeSeriesPGD(model)
    inputs = torch.randn(2, 8, 2)
    target_labels = torch.tensor([2, 0])

    config = AttackConfig(
        epsilon=0.3,
        params={"targeted": True, "num_steps": 5, "alpha": 0.1},
    )
    adv = attack.generate(inputs, target_labels=target_labels, config=config)
    assert adv.shape == inputs.shape


def test_ts_pgd_validation_errors():
    model = SyntheticLSTM()
    attack = TimeSeriesPGD(model)

    # Invalid input type
    with pytest.raises(AttackExecutionError):
        attack.generate(inputs="not_a_tensor", labels=[0])

    # Incompatible dimensions (1D tensor)
    with pytest.raises(AttackExecutionError):
        attack.generate(inputs=torch.randn(10), labels=[0])

    # Negative epsilon
    with pytest.raises(AttackConfigurationError):
        attack.generate(
            torch.randn(2, 5, 3), torch.tensor([0, 1]), AttackConfig(epsilon=-0.1)
        )

    # Invalid steps
    with pytest.raises(AttackExecutionError):
        attack.generate(
            torch.randn(2, 5, 3),
            torch.tensor([0, 1]),
            AttackConfig(params={"num_steps": 0}),
        )

    # Invalid feature bounds (min > max)
    with pytest.raises(AttackConfigurationError):
        attack.generate(
            torch.randn(2, 5, 3),
            torch.tensor([0, 1]),
            AttackConfig(
                params={"feature_min": [1.0, 1.0, 1.0], "feature_max": [0.0, 0.0, 0.0]}
            ),
        )


# =====================================================================
# 2. Smooth PGD Tests
# =====================================================================


def test_smooth_pgd_basic():
    model = SyntheticLSTM(in_features=3, hidden_dim=8, num_classes=2)
    attack = SmoothPGD(model)
    inputs = torch.empty(2, 10, 3).uniform_(-0.8, 0.8)
    labels = torch.tensor([0, 1])

    config = AttackConfig(
        epsilon=0.2,
        clip_min=-1.0,
        clip_max=1.0,
        params={"num_steps": 5, "tv_weight": 0.2, "norm": "Linf"},
    )
    adv = attack.generate(inputs, labels, config)

    assert adv.shape == inputs.shape
    assert not torch.equal(adv, inputs)
    diff = (adv - inputs).abs()
    assert (diff <= 0.2001).all()


def test_smooth_pgd_tv_reduces_roughness():
    # Comparing smoothness of adversarial output with high TV weight vs zero TV weight
    model = SyntheticLSTM(in_features=2, hidden_dim=8, num_classes=2)
    inputs = torch.rand(2, 15, 2) * 0.6 + 0.2
    labels = torch.tensor([0, 1])

    attack_standard = SmoothPGD(model)
    config_rough = AttackConfig(
        epsilon=0.3,
        clip_min=0.0,
        clip_max=1.0,
        params={
            "num_steps": 10,
            "tv_weight": 0.0,
            "random_start": False,
            "alpha": 0.05,
        },
    )
    adv_rough = attack_standard.generate(inputs, labels, config_rough)
    tv_rough = compute_temporal_total_variation(adv_rough).item()

    config_smooth = AttackConfig(
        epsilon=0.3,
        clip_min=0.0,
        clip_max=1.0,
        params={
            "num_steps": 10,
            "tv_weight": 1.0,
            "random_start": False,
            "alpha": 0.05,
        },
    )
    adv_smooth = attack_standard.generate(inputs, labels, config_smooth)
    tv_smooth = compute_temporal_total_variation(adv_smooth).item()

    # The smooth version should penalize TV during optimization
    assert tv_smooth <= tv_rough


def test_smooth_pgd_targeted_and_l2():
    model = Synthetic1DCNN(in_features=3, num_classes=3)
    attack = SmoothPGD(model)
    inputs = torch.rand(2, 8, 3) * 0.6 + 0.2
    targets = torch.tensor([2, 1])

    config = AttackConfig(
        epsilon=0.4,
        clip_min=0.0,
        clip_max=1.0,
        params={"norm": "L2", "targeted": True, "num_steps": 4, "tv_weight": 0.1},
    )
    adv = attack.generate(inputs, target_labels=targets, config=config)
    assert adv.shape == inputs.shape
    delta = (adv - inputs).view(2, -1)
    assert (torch.norm(delta, p=2, dim=1) <= 0.4001).all()


def test_smooth_pgd_validation_error():
    model = SyntheticLSTM()
    attack = SmoothPGD(model)
    with pytest.raises(AttackConfigurationError):
        attack.generate(
            torch.randn(2, 5, 3), [0, 1], AttackConfig(params={"tv_weight": -0.5})
        )


# =====================================================================
# 3. Wavelet Attack Tests
# =====================================================================


def test_wavelet_dwt_haar():
    model = SyntheticGRU(in_features=3, hidden_dim=8, num_classes=2)
    attack = WaveletAttack(model)
    # Test with both even and odd sequence lengths
    for seq_len in [10, 11]:
        inputs = torch.rand(2, seq_len, 3) * 0.6 + 0.2
        labels = torch.tensor([0, 1])
        config = AttackConfig(
            epsilon=0.2,
            clip_min=0.0,
            clip_max=1.0,
            params={"transform_type": "dwt", "band": "low", "num_steps": 4},
        )
        adv = attack.generate(inputs, labels, config)
        assert adv.shape == inputs.shape
        assert not torch.equal(adv, inputs)
        assert ((adv - inputs).abs() <= 0.2001).all()


def test_wavelet_fft():
    model = Synthetic1DCNN(in_features=3, num_classes=2)
    attack = WaveletAttack(model)
    inputs = torch.rand(2, 12, 3) * 0.6 + 0.2
    labels = torch.tensor([1, 0])

    for band in ["all", "low", "high", "mid"]:
        config = AttackConfig(
            epsilon=0.15,
            clip_min=0.0,
            clip_max=1.0,
            params={"transform_type": "fft", "band": band, "num_steps": 3},
        )
        adv = attack.generate(inputs, labels, config)
        assert adv.shape == inputs.shape
        assert ((adv - inputs).abs() <= 0.1501).all()


def test_wavelet_invalid_config():
    model = SyntheticLSTM()
    attack = WaveletAttack(model)
    with pytest.raises(AttackConfigurationError):
        attack.generate(
            torch.randn(2, 6, 3),
            [0, 1],
            AttackConfig(params={"transform_type": "invalid"}),
        )
    with pytest.raises(AttackConfigurationError):
        attack.generate(
            torch.randn(2, 6, 3), [0, 1], AttackConfig(params={"band": "invalid_band"})
        )


# =====================================================================
# 4. Soft-DTW Attack Tests
# =====================================================================


def test_soft_dtw_basic_warping():
    model = SyntheticLSTM(in_features=2, hidden_dim=6, num_classes=2)
    attack = SoftDTWAttack(model)
    inputs = torch.randn(2, 8, 2)
    labels = torch.tensor([0, 1])

    config = AttackConfig(
        epsilon=0.2,
        params={"num_steps": 4, "max_warp": 0.15, "gamma": 0.5, "lambda_dtw": 0.05},
    )
    adv = attack.generate(inputs, labels, config)

    assert adv.shape == inputs.shape
    assert isinstance(adv, torch.Tensor)
    # Warping should alter the time sequence
    assert not torch.equal(adv, inputs)


def test_soft_dtw_targeted():
    model = SyntheticGRU(in_features=3, hidden_dim=6, num_classes=3)
    attack = SoftDTWAttack(model)
    inputs = torch.randn(2, 10, 3)
    targets = torch.tensor([2, 0])

    config = AttackConfig(
        params={"targeted": True, "num_steps": 3, "max_warp": 0.1, "gamma": 0.5},
    )
    adv = attack.generate(inputs, target_labels=targets, config=config)
    assert adv.shape == inputs.shape


def test_soft_dtw_validation_errors():
    model = SyntheticLSTM()
    attack = SoftDTWAttack(model)

    with pytest.raises(AttackConfigurationError):
        attack.generate(
            torch.randn(2, 5, 2), [0, 1], AttackConfig(params={"max_warp": -0.1})
        )
    with pytest.raises(AttackConfigurationError):
        attack.generate(
            torch.randn(2, 5, 2), [0, 1], AttackConfig(params={"gamma": -1.0})
        )


# =====================================================================
# 5. JSMA Tests
# =====================================================================


def test_jsma_sparsity_l0():
    model = SyntheticLSTM(in_features=3, hidden_dim=6, num_classes=3)
    attack = JSMA(model)
    inputs = torch.full((2, 6, 3), 0.5)
    targets = torch.tensor([1, 2])

    config = AttackConfig(
        clip_min=0.0,
        clip_max=1.0,
        params={"targeted": True, "max_iter": 3, "theta": 0.2},
    )
    adv = attack.generate(inputs, target_labels=targets, config=config)

    assert adv.shape == inputs.shape
    # Check sparsity: only a small number of (t, d) locations should change per sample
    delta = (adv - inputs).abs()
    for b in range(2):
        modified_count = (delta[b] > 1e-5).sum().item()
        assert modified_count <= 3  # At most max_iter locations modified!


def test_jsma_untargeted():
    model = SyntheticGRU(in_features=2, hidden_dim=6, num_classes=2)
    attack = JSMA(model)
    inputs = torch.full((2, 5, 2), 0.5)
    labels = torch.tensor([0, 1])

    config = AttackConfig(
        clip_min=0.0,
        clip_max=1.0,
        params={"max_iter": 4, "theta": 0.15},
    )
    adv = attack.generate(inputs, labels, config)
    assert adv.shape == inputs.shape
    assert (adv >= 0.0).all() and (adv <= 1.0).all()


def test_jsma_validation_errors():
    model = SyntheticLSTM()
    attack = JSMA(model)

    with pytest.raises(AttackExecutionError):
        attack.generate(
            torch.randn(2, 5, 3), [0, 1], AttackConfig(params={"max_iter": 0})
        )

    with pytest.raises(AttackConfigurationError):
        attack.generate(
            torch.randn(2, 5, 3), [0, 1], AttackConfig(params={"theta": 0.0})
        )


# =====================================================================
# Integration with AttackEngine & execute_attack
# =====================================================================


def test_attack_engine_integration_time_series():
    model = SyntheticLSTM(in_features=3, hidden_dim=8, num_classes=2)
    engine = AttackEngine(model)
    inputs = torch.rand(2, 8, 3) * 0.6 + 0.2
    labels = torch.tensor([0, 1])

    # Run single attacks via AttackEngine
    for attack_name in ["time_series_pgd", "smooth_pgd", "wavelet", "soft_dtw", "jsma"]:
        res = engine.run_attack(
            attack_name,
            inputs,
            labels,
            AttackConfig(params={"num_steps": 2, "max_iter": 2}),
        )
        assert res.adversarial_examples.shape == inputs.shape
        assert res.metadata.domain == "time_series"
        assert res.perturbation_metrics is not None
        assert "l0" in res.perturbation_metrics
        assert "l2" in res.perturbation_metrics
        assert "linf" in res.perturbation_metrics


def test_attack_pipeline_time_series():
    model = SyntheticLSTM(in_features=2, hidden_dim=6, num_classes=2)
    engine = AttackEngine(model)
    inputs = torch.rand(2, 6, 2) * 0.6 + 0.2
    labels = torch.tensor([1, 0])

    attacks_to_run = ["ts_pgd", "smooth_pgd", "wavelet"]
    configs = {
        "ts_pgd": AttackConfig(epsilon=0.1, params={"num_steps": 2}),
        "smooth_pgd": AttackConfig(epsilon=0.1, params={"num_steps": 2}),
        "wavelet": AttackConfig(epsilon=0.1, params={"num_steps": 2}),
    }
    results = engine.run_pipeline(attacks_to_run, inputs, labels, configs=configs)
    assert len(results) == 3
    for name in attacks_to_run:
        assert name in results
        assert results[name].adv_inputs.shape == inputs.shape


def test_cpu_and_device_handling():
    model = SyntheticLSTM(in_features=2, hidden_dim=4, num_classes=2)
    # Ensure CPU works explicitly
    model.cpu()
    attack = TimeSeriesPGD(model)
    inputs = torch.rand(2, 6, 2, device="cpu") * 0.6 + 0.2
    labels = torch.tensor([0, 1], device="cpu")
    adv = attack.generate(inputs, labels, AttackConfig(params={"num_steps": 2}))
    assert adv.device.type == "cpu"

    # If GPU is available, verify GPU execution
    if torch.cuda.is_available():
        model.cuda()
        attack_gpu = TimeSeriesPGD(model)
        inputs_gpu = inputs.cuda()
        labels_gpu = labels.cuda()
        adv_gpu = attack_gpu.generate(
            inputs_gpu, labels_gpu, AttackConfig(params={"num_steps": 2})
        )
        assert adv_gpu.device.type == "cuda"


# =====================================================================
# Additional Research-Grade Regression Tests
# =====================================================================


class ZeroGradModel(nn.Module):
    """Model with zero gradients w.r.t inputs for edge-case testing."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # Detach input so gradients w.r.t input are exactly zero
        return torch.zeros(B, self.num_classes, device=x.device, requires_grad=True)


def test_ts_pgd_epsilon_zero_and_state_restoration():
    model = SyntheticLSTM(in_features=2, hidden_dim=4, num_classes=2)
    model.train()  # Explicitly in training mode
    attack = TimeSeriesPGD(model)
    inputs = torch.randn(2, 5, 2)
    labels = torch.tensor([0, 1])

    # Epsilon = 0: must return bounded inputs and preserve model.training == True
    adv = attack.generate(
        inputs, labels, AttackConfig(epsilon=0.0, clip_min=None, clip_max=None)
    )
    assert torch.allclose(adv, inputs, atol=1e-5)
    assert model.training is True


def test_ts_pgd_zero_gradient_and_nan_inputs():
    model = ZeroGradModel(num_classes=2)
    attack = TimeSeriesPGD(model)
    inputs = torch.randn(2, 6, 3)
    labels = torch.tensor([0, 1])

    # Zero gradient: should not produce NaN or Inf
    adv = attack.generate(
        inputs, labels, AttackConfig(epsilon=0.1, params={"num_steps": 3})
    )
    assert torch.isfinite(adv).all()

    # Input containing NaN must raise AttackExecutionError
    nan_inputs = inputs.clone()
    nan_inputs[0, 0, 0] = float("nan")
    with pytest.raises(AttackExecutionError):
        attack.generate(nan_inputs, labels, AttackConfig(epsilon=0.1))


def test_ts_pgd_2d_inputs():
    # 2D input (B, T) with univariate time series
    class Simple2DModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = Simple2DModel()
    attack = TimeSeriesPGD(model)
    inputs_2d = torch.rand(3, 8) * 0.6 + 0.2
    labels = torch.tensor([0, 1, 0])

    adv_2d = attack.generate(
        inputs_2d, labels, AttackConfig(epsilon=0.1, params={"num_steps": 2})
    )
    assert adv_2d.dim() == 2
    assert adv_2d.shape == inputs_2d.shape
    assert ((adv_2d - inputs_2d).abs() <= 0.1001).all()


def test_smooth_pgd_tv_zero_and_regularize_perturbation():
    model = SyntheticLSTM(in_features=2, hidden_dim=4, num_classes=2)
    attack = SmoothPGD(model)
    inputs = torch.rand(2, 6, 2) * 0.6 + 0.2
    labels = torch.tensor([0, 1])

    # tv_weight = 0 should execute without computing TV loss and remain valid
    adv_zero_tv = attack.generate(
        inputs,
        labels,
        AttackConfig(epsilon=0.2, params={"tv_weight": 0.0, "num_steps": 3}),
    )
    assert adv_zero_tv.shape == inputs.shape
    assert torch.isfinite(adv_zero_tv).all()

    # regularize_perturbation = True should regularize delta instead of x_adv
    adv_reg_pert = attack.generate(
        inputs,
        labels,
        AttackConfig(
            epsilon=0.2,
            params={"tv_weight": 0.5, "regularize_perturbation": True, "num_steps": 3},
        ),
    )
    assert adv_reg_pert.shape == inputs.shape
    assert torch.isfinite(adv_reg_pert).all()
    assert ((adv_reg_pert - inputs).abs() <= 0.2001).all()


def test_smooth_pgd_seq_len_one_and_epsilon_zero():
    model = Synthetic1DCNN(in_features=3, num_classes=2)
    attack = SmoothPGD(model)
    # Sequence length 1 with values in valid range [0.2, 0.8]
    inputs = torch.rand(2, 1, 3) * 0.6 + 0.2
    labels = torch.tensor([0, 1])

    adv = attack.generate(
        inputs,
        labels,
        AttackConfig(epsilon=0.2, params={"num_steps": 2, "tv_weight": 0.5}),
    )
    assert adv.shape == inputs.shape
    assert torch.isfinite(adv).all()

    # Epsilon = 0
    adv_eps0 = attack.generate(inputs, labels, AttackConfig(epsilon=0.0))
    assert torch.allclose(adv_eps0, inputs, atol=1e-5)


def test_jsma_theta_negative_and_no_repeated_coords():
    model = SyntheticLSTM(in_features=3, hidden_dim=6, num_classes=3)
    attack = JSMA(model)
    inputs = torch.full((2, 5, 3), 0.7)
    targets = torch.tensor([1, 2])

    config = AttackConfig(
        clip_min=0.0,
        clip_max=1.0,
        params={"targeted": True, "max_iter": 4, "theta": -0.2},
    )
    adv = attack.generate(inputs, target_labels=targets, config=config)
    assert adv.shape == inputs.shape
    assert torch.isfinite(adv).all()

    # Verify no coordinates were modified more than max_iter times
    delta = (adv - inputs).abs()
    for b in range(2):
        modified_count = (delta[b] > 1e-4).sum().item()
        assert modified_count <= 4


def test_jsma_already_misclassified_and_2d_shape():
    class ConstantClassModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B = x.size(0)
            logits = torch.zeros(B, 3, device=x.device)
            logits[:, 2] = 10.0  # Always predicts class 2
            return logits

    model = ConstantClassModel()
    attack = JSMA(model)
    # Target is already class 2
    inputs = torch.full((2, 4, 2), 0.5)
    targets = torch.tensor([2, 2])

    adv = attack.generate(
        inputs, target_labels=targets, config=AttackConfig(clip_min=0.0, clip_max=1.0)
    )
    # Already target class -> no modifications made
    assert torch.allclose(adv, inputs, atol=1e-5)


def test_wavelet_dwt_roundtrip_and_invalid_band():
    from app.attack_engine.attacks.time_series.wavelet import (
        _dwt_haar_1d,
        _idwt_haar_1d,
    )

    # Test round-trip reconstruction on odd and even lengths
    for T in [5, 6, 9, 12]:
        x = torch.randn(3, T, 4)
        cA, cD, pad, orig_T, B, D = _dwt_haar_1d(x)
        rec = _idwt_haar_1d(cA, cD, pad, orig_T, B, D)
        assert torch.allclose(x, rec, atol=1e-5)

    model = SyntheticLSTM(in_features=2, hidden_dim=4, num_classes=2)
    attack = WaveletAttack(model)

    # "mid" band is invalid for DWT and must raise AttackConfigurationError
    with pytest.raises(AttackConfigurationError):
        attack.generate(
            torch.randn(2, 6, 2),
            torch.tensor([0, 1]),
            AttackConfig(params={"transform_type": "dwt", "band": "mid"}),
        )

    # Epsilon = 0 returns bounded original
    x_in = torch.rand(2, 6, 2) * 0.6 + 0.2
    adv_zero = attack.generate(x_in, torch.tensor([0, 1]), AttackConfig(epsilon=0.0))
    assert torch.allclose(adv_zero, x_in, atol=1e-5)


def test_soft_dtw_seq_len_one_and_long_sequence():
    model = SyntheticLSTM(in_features=2, hidden_dim=4, num_classes=2)
    attack = SoftDTWAttack(model)

    # Sequence length 1
    x_short = torch.randn(2, 1, 2)
    adv_short = attack.generate(
        x_short, torch.tensor([0, 1]), AttackConfig(params={"num_steps": 2})
    )
    assert adv_short.shape == x_short.shape
    assert torch.isfinite(adv_short).all()

    # Long sequence with downsampling threshold
    x_long = torch.randn(2, 70, 2)
    adv_long = attack.generate(
        x_long,
        torch.tensor([0, 1]),
        AttackConfig(
            params={"num_steps": 2, "downsample_threshold": 32, "max_warp": 0.1}
        ),
    )
    assert adv_long.shape == x_long.shape
    assert torch.isfinite(adv_long).all()

    # Determinism with same seed
    torch.manual_seed(123)
    adv_1 = attack.generate(
        x_short, torch.tensor([0, 1]), AttackConfig(params={"num_steps": 2})
    )
    torch.manual_seed(123)
    adv_2 = attack.generate(
        x_short, torch.tensor([0, 1]), AttackConfig(params={"num_steps": 2})
    )
    assert torch.allclose(adv_1, adv_2, atol=1e-6)
