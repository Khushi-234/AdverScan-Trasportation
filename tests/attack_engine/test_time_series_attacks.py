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

from app.attack_engine import AttackConfig, AttackEngine, execute_attack
from app.attack_engine.attack_discovery import discover_attacks
from app.attack_engine.attack_registry import get_attack, list_attacks
from app.attack_engine.attacks.time_series.pgd import TimeSeriesPGD
from app.attack_engine.attacks.time_series.smooth_pgd import SmoothPGD, compute_temporal_total_variation
from app.attack_engine.attacks.time_series.wavelet import WaveletAttack
from app.attack_engine.attacks.time_series.soft_dtw import SoftDTWAttack, compute_soft_dtw
from app.attack_engine.attacks.time_series.jsma import JSMA
from app.attack_engine.exceptions import (
    AttackConfigurationError,
    AttackExecutionError,
    UnsupportedModelError,
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
    for name in ["time_series_pgd", "ts_pgd", "smooth_pgd", "wavelet", "soft_dtw", "jsma"]:
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
    adv_cnn = attack_cnn.generate(torch.randn(2, 8, 3), torch.tensor([0, 1]), AttackConfig(epsilon=0.1, params={"num_steps": 3}))
    assert adv_cnn.shape == (2, 8, 3)

    attack_tf = TimeSeriesPGD(tf_model)
    adv_tf = attack_tf.generate(torch.randn(2, 6, 4), torch.tensor([1, 0]), AttackConfig(epsilon=0.1, params={"num_steps": 3}))
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
        attack.generate(torch.randn(2, 5, 3), torch.tensor([0, 1]), AttackConfig(epsilon=-0.1))

    # Invalid steps
    with pytest.raises(AttackExecutionError):
        attack.generate(torch.randn(2, 5, 3), torch.tensor([0, 1]), AttackConfig(params={"num_steps": 0}))

    # Invalid feature bounds (min > max)
    with pytest.raises(AttackConfigurationError):
        attack.generate(
            torch.randn(2, 5, 3),
            torch.tensor([0, 1]),
            AttackConfig(params={"feature_min": [1.0, 1.0, 1.0], "feature_max": [0.0, 0.0, 0.0]}),
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
        params={"num_steps": 10, "tv_weight": 0.0, "random_start": False, "alpha": 0.05},
    )
    adv_rough = attack_standard.generate(inputs, labels, config_rough)
    tv_rough = compute_temporal_total_variation(adv_rough).item()

    config_smooth = AttackConfig(
        epsilon=0.3,
        clip_min=0.0,
        clip_max=1.0,
        params={"num_steps": 10, "tv_weight": 1.0, "random_start": False, "alpha": 0.05},
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
        attack.generate(torch.randn(2, 5, 3), [0, 1], AttackConfig(params={"tv_weight": -0.5}))


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
        attack.generate(torch.randn(2, 6, 3), [0, 1], AttackConfig(params={"transform_type": "invalid"}))
    with pytest.raises(AttackConfigurationError):
        attack.generate(torch.randn(2, 6, 3), [0, 1], AttackConfig(params={"band": "invalid_band"}))


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
        attack.generate(torch.randn(2, 5, 2), [0, 1], AttackConfig(params={"max_warp": -0.1}))
    with pytest.raises(AttackConfigurationError):
        attack.generate(torch.randn(2, 5, 2), [0, 1], AttackConfig(params={"gamma": -1.0}))


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
        attack.generate(torch.randn(2, 5, 3), [0, 1], AttackConfig(params={"max_iter": 0}))

    with pytest.raises(AttackConfigurationError):
        attack.generate(torch.randn(2, 5, 3), [0, 1], AttackConfig(params={"theta": 0.0}))


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
        res = engine.run_attack(attack_name, inputs, labels, AttackConfig(params={"num_steps": 2, "max_iter": 2}))
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
    inputs = (torch.rand(2, 6, 2, device="cpu") * 0.6 + 0.2)
    labels = torch.tensor([0, 1], device="cpu")
    adv = attack.generate(inputs, labels, AttackConfig(params={"num_steps": 2}))
    assert adv.device.type == "cpu"

    # If GPU is available, verify GPU execution
    if torch.cuda.is_available():
        model.cuda()
        attack_gpu = TimeSeriesPGD(model)
        inputs_gpu = inputs.cuda()
        labels_gpu = labels.cuda()
        adv_gpu = attack_gpu.generate(inputs_gpu, labels_gpu, AttackConfig(params={"num_steps": 2}))
        assert adv_gpu.device.type == "cuda"
