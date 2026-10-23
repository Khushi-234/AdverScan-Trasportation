"""
Unit tests for HFVisionDatasetLoader image decoding and sample processing.
"""

import io
from PIL import Image
import pytest
from app.evaluation.dataset_loader import HFVisionDatasetLoader


def test_hf_vision_loader_decode_image_bytes():
    """Test decoding PIL Image from raw PNG bytes dict."""
    img = Image.new("RGB", (32, 32), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    sample = {"Path": {"bytes": raw_bytes}, "ClassId": 14}

    loader = HFVisionDatasetLoader.__new__(HFVisionDatasetLoader)
    decoded_img = loader._decode_image(sample)

    assert isinstance(decoded_img, Image.Image)
    assert decoded_img.size == (32, 32)
    assert decoded_img.mode == "RGB"


def test_hf_vision_loader_rgb_conversion():
    """Test explicit conversion of RGBA and grayscale images to RGB."""
    loader = HFVisionDatasetLoader.__new__(HFVisionDatasetLoader)

    # Test RGBA image
    rgba_img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
    sample_rgba = {"image": rgba_img}
    decoded_rgba = loader._decode_image(sample_rgba)
    assert decoded_rgba.mode == "RGB"

    # Test Grayscale image
    gray_img = Image.new("L", (32, 32), color=128)
    sample_gray = {"image": gray_img}
    decoded_gray = loader._decode_image(sample_gray)
    assert decoded_gray.mode == "RGB"


def test_hf_vision_loader_processor_output_shape(mocker=None):
    """Test image processor transforms PIL image into expected (N, 3, 224, 224) tensor shape."""
    from transformers import AutoImageProcessor
    import torch
    
    processor = AutoImageProcessor.from_pretrained("bazyl/gtsrb-model")
    img = Image.new("RGB", (32, 32), color="blue")
    processed = processor(images=[img, img], return_tensors="pt")
    
    assert "pixel_values" in processed
    tensor = processed["pixel_values"]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (2, 3, 224, 224)


def test_hf_vision_loader_decode_invalid_sample():
    """Test error handling when decoding sample with missing keys."""
    loader = HFVisionDatasetLoader.__new__(HFVisionDatasetLoader)
    with pytest.raises(KeyError, match="does not contain a valid image"):
        loader._decode_image({"invalid": "dict"})


def test_time_series_dataset_loader():
    """Test TimeSeriesDatasetLoader with 3D numerical sequence tensor inputs."""
    import torch
    from app.evaluation.dataset_loader import TimeSeriesDatasetLoader

    inputs = torch.randn(10, 24, 5)  # 10 samples, 24 time steps, 5 features
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    loader = TimeSeriesDatasetLoader(inputs=inputs, targets=targets, batch_size=4)
    assert len(loader) == 10

    batches = list(loader.iterate_batches())
    assert len(batches) == 3  # 4 + 4 + 2 samples
    b_in, b_tgt, t_list = batches[0]
    assert b_in.shape == (4, 24, 5)
    assert b_tgt.shape == (4,)
    assert t_list == [0, 1, 0, 1]


def test_tabular_dataset_loader_tuple():
    """Test TabularDatasetLoader with feature matrix and target vector tuple."""
    import torch
    from app.evaluation.dataset_loader import TabularDatasetLoader

    X = torch.randn(12, 8)  # 12 samples, 8 tabular features
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

    loader = TabularDatasetLoader(data=(X, y), batch_size=5)
    assert len(loader) == 12

    batches = list(loader.iterate_batches())
    assert len(batches) == 3
    b_in, b_tgt, t_list = batches[0]
    assert b_in.shape == (5, 8)
    assert b_tgt.shape == (5,)
    assert t_list == [0, 1, 2, 0, 1]


def test_get_dataset_loader_factory_domains():
    """Test get_dataset_loader factory resolving correct domain classes."""
    import torch
    from app.evaluation.dataset_loader import (
        HFVisionDatasetLoader,
        TimeSeriesDatasetLoader,
        TabularDatasetLoader,
        get_dataset_loader,
    )

    # Time series domain
    ts_loader = get_dataset_loader(
        dataset_name="custom_ts",
        data_domain="time-series",
        inputs=torch.randn(4, 10),
        targets=torch.tensor([0, 1, 0, 1]),
    )
    assert isinstance(ts_loader, TimeSeriesDatasetLoader)

    # Tabular domain
    tab_loader = get_dataset_loader(
        dataset_name="custom_tab",
        data_domain="tabular",
        data=(torch.randn(4, 5), torch.tensor([0, 1, 0, 1])),
    )
    assert isinstance(tab_loader, TabularDatasetLoader)

    # Generic domain
    from app.evaluation.dataset_loader import GenericDatasetLoader
    gen_loader = get_dataset_loader(
        dataset_name="custom_gen",
        data_domain="generic",
        inputs=torch.randn(4, 5),
        targets=torch.tensor([0, 1, 0, 1]),
    )
    assert isinstance(gen_loader, GenericDatasetLoader)


def test_get_dataset_loader_validation_and_unsupported_domains():
    """Test get_dataset_loader error handling for missing args and unsupported domains."""
    import torch
    from app.evaluation.dataset_loader import get_dataset_loader

    # Unsupported domain
    with pytest.raises(ValueError, match="Unsupported data domain"):
        get_dataset_loader(dataset_name="unknown", data_domain="quantum_computing")

    # Time series missing inputs or targets
    with pytest.raises(ValueError, match="TimeSeriesDatasetLoader requires"):
        get_dataset_loader(dataset_name="ts", data_domain="time-series")

    # Tabular missing data
    with pytest.raises(ValueError, match="TabularDatasetLoader requires"):
        get_dataset_loader(dataset_name="tab", data_domain="tabular")

    # Generic missing inputs or targets
    with pytest.raises(ValueError, match="GenericDatasetLoader requires"):
        get_dataset_loader(dataset_name="gen", data_domain="generic")


def test_hf_vision_loader_invalid_processor_raises_runtime_error(monkeypatch):
    """Test that specifying an invalid/unresolvable processor_name raises a RuntimeError."""
    from app.evaluation.dataset_loader import HFVisionDatasetLoader

    class DummyDataset:
        column_names = ["image", "label"]
        features = {}
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return {"image": Image.new("RGB", (10, 10)), "label": 0}

    import app.evaluation.dataset_loader as dl_module
    monkeypatch.setattr(dl_module, "load_dataset", lambda *args, **kwargs: DummyDataset())

    with pytest.raises(RuntimeError, match="Failed to load AutoImageProcessor"):
        HFVisionDatasetLoader(
            dataset_name="dummy_dataset",
            processor_name="non_existent_processor_12345XYZ",
        )


def test_hf_vision_loader_missing_label_raises_key_error(monkeypatch):
    """Test that a dataset missing all recognized label columns raises KeyError."""
    from app.evaluation.dataset_loader import HFVisionDatasetLoader

    class DummyUnlabeledDataset:
        column_names = ["image", "unrelated_meta"]
        features = {}
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return {"image": Image.new("RGB", (10, 10)), "unrelated_meta": "test"}

    import app.evaluation.dataset_loader as dl_module
    monkeypatch.setattr(dl_module, "load_dataset", lambda *args, **kwargs: DummyUnlabeledDataset())

    with pytest.raises(KeyError, match="does not contain any recognized label column"):
        HFVisionDatasetLoader(
            dataset_name="unlabeled_dataset",
            processor_name=None,
        )


def test_resolve_dataset_file_and_dynamic_tabular_loader():
    """Verify general dynamic dataset file resolution and automatic tabular loading."""
    from app.evaluation.dataset_loader.factory import get_dataset_loader, resolve_dataset_file

    # 1. Direct path resolution
    resolved_csv = resolve_dataset_file("data/adult_census_test.csv", [".csv"])
    assert resolved_csv is not None
    assert resolved_csv.endswith(".csv")

    # 2. Model-based resolution
    model_resolved = resolve_dataset_file(None, [".csv"], model_identifier="models/tabular-mlp-adult-census.pt")
    assert model_resolved is not None
    assert "adult_census" in model_resolved

    # 3. Dynamic loading without explicit kwargs['data']
    tab_loader = get_dataset_loader(
        dataset_name="adult_census",
        data_domain="tabular",
        batch_size=16,
    )
    assert len(tab_loader) > 0
    batch_x, batch_y, labels = next(tab_loader.iterate_batches())
    assert batch_x.shape[0] == 16
    assert batch_y.shape[0] == 16


def test_resolve_dataset_file_and_dynamic_timeseries_loader():
    """Verify general dynamic dataset file resolution and automatic time-series loading."""
    from pathlib import Path
    import torch
    from app.evaluation.dataset_loader.factory import get_dataset_loader, resolve_dataset_file

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    test_ts_file = data_dir / "time_series_ecg.pt"
    created_here = False
    if not test_ts_file.exists():
        torch.save({"inputs": torch.randn(16, 1, 50), "targets": torch.randint(0, 2, (16,))}, test_ts_file)
        created_here = True

    try:
        # 1. Model-based resolution for TS
        resolved_ts = resolve_dataset_file(None, [".pt"], model_identifier="models/time-series-ecg-classifier.pt")
        assert resolved_ts is not None
        assert resolved_ts.endswith(".pt")

        # 2. Dynamic loading without explicit kwargs['inputs'] or kwargs['targets']
        ts_loader = get_dataset_loader(
            dataset_name="time_series_ecg",
            data_domain="time-series",
            batch_size=8,
        )
        assert len(ts_loader) > 0
        batch_x, batch_y, labels = next(ts_loader.iterate_batches())
        assert batch_x.shape[0] == 8
        assert batch_y.shape[0] == 8
    finally:
        if created_here and test_ts_file.exists():
            test_ts_file.unlink()


def test_tabular_loader_multi_format():
    """Verify TabularDatasetLoader supports .tsv, .parquet, .json, and .jsonl."""
    import tempfile
    from pathlib import Path
    import pandas as pd
    from app.evaluation.dataset_loader import TabularDatasetLoader

    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0],
        "feat_b": [10.0, 20.0, 30.0, 40.0],
        "target": [0, 1, 0, 1],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. TSV
        tsv_path = Path(tmpdir) / "data.tsv"
        df.to_csv(tsv_path, sep="\t", index=False)
        loader_tsv = TabularDatasetLoader(data_path=tsv_path, target_column="target", batch_size=2)
        assert len(loader_tsv) == 4
        x_tsv, y_tsv, _ = next(loader_tsv.iterate_batches())
        assert x_tsv.shape == (2, 2)  # target must not be in features
        assert y_tsv.shape == (2,)

        # 2. Parquet
        pq_path = Path(tmpdir) / "data.parquet"
        df.to_parquet(pq_path, index=False)
        loader_pq = TabularDatasetLoader(data_path=pq_path, target_column="target", batch_size=2)
        assert len(loader_pq) == 4
        x_pq, y_pq, _ = next(loader_pq.iterate_batches())
        assert x_pq.shape == (2, 2)

        # 3. JSON
        json_path = Path(tmpdir) / "data.json"
        df.to_json(json_path, orient="records")
        loader_json = TabularDatasetLoader(data_path=json_path, target_column="target", batch_size=2)
        assert len(loader_json) == 4
        x_json, y_json, _ = next(loader_json.iterate_batches())
        assert x_json.shape == (2, 2)

        # 4. JSONL
        jsonl_path = Path(tmpdir) / "data.jsonl"
        df.to_json(jsonl_path, orient="records", lines=True)
        loader_jsonl = TabularDatasetLoader(data_path=jsonl_path, target_column="target", batch_size=2)
        assert len(loader_jsonl) == 4
        x_jsonl, y_jsonl, _ = next(loader_jsonl.iterate_batches())
        assert x_jsonl.shape == (2, 2)


def test_tabular_loader_target_separation_and_validation():
    """Verify strict target/feature separation and missing target column detection."""
    import tempfile
    from pathlib import Path
    import pandas as pd
    from app.evaluation.dataset_loader import TabularDatasetLoader
    from app.ingestion.exceptions import DatasetFormatError

    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty file check
        empty_csv = Path(tmpdir) / "empty.csv"
        empty_csv.write_text("")
        with pytest.raises(DatasetFormatError, match="empty"):
            TabularDatasetLoader(data_path=empty_csv)

        # Missing target column check
        csv_path = Path(tmpdir) / "data.csv"
        df = pd.DataFrame({"feat_1": [1.0, 2.0], "feat_2": [3.0, 4.0]})
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="Specified target column 'nonexistent_col' not found"):
            TabularDatasetLoader(data_path=csv_path, target_column="nonexistent_col")


def test_tabular_loader_xlsx_missing_dependency():
    """Verify TabularDatasetLoader raises MissingDependencyError when openpyxl is not installed."""
    import tempfile
    from pathlib import Path
    from app.evaluation.dataset_loader import TabularDatasetLoader
    from app.ingestion.exceptions import MissingDependencyError

    with tempfile.TemporaryDirectory() as tmpdir:
        xlsx_path = Path(tmpdir) / "test.xlsx"
        xlsx_path.touch()

        try:
            import openpyxl
        except ImportError:
            with pytest.raises(MissingDependencyError, match="pip install openpyxl"):
                TabularDatasetLoader(data_path=xlsx_path)


def test_timeseries_loader_multi_format():
    """Verify TimeSeriesDatasetLoader loads .npy, .npz, and .csv data."""
    import tempfile
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from app.evaluation.dataset_loader import TimeSeriesDatasetLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. NPY (3D array)
        npy_path = Path(tmpdir) / "series.npy"
        arr = np.random.randn(6, 20, 2).astype(np.float32)
        np.save(npy_path, arr)
        loader_npy = TimeSeriesDatasetLoader(data_path=npy_path, batch_size=3)
        assert len(loader_npy) == 6
        b_x, b_y, _ = next(loader_npy.iterate_batches())
        assert b_x.shape == (3, 20, 2)
        assert b_y.shape == (3,)

        # 2. NPZ (inputs + targets dictionary)
        npz_path = Path(tmpdir) / "series.npz"
        np.savez(npz_path, inputs=arr, targets=np.array([0, 1, 0, 1, 0, 1]))
        loader_npz = TimeSeriesDatasetLoader(data_path=npz_path, batch_size=2)
        assert len(loader_npz) == 6
        b_x_npz, b_y_npz, _ = next(loader_npz.iterate_batches())
        assert b_x_npz.shape == (2, 20, 2)
        assert b_y_npz.shape == (2,)

        # 3. CSV time series with target column
        csv_path = Path(tmpdir) / "ts.csv"
        df = pd.DataFrame({
            "val_t0": [1.0, 2.0, 3.0, 4.0],
            "val_t1": [1.5, 2.5, 3.5, 4.5],
            "val_t2": [1.8, 2.8, 3.8, 4.8],
            "label": [0, 1, 0, 1],
        })
        df.to_csv(csv_path, index=False)
        loader_csv = TimeSeriesDatasetLoader(data_path=csv_path, target_column="label", batch_size=2)
        assert len(loader_csv) == 4
        b_x_csv, b_y_csv, _ = next(loader_csv.iterate_batches())
        assert b_x_csv.shape == (2, 1, 3)  # [batch, 1, seq_len]
        assert b_y_csv.shape == (2,)



