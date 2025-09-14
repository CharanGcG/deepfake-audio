import os
import tempfile
import shutil
import pytest
import torch
from PIL import Image
import numpy as np

# adjust import based on your module name; assume code.dataset.DeepfakeDataset
from code.dataset import DeepfakeDataset

def make_dummy_image(path, size=(64,64)):
    arr = (np.random.rand(*size,3) * 255).astype('uint8')
    img = Image.fromarray(arr)
    img.save(path)

def test_dataset_basic(tmp_path):
    # Setup dummy dataset structure
    root = tmp_path / "real_vs_fake"
    real = root / "real"
    fake = root / "fake"
    real.mkdir(parents=True)
    fake.mkdir(parents=True)
    img_r = real / "r1.jpg"
    img_f = fake / "f1.jpg"
    make_dummy_image(img_r, size=(32,32))
    make_dummy_image(img_f, size=(32,32))
    # Create CSV file
    csv_path = tmp_path / "df.csv"
    # paths in CSV are relative to root
    with open(csv_path, "w") as f:
        f.write("path,label\n")
        f.write(f"{real.name}/{img_r.name},1\n")
        f.write(f"{fake.name}/{img_f.name},0\n")
    # Instantiate
    ds = DeepfakeDataset(
        csv_file=str(csv_path),
        root_dir=str(root),
        transform=None,
        image_size=(32,32)
    )
    assert len(ds) == 2
    # Test first sample
    sample_img, sample_label = ds[0]
    assert isinstance(sample_label, int)
    assert sample_label in (0,1)
    # check image is tensor or something convertible
    assert hasattr(sample_img, "shape")
    # labels correspond correctly
    labels = [ds[i][1] for i in range(len(ds))]
    assert set(labels) == {0,1}

def test_dataset_missing_file(tmp_path, caplog):
    # simulate missing file
    root = tmp_path / "root"
    real = root / "real"
    real.mkdir(parents=True)
    # no images here
    csv_path = tmp_path / "df2.csv"
    with open(csv_path, "w") as f:
        f.write("path,label\n")
        f.write(f"real/missing.jpg,1\n")
    ds = DeepfakeDataset(
        csv_file=str(csv_path),
        root_dir=str(root),
        transform=None,
        image_size=(32,32)
    )
    # length might still be 1, but image loading should be handled
    img, label = ds[0]
    # Maybe you return a black image tensor or something defined
    # Check that you log a warning or error
    assert "missing" in caplog.text.lower()
    assert label == 1
    assert hasattr(img, "shape")
