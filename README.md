# Deepfake Detection Project — README

**Last updated:** 2025-09-12

This README explains how to set up, run, and maintain the deepfake detection project. It covers environment setup, dataset preparation, training (two-phase head-only + fine-tune flow), checkpointing/versioning, logging, GradCAM explainability, testing, reproducibility, troubleshooting, and maintenance.

---

# Table of contents

1. Project overview
2. Repository structure
3. Prerequisites
4. Installation
5. Dataset preparation
6. Quick start (run.sh)
7. Training: session-by-session
8. Checkpointing & model versioning
9. Logging, TensorBoard & session logs
10. GradCAM (explainability)
11. Testing / Final evaluation
12. Reproducibility & environment management
13. Experiment tracking and HPO (optional)
14. Maintenance tasks & best practices
15. Troubleshooting
16. Contributing
17. Ethics, privacy & legal
18. Contact

---

# 1. Project overview

This repository implements a reproducible training pipeline for binary deepfake detection (real vs fake). It uses a CvT backbone (configurable), PyTorch training loops, and integrated explainability via GradCAM. The workflow uses two training phases per session: (1) head-only training, (2) backbone fine-tuning.

Key features:

* CSV-driven dataset loader (Kaggle split assumed)
* Deterministic validation/test preprocessing (images are 256×256)
* Two-phase training with checkpointing
* Global "best" model promotion across runs
* TensorBoard + per-session log files
* GradCAM visualization module
* Robust handling for missing/corrupt images
* Clear experiment reproducibility guidance

---

# 2. Repository structure

(Important files / folders only)

```
deepfake-project/
├── code/                      # Project code
│   ├── train.py               # Entry point for training
│   ├── test.py                # Entry point for final evaluation
│   ├── cam.py                 # GradCAM utility
│   ├── dataset.py             # DeepfakeDataset class
│   ├── transforms.py          # train/val transforms
│   ├── models/                # model backbones & classifier
│   ├── engine/                # training & evaluation loops
│   └── utils/                 # checkpoint, metrics, logger, seed, scheduler
├── dataset/                   # CSVs and downloaded images
│   ├── train.csv
│   ├── valid.csv
│   ├── test.csv
│   └── real_vs_fake/...
├── models/                    # Stores run artifacts and checkpoints
├── log_file/                  # Per-session logs
├── outputs/                   # GradCAM images, plots
├── requirements.txt
├── run.sh                     # Example script to run training/test/cam
└── README.md
```

---

# 3. Prerequisites

* Linux / macOS / WSL environment (Linux recommended for GPU drivers)
* Python 3.9+ (3.10/3.11 recommended)
* NVIDIA GPU + CUDA (optional but strongly recommended for training)
* \~16GB RAM (more for large batch sizes) and sufficient disk space for dataset (140K images) and model checkpoints.

Software dependencies are listed in `requirements.txt`.

---

# 4. Installation

1. Clone the repository:

```bash
git clone <your-repo-url> deepfake-project
cd deepfake-project
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) If you have GPU and CUDA, confirm PyTorch with CUDA support installed:

```python
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

---

# 5. Dataset preparation

We expect the Kaggle dataset already downloaded and placed under `dataset/` with CSVs and a `real_vs_fake/real-vs-fake/` folder structure. The pipeline expects CSV files named `train.csv`, `valid.csv`, and `test.csv` and each CSV must contain at least these columns:

* `path` — relative path within `dataset/real_vs_fake/real-vs-fake/` (e.g. `train/real/0001.jpg`)
* `label` — integer 1 (real) or 0 (fake)

If your CSVs include extra columns (e.g., `id`, `original_path`), they will be ignored.

Place the dataset like:

```
dataset/
  train.csv
  valid.csv
  test.csv
  real_vs_fake/real-vs-fake/
    train/real/...
    train/fake/...
    valid/...
    test/...
```

---

# 6. Quick start

A ready-to-run example `run.sh` is included. Edit the paths at the top of `run.sh` to point to your dataset and desired output folders, then run:

```bash
bash run.sh
```

`run.sh` executes three stages sequentially:

1. `train` — runs head-only + fine-tune phases (produces model artifacts in `models/` and logs in `log_file/`).
2. `test` — evaluates the saved `best.pth` on the test set and prints test metrics.
3. `cam` — generates GradCAM visualizations for a small subset of validation images.

You may also run the Python modules directly for more control. Example:

```bash
python -m code.train --train_csv dataset/train.csv --val_csv dataset/valid.csv --root_dir dataset/real_vs_fake/real-vs-fake --run_dir models/version_YYYYMMDD_HHMM --batch_size 64 --epochs_head 3 --epochs_finetune 5 --lr_head 1e-3 --lr_backbone 1e-5

# After training, run tests
python -m code.test --test_csv dataset/test.csv --root_dir dataset/real_vs_fake/real-vs-fake --run_dir models/version_YYYYMMDD_HHMM

# Generate GradCAMs
python -m code.cam --val_csv dataset/valid.csv --root_dir dataset/real_vs_fake/real-vs-fake --checkpoint models/version_YYYYMMDD_HHMM/best.pth --outputs_dir outputs/cam
```

> Note: Flag names vary slightly across helper scripts — prefer editing `run.sh` for quick runs.

---

# 7. Training: session-by-session (protocol)

Each training session is designed to be a complete experiment and follows this protocol:

1. Create a new run directory (e.g., `models/version_2025-09-12-2305_expt1/`).
2. Save the run config (CLI args) and `git commit` hash in the run folder.
3. **Phase 1 — Head-only:** freeze backbone weights, train classifier head for `epochs_head` epochs. Save `last.pth` and `best.pth` (within run folder if improved).
4. **Phase 2 — Fine-tune:** unfreeze backbone, train backbone + head with lower backbone learning rate. Use scheduler (cosine with warmup). Save `last.pth` each epoch and `best.pth` if validation AUC improves.
5. At session end: record the run summary in a central `models/model_metadata.json` or `models/registry.csv` (run name, best metric, config path, git hash, timestamp).

**Selecting the global best model**

* A single global best is maintained (e.g., `models/best_model.pth` or by adding an entry in the `model_metadata.json`). If a newly completed run has a better validation AUC than current global best, copy its `best.pth` to `models/best_model.pth` and update `model_metadata.json`.

---

# 8. Checkpointing & model versioning

Per-run (inside `models/version_x/`) store:

* `last.pth` — most recent checkpoint (for resuming)
* `best.pth` — checkpoint with highest validation AUC in this run
* `config.yaml` or `args.txt` — hyperparameters used
* `metrics_log.csv` — per-epoch metrics (loss, val\_loss, auc, acc, etc.)

Global:

* `models/best_model.pth` — global best across all runs (copy of `version_x/best.pth`)
* `models/model_metadata.json` — registry of runs and their best metrics (recommended)

**Atomic saves**: The code saves checkpoints atomically (write to temp file then move) to avoid corruption.

---

# 9. Logging, TensorBoard & session logs

* Each session writes a `run.log` (plain text) under `log_file/<run_name>/` with hyperparameters, start/end times, per-epoch metrics, warnings, list of any missing/corrupt images encountered.
* TensorBoard scalars are written to `log_file/<run_name>/tb/`. To view TensorBoard:

```bash
tensorboard --logdir log_file
# then open http://localhost:6006
```

* For clarity, include in your run logs the first N image paths used in training or validation (the code logs paths per batch optionally). This helps reproduce and debug if a particular image causes problems.

---

# 10. GradCAM (explainability)

GradCAM visualizations are generated after training (optional) and saved under `outputs/<run_name>/cam/`.

Usage example:

```bash
python -m code.cam --checkpoint models/version_.../best.pth --val_csv dataset/valid.csv --root_dir dataset/real_vs_fake/real-vs-fake --outputs_dir outputs/myrun/cam --num_samples 20
```

Notes:

* GradCAM targets the class predicted or can be forced to visualize the "real" class.
* Visualizations include the original image overlaid with the heatmap and predicted probability.

---

# 11. Testing / Final evaluation

After training and selecting the final model, evaluate it exactly once on the held-out test set to get final metrics:

```bash
python -m code.test --test_csv dataset/test.csv --root_dir dataset/real_vs_fake/real-vs-fake --run_dir models/version_x
```

The script loads `best.pth` under `run_dir` by default and prints/writes metrics to `models/version_x/metrics_test.json` (script will do that if implemented).

---

# 12. Reproducibility & environment management

* Always record the git commit hash in the run directory: `git rev-parse --short HEAD > models/version_x/git.txt`.
* Save `pip freeze > models/version_x/requirements.txt` or the `requirements.txt` used to create the environment.
* Use the `seed_everything(seed)` helper at run start to make runs deterministic (note: `cudnn.deterministic=True` may slow training).
* For strict reproducibility, save optimizer and scheduler states in `last.pth` so runs can resume exactly.

---

# 13. Experiment tracking & HPO (optional)

* Use Weights & Biases (W\&B) or MLflow for richer experiment tracking. Integrations are straightforward: initialize `wandb.init()` in `train.py`, log per-epoch metrics, and upload `best.pth` as an artifact.
* For HPO use `optuna` or W\&B Sweeps. Always evaluate candidates on the same validation split.

---

# 14. Maintenance tasks & best practices

Daily / weekly tasks:

* Archive old runs to remote storage (S3, GCS) after verifying the model is promoted or obsolete.
* Monitor disk use and delete `last.pth` of very old runs or compress model folders.
* Keep `model_metadata.json` up to date and clean.
* Re-run unit smoke tests (small-run training) on PRs (CI) to catch regressions.

When changing preprocessing/transformations:

* Re-generate validation/test preprocessed tensors (if you cached them) and note this in the run metadata.
* If you change `img_size` or normalization, previous preprocessed sets are invalid.

Backing up dataset:

* Keep an immutable manifest with checksums (`sha256`) for the dataset. This prevents accidental data drift.

---

# 15. Troubleshooting (common issues)

**Missing images / FileNotFoundError**

* Check `dataset.root` path and `path` column in CSV. Use absolute paths if unsure.
* The code logs missing files and substitutes black images for robustness.

**CUDA OOM (out of memory)**

* Reduce `batch_size` or use gradient accumulation (`accumulate_grad_batches` pattern).
* Use mixed precision (`torch.cuda.amp`) to reduce memory.

**Slow dataloader**

* Increase `num_workers` in `DataLoader` (start with 4, scale up as system allows).
* Consider WebDataset / LMDB if IO is a bottleneck.

**Checkpoint load errors**

* If optimizer state fails to load across PyTorch versions, the loader will warn and continue; fine-tune from model weights only.

**Inconsistent results across runs**

* Ensure seed set via `seed_everything` and record `git` commit & environment. CPU vs GPU differences and cudnn nondeterminism may still cause tiny variations.

---

# 16. Contributing

If you want to contribute:

1. Fork repository and create a feature branch.
2. Add unit tests for any new core behavior (e.g., dataset, scheduler).
3. Run the smoke training test locally (`--smoke_run` mode if implemented).
4. Submit a PR describing the change and impact on training/runtime.

Suggested tests:

* Data loader returns tensors of correct shape.
* Training loop runs 1 epoch without crashing on a tiny subset.
* Checkpoint save/load roundtrip works.

---

# 17. Ethics, privacy & legal

This project involves face images and possibly images of real people. Use extreme caution:

* Make sure you have proper rights to use the dataset (check license and consent).
* Create a `DATA_LICENSE.md` and `PRIVACY_AND_CONSENT.md` that documents allowed uses, consent, and removal procedures.
* Avoid publishing or deploying models trained on private or non-consensual images.
* Restrict dataset & model access to authorized personnel; consider encryption for storage if required.

---

# 18. Contact

If you need help or want additional features (CI, Dockerfile, W\&B integration, ONNX export), open an issue or reach out to the repo owner.

---

Thank you — once you verify this README matches your local file names and script flags, I can:

* generate a `model_metadata.json` template,
* add a `Dockerfile` for reproducible environments, or
* generate CI smoke tests to run on each PR.

Which one would you like next?





Running: 

python -m code.train `    --train_csv dataset/train.csv `    --val_csv dataset/valid.csv `    --root_dir dataset/real_vs_fake/real_vs_fake `    --backbone cvt_13 `    --batch_size 5 `    --epochs_head 1 `    --epochs_finetune 1 `    --lr_head 1e-3 `    --lr_backbone 1e-4 `    --seed 42 `    --run_dir outputs/run1    --pretrained   --do_cam



. venv\Scripts\Activate.ps1




python -m code.train \
    --train_csv dataset/batches/train1.csv \
    --val_csv dataset/batches/valid1.csv \
    --root_dir dataset/real_vs_fake/real_vs_fake \
    --backbone cvt_13 \
    --batch_size 5 \
    --epochs_head 1 \
    --epochs_finetune 1 \
    --lr_head 1e-3 \
    --lr_backbone 1e-4 \
    --seed 42 \
    --run_dir outputs/run1 \
    --pretrained \
    --do_cam





python -m  code.train --train_csv dataset/batches/batch1/train1.csv --val_csv dataset/batches/batch1/valid1.csv --root_dir dataset/real_vs_fake/real_vs_fake --backbone cvt_13 --batch_size 4 --epochs_head 25 --epochs_finetune 15 --lr_head 1e-3 --lr_backbone 1e-4 --seed 42 --run_dir outputs/train_session_1 --pretrained --do_cam




python -m code.train --train_csv dataset/batches/small_batch/train.csv --val_csv dataset/batches/small_batch/valid.csv --root_dir dataset/real_vs_fake/real_vs_fake --backbone cvt_13 --batch_size 32 --epochs_head 10 --epochs_finetune 10 --lr_head 1e-3 --lr_backbone 1e-4 --seed 42 --run_dir outputs/sample_train_1 --pretrained --do_cam



python -m code.train --train_csv dataset/batches/small_batch/train.csv --val_csv dataset/batches/small_batch/valid.csv --root_dir dataset/real_vs_fake/real_vs_fake --backbone cvt_13 --batch_size 32 --epochs_head 5 --epochs_finetune 5 --lr_head 1e-3 --lr_backbone 1e-4 --seed 42 --run_dir outputs/sample_train_1 --pretrained --do_cam


python -m code.train --train_csv dataset/batches/train2.csv --val_csv dataset/batches/valid2.csv --root_dir dataset/real_vs_fake/real_vs_fake --backbone cvt_13 --batch_size 2 --epochs_head 1 --epochs_finetune 1 --lr_head 1e-3 --lr_backbone 1e-4 --seed 42 --run_dir outputs/sample_train_1 --pretrained --do_cam