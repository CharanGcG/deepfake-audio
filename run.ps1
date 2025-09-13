# -------------------- run.ps1 --------------------
$ErrorActionPreference = "Stop"

# Paths
$DATASET_DIR = ".\dataset"
$TRAIN_CSV = "$DATASET_DIR\train.csv"
$VAL_CSV   = "$DATASET_DIR\valid.csv"
$TEST_CSV  = "$DATASET_DIR\test.csv"
$ROOT_DIR  = "$DATASET_DIR\real_vs_fake\real-vs-fake"

# Training
python -m code.train `
  --train_csv $TRAIN_CSV `
  --val_csv $VAL_CSV `
  --root_dir $ROOT_DIR `
  --batch_size 64 `
  --epochs_head 1 `
  --epochs_finetune 0 `
  --img_size 256 `
  --lr_head 1e-4 `
  --lr_backbone 1e-5 `
  --run_dir .\models

# Testing
python -m code.test `
  --test_csv $TEST_CSV `
  --root_dir $ROOT_DIR `
  --checkpoint .\models\best.pth `
  --batch_size 64 `
  --img_size 256

# GradCAM
python -m code.cam `
  --val_csv $VAL_CSV `
  --root_dir $ROOT_DIR `
  --checkpoint .\models\best.pth `
  --outputs_dir .\outputs\cam_results `
  --num_samples 10
