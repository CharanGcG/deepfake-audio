# -------------------- run.sh --------------------
#!/bin/bash


# Exit if any command fails
set -e


# Default values
DATASET_DIR="./dataset"
TRAIN_CSV="${DATASET_DIR}/train.csv"
VAL_CSV="${DATASET_DIR}/valid.csv"
TEST_CSV="${DATASET_DIR}/test.csv"
ROOT_DIR="${DATASET_DIR}/real_vs_fake/real-vs-fake"


# Training
python -m code.train \
--train_csv $TRAIN_CSV \
--val_csv $VAL_CSV \
--root_dir $ROOT_DIR \
--batch_size 64 \
--epochs 10 \
--img_size 224 \
--lr 1e-4 \
--log_dir ./log_file \
--models_dir ./models \
--outputs_dir ./outputs


# Testing
python -m code.test \
--test_csv $TEST_CSV \
--root_dir $ROOT_DIR \
--checkpoint ./models/best_model.pth \
--batch_size 64 \
--img_size 224


# GradCAM
python -m code.cam \
--val_csv $VAL_CSV \
--root_dir $ROOT_DIR \
--checkpoint ./models/best_model.pth \
--outputs_dir ./outputs/cam_results \
--num_samples 10