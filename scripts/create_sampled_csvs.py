import os
import pandas as pd

new_train_path = "train.csv"
new_valid_path = "valid.csv"

def create_sampled_csvs(
    train_csv_path="dataset/train.csv",
    valid_csv_path="dataset/valid.csv",
    output_dir="dataset/batches/small_batch",
    total_train_images=100,
    total_valid_images=20,
    train_offset=0,
    valid_offset=0
):
    """
    Create sampled CSV files from main train and valid CSVs.
    
    Args:
        train_csv_path (str): Path to main train CSV.
        valid_csv_path (str): Path to main valid CSV.
        output_dir (str): Directory to save sampled CSVs.
        total_train_images (int): Total images in sampled train CSV.
        total_valid_images (int): Total images in sampled valid CSV.
        train_offset (int): Offset for starting point in train CSV.
        valid_offset (int): Offset for starting point in valid CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Train CSV ---
    train_df = pd.read_csv(train_csv_path)
    half_train = total_train_images // 2

    # Real and fake slices
    real_train_start = train_offset
    fake_train_start = 50000 + train_offset  # assuming first 50k real, next 50k fake
    real_train = train_df.iloc[real_train_start : real_train_start + half_train]
    fake_train = train_df.iloc[fake_train_start : fake_train_start + half_train]

    sampled_train = pd.concat([real_train, fake_train])
    sampled_train = sampled_train.sample(frac=1).reset_index(drop=True)  # shuffle rows
    train_output_path = os.path.join(output_dir, new_train_path)
    sampled_train.to_csv(train_output_path, index=False)
    print(f"Sampled train CSV saved to {train_output_path}")

    # --- Valid CSV ---
    valid_df = pd.read_csv(valid_csv_path)
    half_valid = total_valid_images // 2

    real_valid_start = valid_offset
    fake_valid_start = 10000 + valid_offset  # assuming first 10k real, next 10k fake
    real_valid = valid_df.iloc[real_valid_start : real_valid_start + half_valid]
    fake_valid = valid_df.iloc[fake_valid_start : fake_valid_start + half_valid]

    sampled_valid = pd.concat([real_valid, fake_valid])
    sampled_valid = sampled_valid.sample(frac=1).reset_index(drop=True)  # shuffle rows
    valid_output_path = os.path.join(output_dir, new_valid_path)
    sampled_valid.to_csv(valid_output_path, index=False)
    print(f"Sampled valid CSV saved to {valid_output_path}")


# Example usage:
if __name__ == "__main__":
    create_sampled_csvs(
        total_train_images=100,
        total_valid_images=20,
        train_offset=0,
        valid_offset=0
    )
