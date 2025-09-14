import os
import pandas as pd

def check_image_exists(csv_path: str, root_dir: str, index: int = 0):
    """
    Checks if the image at a given CSV row index exists on disk.

    Args:
        csv_path: Path to the CSV file containing 'path' column.
        root_dir: Root directory where images are stored.
        index: Row index in CSV to check image path (default 0).
    """
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError("CSV must contain a 'path' column")

    rel_path = df.iloc[index]["path"]
    full_path = os.path.normpath(os.path.join(root_dir, rel_path))

    if os.path.isfile(full_path):
        print(f"Image FOUND at row {index}: {full_path}")
    else:
        print(f"Image MISSING at row {index}: {full_path}")

# Example usage
if __name__ == "__main__":
    csv_file = "dataset/batches/train1.csv"  # Change to your CSV file path
    dataset_root = "dataset/real_vs_fake/real_vs_fake"  # Change to your dataset root dir
    row_to_check = 0  # Change to desired row index to check

    check_image_exists(csv_file, dataset_root, row_to_check)
