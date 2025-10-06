import os
import pandas as pd

# ✅ Update this path to your dataset root
DATA_ROOT = r"C:\Charan Files\deepfake-audio\dataset\LA\LA"
OUTPUT_ROOT = r"C:\Charan Files\deepfake-audio\dataset"

def parse_protocol(protocol_path, audio_base_path):
    """
    Reads ASVspoof2019 protocol file and returns a dataframe
    containing file path and label (bonafide/spoof).
    """
    rows = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:  # skip malformed lines
                continue
            speaker_id = parts[0]
            file_name = parts[1]
            system_id = parts[3]
            key = parts[4]  # 'bonafide' or 'spoof'

            audio_path = os.path.join(audio_base_path, "flac", file_name + ".flac")
            rows.append({
                "speaker_id": speaker_id,
                "file_name": file_name,
                "system_id": system_id,
                "label": 1 if key == "bonafide" else 0,
                "label_text": key,
                "path": audio_path
            })
    return pd.DataFrame(rows)

# === Parse train/dev (and eval if needed)
protocols_dir = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols")

train_protocol = os.path.join(protocols_dir, "ASVspoof2019.LA.cm.train.trn.txt")
dev_protocol   = os.path.join(protocols_dir, "ASVspoof2019.LA.cm.dev.trl.txt")
eval_protocol  = os.path.join(protocols_dir, "ASVspoof2019.LA.cm.eval.trl.txt")

train_df = parse_protocol(train_protocol, os.path.join(DATA_ROOT, "ASVspoof2019_LA_train"))
dev_df   = parse_protocol(dev_protocol,   os.path.join(DATA_ROOT, "ASVspoof2019_LA_dev"))
eval_df  = parse_protocol(eval_protocol,  os.path.join(DATA_ROOT, "ASVspoof2019_LA_eval"))

# === Save as CSV
output_dir = os.path.join(OUTPUT_ROOT, "metadata_csv")
os.makedirs(output_dir, exist_ok=True)

train_csv = os.path.join(output_dir, "train.csv")
dev_csv   = os.path.join(output_dir, "valid.csv")
eval_csv  = os.path.join(output_dir, "test.csv")

train_df.to_csv(train_csv, index=False)
dev_df.to_csv(dev_csv, index=False)
eval_df.to_csv(eval_csv, index=False)

print(f"✅ CSVs saved to {output_dir}")
print(f"Train samples: {len(train_df)}, Dev samples: {len(dev_df)}, Eval samples: {len(eval_df)}")
