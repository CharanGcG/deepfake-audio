import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ✅ Adjust this to your dataset root
DATA_ROOT = "C:\Charan Files\deepfake-audio\dataset\LA\LA"


def parse_protocol(protocol_path, audio_base_path):
    """
    Reads ASVspoof2019 protocol file and returns a dataframe
    containing file path and label (bonafide/spoof).
    """
    rows = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            speaker_id = parts[0]
            file_name = parts[1]
            system_id = parts[3]
            key = parts[4]  # 'bonafide' or 'spoof'

            audio_path = os.path.join(audio_base_path, file_name + ".flac")
            rows.append({
                "speaker_id": speaker_id,
                "file_name": file_name,
                "system_id": system_id,
                "label": 1 if key == "bonafide" else 0,
                "label_text": key,
                "path": audio_path
            })
    return pd.DataFrame(rows)

# Example usage:
train_protocol = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
dev_protocol   = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt")

train_df = parse_protocol(train_protocol, os.path.join(DATA_ROOT, "ASVspoof2019_LA_train"))
dev_df   = parse_protocol(dev_protocol,   os.path.join(DATA_ROOT, "ASVspoof2019_LA_dev"))

print(train_df.head())
print(f"Train samples: {len(train_df)}, Dev samples: {len(dev_df)}")
