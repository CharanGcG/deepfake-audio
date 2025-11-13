import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve
from sklearn.metrics import roc_curve
import os

path = r"C:\Charan Files\deepfake-audio\eval_scores_using_best_dev_model.txt"

scores = []
labels = []
utts = []

with open(path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        utt = parts[0]
        label_str = parts[2].lower()   # 'bonafide' or 'spoof'
        score = float(parts[3])

        utts.append(utt)
        labels.append(0 if label_str == "bonafide" else 1)  # 0=real, 1=spoof
        scores.append(score)

scores = np.array(scores)
labels = np.array(labels)

# If needed, invert score direction so that higher score => positive class (spoof)
# We expect spoofs to have higher mean after this transform.
if np.mean(scores[labels == 1]) < np.mean(scores[labels == 0]):
    scores = -scores
    inverted = True
else:
    inverted = False

print(f"Scores inverted: {inverted}")
print("Counts: real =", int((labels==0).sum()), " spoof =", int((labels==1).sum()))
print("Real range:", scores[labels==0].min(), "to", scores[labels==0].max())
print("Spoof range:", scores[labels==1].min(), "to", scores[labels==1].max())

# DET (FPR vs FNR)
fpr, fnr, thr = det_curve(labels, scores, pos_label=1)

# Find EER: where FPR and FNR cross (min abs difference)
eer_idx = np.nanargmin(np.abs(fpr - fnr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
eer_threshold = thr[eer_idx]

print(f"EER ≈ {eer:.6f} (index {eer_idx}), threshold at EER = {eer_threshold:.6f}")
print("FPR at EER =", fpr[eer_idx], " FNR at EER =", fnr[eer_idx])

# Plot DET curve
plt.figure(figsize=(7,6))
plt.plot(fpr * 100, fnr * 100, lw=2, label="DET curve")
plt.scatter([fpr[eer_idx] * 100], [fnr[eer_idx] * 100], color='red', zorder=5,
            label=f"EER = {eer*100:.3f}%\nthr = {eer_threshold:.4f}")
plt.xlabel("False Positive Rate (%)")
plt.ylabel("False Negative Rate (%)")
plt.title("DET Curve - AASIST")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="upper right")
plt.xlim(0, max(1.0, fpr.max())*100)
plt.ylim(0, max(1.0, fnr.max())*100)

outname = "det_curve.png"
plt.tight_layout()
plt.savefig(outname, dpi=150)
plt.show()

print(f"Saved DET plot to {os.path.abspath(outname)}")
