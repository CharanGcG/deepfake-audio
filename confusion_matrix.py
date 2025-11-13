import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, classification_report
import matplotlib.pyplot as plt

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
        label_str = parts[2].lower()
        score = float(parts[3])
        
        utts.append(utt)
        labels.append(0 if label_str == "bonafide" else 1)  # bonafide=0, spoof=1
        scores.append(score)

scores = np.array(scores)
labels = np.array(labels)

# Invert score direction if spoof scores < bonafide scores
if np.mean(scores[labels == 1]) < np.mean(scores[labels == 0]):
    scores = -scores
    print("Score direction inverted for evaluation.")

# Compute ROC to get threshold at EER
fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
fnr = 1 - tpr

eer_idx = np.nanargmin(np.abs(fpr - fnr))
eer_threshold = thresholds[eer_idx]
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

print(f"\nEER ≈ {eer*100:.3f}%")
print(f"EER threshold = {eer_threshold:.6f}\n")

# Predict using EER threshold
pred = (scores >= eer_threshold).astype(int)

# Confusion matrix
cm = confusion_matrix(labels, pred)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix (at EER threshold):")
print(cm)

print("\nDetailed classification report:")
print(classification_report(labels, pred, target_names=["bonafide (0)", "spoof (1)"]))

# Plot confusion matrix heatmap
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix at EER Threshold")
plt.xticks([0,1], ["Pred: Real(0)", "Pred: Fake(1)"])
plt.yticks([0,1], ["True: Real(0)", "True: Fake(1)"])
plt.colorbar()

# Label counts
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)

plt.tight_layout()
plt.savefig("confusion_matrix.png")

plt.show()
