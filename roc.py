import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

scores = []
labels = []

with open(r"C:\Charan Files\deepfake-audio\eval_scores_using_best_dev_model.txt") as f:
    for line in f:
        parts = line.strip().split()
        
        # parts structure:
        # [utt_id, attack_id, label, score]
        label_str = parts[2]  # 'bonafide' or 'spoof'
        score = float(parts[3])
        
        label = 0 if label_str == "bonafide" else 1  # bonafide=0, spoof=1
        
        scores.append(score)
        labels.append(label)

scores = np.array(scores)
labels = np.array(labels)

scores = -scores

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

# Plot
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - AASIST Scores")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.show()
