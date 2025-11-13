import matplotlib.pyplot as plt
import numpy as np

scores = []
labels = []

with open(r"C:\Charan Files\deepfake-audio\eval_scores_using_best_dev_model.txt") as f:
    for line in f:
        parts = line.strip().split()
        score = float(parts[-1])
        label_str = parts[2]
        label = 0 if label_str == "bonafide" else 1  # Map bonafide to 0 (Real), spoof to 1 (Fake)
        scores.append(score)
        labels.append(label)

scores = np.array(scores)
labels = np.array(labels)

plt.hist(scores[labels == 0], bins=50, alpha=0.6, label="Real")
plt.hist(scores[labels == 1], bins=50, alpha=0.6, label="Fake")
plt.legend()
plt.title("AASIST Score Distribution")
plt.xlabel("Score")
plt.ylabel("Count")
plt.savefig("score_histogram.png")
plt.show()
