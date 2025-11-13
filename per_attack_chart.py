import matplotlib.pyplot as plt

attacks = ["A07","A08","A09","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]
eers = [0.529676029, 0.424416206, 0.0, 0.855630509, 0.179950347,
        0.709626376, 0.146004133, 0.162977240, 0.553447233,
        0.651908959, 1.263073608, 2.607635837, 0.651908959]

plt.figure(figsize=(10, 5))
plt.bar(attacks, eers, color='skyblue')
plt.title("Per-Attack Equal Error Rate (EER) for AASIST (%)")
plt.xlabel("Attack Type (A07 - A19)")
plt.ylabel("EER (%)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig("per_attack_eer.png")
plt.show()
