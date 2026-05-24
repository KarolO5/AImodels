import matplotlib.pyplot as plt
import numpy as np

# ===== EJEMPLO (reemplaza con tus resultados) =====
models = ["YOLO", "Edge Impulse"]

accuracy = [0.92, 0.81]
precision = [0.90, 0.78]
recall = [0.88, 0.75]
mAP = [0.89, 0.72]

x = np.arange(len(models))

# ===== GRAFICA 1 =====
plt.figure(figsize=(10,6))
plt.bar(x - 0.15, accuracy, width=0.3, label="Accuracy")
plt.bar(x + 0.15, precision, width=0.3, label="Precision")

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Comparación Accuracy vs Precision")
plt.legend()
plt.grid()

plt.savefig("accuracy_precision.png")
plt.show()

# ===== GRAFICA 2 =====
plt.figure(figsize=(10,6))
plt.bar(models, recall, color='orange', label="Recall")
plt.bar(models, mAP, color='green', alpha=0.7, label="mAP")

plt.title("Recall vs mAP")
plt.legend()
plt.grid()

plt.savefig("recall_map.png")
plt.show()