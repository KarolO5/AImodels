from ultralytics import YOLO
import matplotlib.pyplot as plt

# ===== CARGAR MODELO =====
model = YOLO("yolov8n.pt")

# ===== ENTRENAMIENTO =====
results = model.train(
    data="C:/dataset_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    plots=True
)

# ===== EVALUACION =====
metrics = model.val()

print("\n===== METRICAS =====")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")

# ===== MOSTRAR MATRIZ =====
metrics.confusion_matrix.plot(save_dir=".")
print("✅ Se generó matriz de confusión")
