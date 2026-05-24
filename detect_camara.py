import cv2
from ultralytics import YOLO

# ===== CARGAR MODELO ENTRENADO =====
model = YOLO("runs/detect/train/weights/best.pt")

# ===== SELECCIONAR CAMARA =====
# 0 = webcam laptop
# 1 = USB externa
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("Deteccion de Señales", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()