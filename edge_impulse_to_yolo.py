import json
import os
from PIL import Image

# ===== CONFIG =====
EDGE_DATASET_DIR = r"C:\Users\kfloresm\Downloads\change\dataset"   # carpeta original
YOLO_DATASET_DIR = "dataset_yolo"    # carpeta destino
LABELS_FILE = os.path.join(EDGE_DATASET_DIR, "info.labels")

CLASSES = [
    "STOP",
    "Crossing",
    "TurnL",
    "TurnR",
    "AOnly",
    "Give",
    "Round"
]

# ==================

os.makedirs(f"{YOLO_DATASET_DIR}/images/train", exist_ok=True)
os.makedirs(f"{YOLO_DATASET_DIR}/images/val", exist_ok=True)
os.makedirs(f"{YOLO_DATASET_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{YOLO_DATASET_DIR}/labels/val", exist_ok=True)

with open(LABELS_FILE, "r") as f:
    data = json.load(f)

for item in data["files"]:
    img_rel_path = item["path"]              # testing/xxx.jpg....
    split = item["category"]                 # training o testing
    bboxes = item.get("boundingBoxes", [])

    img_path = os.path.join(EDGE_DATASET_DIR, img_rel_path)

    if not os.path.exists(img_path):
        print(f"⚠ Imagen no encontrada: {img_path}")
        continue

    # Abrir imagen para obtener tamaño real
    img = Image.open(img_path)
    img_w, img_h = img.size

    # Nombre base sin hashes raros
    base_name = os.path.basename(img_rel_path)
    base_name = base_name.split(".jpg")[0]   # 000009

    # Train / Val
    if split == "training":
        img_out = f"{YOLO_DATASET_DIR}/images/train/{base_name}.jpg"
        lbl_out = f"{YOLO_DATASET_DIR}/labels/train/{base_name}.txt"
    else:
        img_out = f"{YOLO_DATASET_DIR}/images/val/{base_name}.jpg"
        lbl_out = f"{YOLO_DATASET_DIR}/labels/val/{base_name}.txt"

    # Copiar imagen
    img.convert("RGB").save(img_out)

    # Crear archivo YOLO
    with open(lbl_out, "w") as f:
        for box in bboxes:
            label = box["label"]
            if label not in CLASSES:
                continue

            class_id = CLASSES.index(label)

            x = box["x"]
            y = box["y"]
            w = box["width"]
            h = box["height"]

            # YOLO format
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("✅ Conversión completada")