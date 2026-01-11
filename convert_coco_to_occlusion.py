import os
import cv2
from tqdm import tqdm

# ---------- CONFIG ----------
SRC_IMG_DIR = "datasets/coco128/images/train2017"
SRC_LBL_DIR = "datasets/coco128/labels/train2017"

DST_IMG_DIR = "datasets/coco_occlusion/images/train2017"
DST_LBL_DIR = "datasets/coco_occlusion/labels/train2017"

os.makedirs(DST_IMG_DIR, exist_ok=True)
os.makedirs(DST_LBL_DIR, exist_ok=True)
# ----------------------------


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def compute_visibility(boxes):
    vis = []
    for i, b1 in enumerate(boxes):
        x1, y1, x2, y2 = b1
        area = max(1.0, (x2 - x1) * (y2 - y1))
        occluded = 0.0

        for j, b2 in enumerate(boxes):
            if i == j:
                continue
            xx1 = max(x1, b2[0])
            yy1 = max(y1, b2[1])
            xx2 = min(x2, b2[2])
            yy2 = min(y2, b2[3])

            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            occluded += inter

        v = 1.0 - (occluded / area)
        v = max(0.05, min(1.0, v))
        vis.append(v)

    return vis


# ---------- MAIN LOOP ----------
label_files = [f for f in os.listdir(SRC_LBL_DIR) if f.endswith(".txt")]

for lbl_file in tqdm(label_files):
    img_file = lbl_file.replace(".txt", ".jpg")
    img_path = os.path.join(SRC_IMG_DIR, img_file)
    lbl_path = os.path.join(SRC_LBL_DIR, lbl_file)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    classes, yolo_boxes = [], []

    with open(lbl_path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            classes.append(int(parts[0]))
            yolo_boxes.append(parts[1:5])

    if len(yolo_boxes) == 0:
        continue

    xyxy_boxes = [
        yolo_to_xyxy(cx, cy, bw, bh, w, h)
        for cx, cy, bw, bh in yolo_boxes
    ]

    visibilities = compute_visibility(xyxy_boxes)

    # write new label file
    dst_lbl_path = os.path.join(DST_LBL_DIR, lbl_file)
    with open(dst_lbl_path, "w") as f:
        for cls, box, v in zip(classes, yolo_boxes, visibilities):
            cx, cy, bw, bh = box
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {v:.4f}\n")

print("✅ COCO train2017 conversion completed.")
