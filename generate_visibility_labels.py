import os
import cv2
import numpy as np

def yolo_to_xyxy(label, img_w, img_h):
    cx, cy, w, h = label
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]

def compute_visibility(boxes):
    visibilities = []

    for i, box_i in enumerate(boxes):
        x1_i, y1_i, x2_i, y2_i = box_i
        area_i = max(1.0, (x2_i - x1_i) * (y2_i - y1_i))
        occluded = 0.0

        for j, box_j in enumerate(boxes):
            if i == j:
                continue

            x1_j, y1_j, x2_j, y2_j = box_j

            xA = max(x1_i, x1_j)
            yA = max(y1_i, y1_j)
            xB = min(x2_i, x2_j)
            yB = min(y2_i, y2_j)

            inter_area = max(0, xB - xA) * max(0, yB - yA)
            occluded += inter_area

        visibility = 1.0 - (occluded / area_i)
        visibility = max(0.05, min(1.0, visibility))
        visibilities.append(visibility)

    return visibilities
