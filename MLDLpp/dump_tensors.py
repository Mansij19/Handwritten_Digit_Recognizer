import cv2
import numpy as np
import os
from app import preprocess_image_candidates

images = ["test_6.png", "test_7.png", "test_8.png"]
for img_name in images:
    if os.path.exists(img_name):
        with open(img_name, "rb") as f:
            tensors = preprocess_image_candidates(f.read())
        for i, tensor in enumerate(tensors):
            # tensor is (1, 28, 28, 1) float32
            out_img = (tensor[0, :, :, 0] * 255).astype(np.uint8)
            cv2.imwrite(f"out_{img_name}_{i}.png", out_img)
