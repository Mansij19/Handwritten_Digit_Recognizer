import os
import numpy as np
import cv2

images = [f for f in os.listdir() if f.startswith('out_')]
for img_name in sorted(images):
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    print(f"\n--- {img_name} ---")
    for row in range(0, 28, 2):
        s = ""
        for col in range(28):
            s += "#" if img[row, col] > 127 else "."
        print(s)
