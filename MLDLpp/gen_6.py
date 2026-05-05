import cv2
import numpy as np

# Create a 6 with a hole
img = np.ones((512, 512), dtype=np.uint8) * 255
cv2.circle(img, (256, 300), 60, 0, 15)
cv2.line(img, (200, 300), (256, 100), 0, 15)
cv2.imwrite("test_6.png", img)
