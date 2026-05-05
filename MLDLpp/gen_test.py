import cv2
import numpy as np

# Create a 512x512 white image
img7 = np.ones((512, 512), dtype=np.uint8) * 255
# Draw a 7
cv2.line(img7, (150, 150), (350, 150), 0, 15)
cv2.line(img7, (350, 150), (250, 400), 0, 15)
cv2.imwrite("test_7.png", img7)

img8 = np.ones((512, 512), dtype=np.uint8) * 255
# Draw an 8 (two circles)
cv2.circle(img8, (256, 200), 70, 0, 15)
cv2.circle(img8, (256, 320), 80, 0, 15)
cv2.imwrite("test_8.png", img8)
