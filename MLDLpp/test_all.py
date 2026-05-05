import requests
import os

url = 'http://127.0.0.1:5000/predict'

test_images = ['test_6.png', 'test_7.png', 'test_8.png']

for img_name in test_images:
    img_path = os.path.join(r'd:\MLDLpp', img_name)
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        continue
        
    with open(img_path, 'rb') as f:
        files = {'file': f}
        try:
            r = requests.post(url, files=files)
            print(f"Testing {img_name}:")
            print(f"  Status Code: {r.status_code}")
            print(f"  Response: {r.text}")
        except Exception as e:
            print(f"  Error testing {img_name}: {e}")

# Test with a non-digit image (if exists) or an empty one
# I'll create a blank image for testing the "no digit" case
import cv2
import numpy as np

blank_path = r'd:\MLDLpp\blank.png'
cv2.imwrite(blank_path, np.zeros((100, 100), dtype=np.uint8))

with open(blank_path, 'rb') as f:
    files = {'file': f}
    r = requests.post(url, files=files)
    print(f"Testing blank.png:")
    print(f"  Status Code: {r.status_code}")
    print(f"  Response: {r.text}")

os.remove(blank_path)
