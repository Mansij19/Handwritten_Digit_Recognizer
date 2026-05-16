import requests
import json
url = 'http://127.0.0.1:5000/predict'
files = {'file': open('test_8.png', 'rb')}
r = requests.post(url, files=files)
print(r.status_code)
print(r.text)
