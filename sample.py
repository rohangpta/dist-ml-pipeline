import requests


f = open("three.jpg", "rb")
files = {"raw_image": (f.name, f, "multipart/form-data")}
r = requests.post(url="http://0.0.0.0:8000/predict", files=files)

print(r.text)
