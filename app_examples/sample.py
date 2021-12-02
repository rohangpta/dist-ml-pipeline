import sys

import requests

with open("app_examples/imgs/four.jpg", "rb") as f:
    r = requests.post(
        url="http://0.0.0.0:8000/predict",
        files={"raw_image": (f.name, f, "multipart/form-data")},
    )
    assert r.json()["Prediction"] == 4


with open("app_examples/imgs/three.jpg", "rb") as f:
    r = requests.post(
        url="http://0.0.0.0:8000/predict",
        files={"raw_image": (f.name, f, "multipart/form-data")},
    )
    assert r.json()["Prediction"] == 3

sys.stdout.write("Success!")
