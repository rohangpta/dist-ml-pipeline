import io
import json

import boto3
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps
from torchvision.transforms import transforms
import os
from models.model1 import Model1

app = FastAPI()

client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("ACCESS_KEY"),
    aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
    region_name="us-east-1",
)


@app.get("/")
def landing():
    return {"Welcome": "User"}


@app.post("/predict")
def predict_mnist(raw_image: UploadFile = File(...)):
    """
    We take in a raw image that is uploaded and we return the prediction our model gives on this image.
    """
    image = preprocess_image_mnist(raw_image)
    if image is None:
        raise HTTPException(status_code=404)

    # Load latest model from s3

    client.download_file("ml-model-188-project", "mnist_model1.pt", "mnist_model1.pt")
    model = Model1()
    model.load_state_dict(torch.load("mnist_model1.pt"))
    model.eval()

    # Predict
    output = model(image)
    prediction = int(torch.max(output.data, 1)[1].numpy())
    return {"Prediction": prediction}


def preprocess_image_mnist(img):
    """
    We do some basic preprocessing on the image, allowing our
    model to accept coloured numbers drawn on white backgrounds with minimal noise
    """
    try:
        img = Image.open(io.BytesIO(img.file.read()))
    except Exception:
        return None

    # Greyscale and Resize
    img = ImageOps.grayscale(img).resize((28, 28))

    # Invert
    img = ImageOps.invert(img)

    img.show()
    trans = transforms.ToTensor()
    return trans(img).view(1, 1, 28, 28)
