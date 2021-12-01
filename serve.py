from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageOps
import io
import torch
from models.model1 import Model1
from torchvision.transforms import transforms

app = FastAPI()


@app.post("/predict")
def predict_mnist(raw_image: UploadFile = File(...)):
    image = preprocess_image_mnist(raw_image)
    if image is None:
        raise HTTPException(status_code=404)

    # Load model
    model = Model1()
    model.load_state_dict(torch.load("mnist_model.pt"))
    model.eval()

    # Predict
    output = model(image)
    prediction = int(torch.max(output.data, 1)[1].numpy())
    print(output)
    return {"Prediction": prediction}


def preprocess_image_mnist(img):
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
