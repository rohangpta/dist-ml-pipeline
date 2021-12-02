import json

import boto3
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from models.model1 import Model1

train_data = datasets.MNIST(
    root="../data",
    train=True,
    transform=ToTensor(),
    download=False,
)
test_data = datasets.MNIST(
    root="../data", train=False, transform=ToTensor(), download=False
)

loaders = {
    "train": DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0),
    "test": DataLoader(test_data, batch_size=100, shuffle=True, num_workers=0),
}


def train(num_epochs, model, loaders, optimizer):
    model.train()
    total_step = len(loaders["train"])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders["train"]):
            pred = model(images)
            loss = F.cross_entropy(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i+1}/{total_step}], "
                    f"Loss: {loss.item()}"
                )


def test(model, loaders):
    model.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    test_loader = loaders["test"]
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )


def main():
    creds = json.load(open("/run/secrets/aws_creds"))
    client = boto3.client(
        "s3",
        aws_access_key_id=creds["access"],
        aws_secret_access_key=creds["secret_access"],
        region_name="ap-south-1",
    )

    LEARNING_RATE = 0.01
    N_EPOCHS = 10
    MOMENTUM = 0.5

    model1 = Model1()
    optimizer = torch.optim.SGD(
        model1.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
    )
    train(N_EPOCHS, model1, loaders, optimizer)
    torch.save(model1.state_dict(), "mnist_model1.pt")
    with open("mnist_model1.pt", "rb") as f:
        client.upload_fileobj(f, "ml-model-188-project", "mnist_model1.pt")


if __name__ == "__main__":
    main()
