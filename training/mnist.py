import argparse
import os
import boto3
import torch
import torch.nn.functional as F
import torch.distributed as dist
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
    client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        region_name="us-east-1",
    )

    # Recieve hyperparameter arguments on runtime

    parser = argparse.ArgumentParser(description="Hyperparams Parsing")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="N",
        help="Default Learning Rate for Training (default: 0.01)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="Batch Size (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="Number of Epochs to Train for (default: 1)",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )

    args = parser.parse_args()

    LEARNING_RATE = args.lr
    N_EPOCHS = args.epochs
    MOMENTUM = args.momentum
    BATCH_SIZE = args.batch_size
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

    loaders = {
        "train": DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        ),
        "test": DataLoader(
            test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        ),
    }

    # Distribute training

    if WORLD_SIZE > 1 and dist.is_available():
        # Use CPU backend (NOTE: ideally this would be GPU but I'm poor)
        dist.init_process_group(backend=dist.Backend.GLOO)

    model = Model1().to(torch.device("cpu"))

    if dist.is_initialized and dist.is_available():
        Distributor = torch.nn.parallel.DistributedDataParallel
        model = Distributor(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    train(N_EPOCHS, model, loaders, optimizer)

    # Only save the master node's model

    if int(os.getenv("RANK")) == 0:
        torch.save(model.module.state_dict(), "mnist_model1.pt")
        with open("mnist_model1.pt", "rb") as f:
            client.upload_fileobj(f, "ml-model-188-project", "mnist_model1.pt")
    else:
        print("Worker node successfully completed, exiting...")


if __name__ == "__main__":
    main()
