import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F


train_data = datasets.MNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    download=True,
)
test_data = datasets.MNIST(root="data", train=False, transform=ToTensor())


class Model1(torch.nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=2, padding="same"),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28 * 5, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.layers(x)


loaders = {
    "train": DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0),
    "test": DataLoader(test_data, batch_size=100, shuffle=True, num_workers=0),
}

model1 = Model1()
loss_func = torch.nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model1.parameters(), lr=learning_rate, momentum=0.5)


def train(num_epochs, model, loaders, loss_function):
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
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}"
                )
    torch.save(model.state_dict(), "/mnist_model.pt")


def test():
    model1.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    test_loader = loaders["test"]
    with torch.no_grad():
        for data, target in test_loader:
            output = model1(data)
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


train(3, model1, loaders)
test()
