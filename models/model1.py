import torch

# Actual model class definitions


class Model1(torch.nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=2, padding="same"),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28 * 10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.layers(x)
