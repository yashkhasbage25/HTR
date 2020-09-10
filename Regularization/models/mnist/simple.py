import torch.nn as nn

class SimpleMNIST(nn.Module):

    def __init__(self):
        super(SimpleMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lin3 = nn.Linear(64 * 6 * 6, 128)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(128, 10)
        # self.smax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.lin3(x)
        x = self.relu3(x)
        x = self.lin4(x)
        # x = self.smax(x)
        return x