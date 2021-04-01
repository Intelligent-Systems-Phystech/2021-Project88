from torch import nn


class Convolutional3Layers(nn.Module):
    """
        Нейронная сеть с применением сверточных слоев
    """

    def __init__(self):
        super(Convolutional3Layers, self).__init__()

        self.conv1 = nn.Conv1d(2, 10, 5, 2, 2)
        self.norm2 = nn.BatchNorm1d(10)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(10, 10, 5, 2, 2)
        self.norm3 = nn.BatchNorm1d(10)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv1d(10, 20, 5, 2, 2)
        self.norm4 = nn.BatchNorm1d(20)
        self.relu4 = nn.ReLU()
        self.conv4 = nn.Conv1d(20, 5, 5, 2, 2)
        self.norm5 = nn.BatchNorm1d(5)
        self.relu5 = nn.ReLU()
        self.flat = nn.Flatten()
        self.linear5 = nn.Linear(100, 30)
        self.norm6 = nn.BatchNorm1d(30)
        self.relu6 = nn.ReLU()
        self.linear6 = nn.Linear(30, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.relu2(self.norm2(x)))
        x = self.conv3(self.relu3(self.norm3(x)))
        x = self.conv4(self.relu4(self.norm4(x)))
        x = self.linear5(self.flat(self.relu5(self.norm5(x))))
        x = self.linear6(self.relu6(self.norm6(x)))

        return x


class Linear2Layers(nn.Module):
    """
        Нейронная сеть с применением линейных слоев
    """

    def __init__(self):
        super(Linear2Layers, self).__init__()

        self.flat = nn.Flatten()
        self.norm1 = nn.BatchNorm1d(640)
        self.linear1 = nn.Linear(640, 100)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm1d(100)
        self.linear2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.linear1(self.norm1(self.flat(x)))
        x = self.linear2(self.norm2(self.relu2(x)))

        return x
