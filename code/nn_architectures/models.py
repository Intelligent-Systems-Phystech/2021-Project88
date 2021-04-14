from torch import nn

class ConvBlock(nn.Module):
    """
        Вспомогательный сверточный блок: Convolutional -> BatchNormalization -> ReLU
    """
    
    def __init__(self, input_ch, output_ch, *args):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv1d(input_ch, output_ch, *args)
        self.norm = nn.BatchNorm1d(output_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class LinearBlock(nn.Module):
    """
        Вспомогательный линейный блок: Linear -> BatchNorm -> ReLU
    """
    
    def __init__(self, input_ch, output_ch):
        super(LinearBlock, self).__init__()
        
        self.linear = nn.Linear(input_ch, output_ch)
        self.norm = nn.BatchNorm1d(output_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.norm(self.linear(x)))
    

class Convolutional5Layers(nn.Module):
    """
        Нейронная сеть с применением сверточных слоев
    """

    def __init__(self):
        super(Convolutional5Layers, self).__init__()
        
        self.conv1 = ConvBlock(2, 32, 3, 1, 1)
        self.conv2 = ConvBlock(32, 64, 5, 2, 2)
        self.conv3 = ConvBlock(64, 64, 5, 2, 2)
        self.conv4 = ConvBlock(64, 64, 5, 2, 2)
        self.conv5 = ConvBlock(64, 5, 5, 2, 2)
        self.flat = nn.Flatten()
        self.linear6 = LinearBlock(100, 30)
        self.linear7 = nn.Linear(30, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.linear6(self.flat(x))
        x = self.linear7(x)

        return x

    
class Convolutional(nn.Module):
    """
        Нейронная сеть с применением сверточных слоев
    """

    def __init__(self):
        super(Convolutional, self).__init__()

        self.conv1 = ConvBlock(2, 8, 3, 1, 1)
        self.conv2 = ConvBlock(8, 16, 5, 2, 2)
        self.conv3 = ConvBlock(16, 32, 5, 2, 2)
        self.conv4 = ConvBlock(32, 64, 5, 2, 2)
        self.conv5 = ConvBlock(64, 5, 2, 2)
        self.flat = nn.Flatten()
        self.linear6 = nn_architectures.LinearBlock(100, 30)
        self.linear7 = nn.Linear(30, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.linear6(self.flat(x))
        x = self.linear7(x)

        return x

    
class Reccurent(nn.Module):
    """
        Нейронная сеть с применением сверточных слоев
    """

    def __init__(self, hidden_size, num_layers):
        super(Reccurent, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(2, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.flat = nn.Flatten()
        self.linear = nn.Linear(hidden_size*320, 2)
        
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        batch_size = x.shape[0]
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.dummy_param.device)
        
        x, hidden = self.rnn(x, hidden)   
        x = self.flat(x)

        return self.linear(x)
    

class Linear2Layers(nn.Module):
    """
        Нейронная сеть с применением линейных слоев
    """

    def __init__(self):
        super(Linear2Layers, self).__init__()

        self.flat = nn.Flatten()
        self.norm1 = nn.BatchNorm1d(640)
        self.linear1 = LinearBlock(640, 256)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.linear1(self.norm1(self.flat(x)))
        x = self.linear2(x)

        return x
