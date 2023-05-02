import torch
import torchaudio
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, activation='ReLU', layers=4):
        super().__init__()
        self.name = 'SimpleCNN'
        assert activation in ['ReLU', 'tanh'], "Activation must be either 'ReLU' or 'tanh'"

        # Choose the activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU
        else:
            self.activation = nn.Tanh

        self.layers_number = layers
        self.F1 = 4
        self.F2 = 8
        self.F3 = 8
        self.F4 = 16
        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, self.F1, kernel_size=(8, 8), stride=(4, 4), padding=(2, 2))
        self.relu1 = self.activation()
        self.bn1 = nn.BatchNorm2d(self.F1)

        # Second Convolution Block
        self.conv2 = nn.Conv2d(self.F1, self.F2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = self.activation ()
        self.bn2 = nn.BatchNorm2d(self.F2)

        # Third Convolution Block
        self.conv3 = nn.Conv2d(self.F2, self.F3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = self.activation ()
        self.bn3 = nn.BatchNorm2d(self.F3)

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(self.F3, self.F4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = self.activation()
        self.bn4 = nn.BatchNorm2d(self.F4)

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=self.F4, out_features=8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        a1 = torch.flatten(x, 2, 3)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        a2 = torch.flatten(x, 2, 3)
        x = self.bn2(x)
        if self.layers_number == 4:
            x = self.conv3(x)
            x = self.relu3(x)
            a3 = torch.flatten(x, 2, 3)
            x = self.bn3(x)
            x = self.conv4(x)
            x = self.relu4(x)
            a4 = torch.flatten(x, 2, 3)
            x = self.bn4(x)

        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)

        if self.layers_number == 4:
            return x, a1, a2, a3, a4
        else:
            return x, a1, a2


class Res2DBlock(nn.Module):
    expansion = 1 #we don't use the block.expansion here

    def __init__(self, inplanes, planes, stride=1,padding = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride=stride,
                     padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride=1,
                     padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, FN=16, num_classes=8, p_dropout=None):
        super().__init__()

        self.FN = FN
        if FN == 128:
            self.name = 'ResNet34-XL'
        elif FN == 64:
            self.name = 'ResNet34-L'
        elif FN == 32:
            self.name = 'ResNet34-M'
        elif FN == 16:
            self.name = 'ResNet34-S'
        else:
            self.name = 'ResNet34'
        layers = [3, 4, 6, 3]
        self.c1 = nn.Conv2d(1, FN, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(FN)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(FN, FN, layers[0])
        self.layer2 = self._make_layer(FN, FN * 2, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.fc = nn.Linear(FN * 98, num_classes)
        self.p_dropout = p_dropout
        if p_dropout:
            self.dropout = nn.Dropout(p=p_dropout)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(Res2DBlock(inplanes, planes, stride))

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(Res2DBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.p_dropout:
            x = self.dropout(x)

        return x


class SimpleCNN2(nn.Module):
    def __init__(self, activation='ReLU'):
        super().__init__()
        self.name = 'SimpleCNN'
        assert activation in ['ReLU', 'tanh'], "Activation must be either 'ReLU' or 'tanh'"

        # Choose the activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU
        else:
            self.activation = nn.Tanh

        # First Convolution Block with Activation and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(8, 8), stride=(4, 4), padding=(2, 2))
        self.act1 = self.activation()
        self.bn1 = nn.BatchNorm2d(4)

        # Second Convolution Block
        self.conv2 = nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.act2 = self.activation()
        self.bn2 = nn.BatchNorm2d(16)

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.act3 = self.activation()
        self.bn3 = nn.BatchNorm2d(32)

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.act4 = self.activation()
        self.bn4 = nn.BatchNorm2d(64)

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        a1 = torch.flatten(x, 2, 3)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.act2(x)
        a2 = torch.flatten(x, 2, 3)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.act3(x)
        a3 = torch.flatten(x, 2, 3)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.act4(x)
        a4 = torch.flatten(x, 2, 3)
        x = self.bn4(x)

        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        return x, a1, a2, a3, a4
class SimpleCNN2(nn.Module):
    def __init__(self, activation='ReLU'):
        super().__init__()
        self.name = 'SimpleCNN'
        assert activation in ['ReLU', 'tanh'], "Activation must be either 'ReLU' or 'tanh'"
        
        # Choose the activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU
        else:
            self.activation = nn.Tanh
        
        # First Convolution Block with Activation and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(8, 8), stride=(4, 4), padding=(2, 2))
        self.act1 = self.activation()
        self.bn1 = nn.BatchNorm2d(4)

        # Second Convolution Block
        self.conv2 = nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.act2 = self.activation()
        self.bn2 = nn.BatchNorm2d(16)
        
         # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.act3 = self.activation()
        self.bn3 = nn.BatchNorm2d(32)

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.act4 = self.activation()
        self.bn4 = nn.BatchNorm2d(64)

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        a1 = torch.flatten(x, 2, 3)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        a2 = torch.flatten(x, 2, 3)
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = self.act3(x)
        a3 = torch.flatten(x, 2, 3)
        x = self.bn3(x)
        
        x = self.conv4(x)
        x = self.act4(x)
        a4 = torch.flatten(x, 2, 3)
        x = self.bn4(x)       
        
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        return x, a1, a2, a3, a4
