import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ENet(nn.Module):
    def __init__(self, num_input_channels):
        super(ENet, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_feature_maps = 36
        self.num_conv_layers = 7
        self.downsampled_channels = 1

        self.intermediate_ecnn = IntermediateECNN(input_features=self.downsampled_channels,
                                                    middle_features=self.num_feature_maps,
                                                    num_conv_layers=self.num_conv_layers)
        self.mlp1 = nn.Linear(4096, 2000)
        self.mlp2 = nn.Linear(2000, 200)
        self.mlp3 = nn.Linear(200, 50)
        self.mlp4 = nn.Linear(50, 1)

    def forward(self, x):
        h_dncnn = self.intermediate_ecnn(x)
        x = 0.1*F.relu(self.mlp1(x.view(h_dncnn.size(0), -1)))
        x = 0.1*F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x = self.mlp4(x)
        return x


class IntermediateECNN(nn.Module):
    def __init__(self, input_features, middle_features, num_conv_layers):
        super(IntermediateECNN, self).__init__()
        self.kernel_size = 3

        self.padding = 1
        self.input_features = input_features
        self.num_conv_layers = num_conv_layers
        self.middle_features = middle_features
        self.output_features = 1

        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_features, out_channels=self.middle_features,
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features,
                                    kernel_size=self.kernel_size, padding=self.padding, bias=False))

            layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.output_features,
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        self.itermediate_encnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.itermediate_encnn(x)
        return out


class CBDNet(nn.Module):
    def __init__(self, num_input_channels):
        super(CBDNet, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_feature_maps = 64
        self.num_conv_layers = 10
        self.downsampled_channels = 1

        self.intermediate_dncnn = IntermediateDCNN(input_features=self.downsampled_channels,
                                                    middle_features=self.num_feature_maps,
                                                    num_conv_layers=self.num_conv_layers)

    def forward(self, x):
        h_dncnn = self.intermediate_dncnn(x)
        return h_dncnn


class IntermediateDCNN(nn.Module):
    def __init__(self, input_features, middle_features, num_conv_layers):
        super(IntermediateDCNN, self).__init__()
        self.kernel_size = 3
        self.padding = 1
        self.input_features = input_features
        self.num_conv_layers = num_conv_layers
        self.middle_features = middle_features
        self.output_features = 1

        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_features, out_channels=self.middle_features,
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features,
                                    kernel_size=self.kernel_size, padding=self.padding, bias=False))
            layers.append(nn.BatchNorm2d(self.middle_features))
            layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.output_features,
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        self.itermediate_dcnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.itermediate_dcnn(x)
        return out
