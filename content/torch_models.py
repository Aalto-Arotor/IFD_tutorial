import torch.nn as nn
import torch.nn.functional as F


class WDCNN(nn.Module):
    def __init__(self, bias=False, dropout=0.0):
        super(WDCNN, self).__init__()

        self.cn_layer1 = ConvLayer_WDCNN(
            1,
            16,
            kernel_size=64,
            stride=16,
            padding=24,
            bias=bias,
            dropout=dropout,
        )
        self.cn_layer2 = ConvLayer_WDCNN(16, 32, bias=bias, dropout=dropout)
        self.cn_layer3 = ConvLayer_WDCNN(32, 64, bias=bias, dropout=dropout)
        self.cn_layer4 = ConvLayer_WDCNN(64, 64, bias=bias, dropout=dropout)
        self.cn_layer5 = ConvLayer_WDCNN(64, 64, bias=bias, dropout=dropout)
        self.cn_layer6 = ConvLayer_WDCNN(64, 64, bias=bias, dropout=dropout)
        self.cn_layer7 = ConvLayer_WDCNN(64, 64, bias=bias, dropout=dropout)

        # Classifier
        self.fc1 = nn.Linear(
            64,  # FIXME
            32,
        )
        # self.bn1 = nn.BatchNorm1d(config["wdcnn_fc1_output_size"])
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        verbose = False

        if verbose:
            print(x.shape)

        # Convolution layers
        out = self.cn_layer1(x)
        if verbose:
            print(out.shape)

        out = self.cn_layer2(out)
        if verbose:
            print(out.shape)

        out = self.cn_layer3(out)
        if verbose:
            print(out.shape)

        out = self.cn_layer4(out)
        if verbose:
            print(out.shape)

        out = self.cn_layer5(out)
        if verbose:
            print(out.shape)

        out = self.cn_layer6(out)
        if verbose:
            print(out.shape)

        out = self.cn_layer7(out)
        if verbose:
            print(out.shape)

        out = F.avg_pool1d(out, out.shape[-1]).squeeze(2)
        if verbose:
            print(out.shape)

        # Reshape channels
        # n_features = out.shape[1] * out.shape[2]
        # out = out.view(-1, n_features).contiguous()
        # if verbose:
        #     print(out.shape)

        # Classifier
        out = F.relu(self.fc1(out))
        if verbose:
            print(out.shape)

        out = self.fc2(out)
        if verbose:
            print(out.shape)
            quit()

        return out


class ConvLayer_ZoomCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=25,
        pool_size=2,
        stride=1,
        padding=0,
        bias=False,
        pool=True,
    ):
        super(ConvLayer_ZoomCNN, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.pool = None
        if pool:
            self.pool = nn.MaxPool1d(pool_size, stride=pool_size)
        # self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        if self.pool:
            out = self.pool(out)
            # out = self.bn(out)

        return out


class ZoomCNN(nn.Module):
    def __init__(self):
        super(ZoomCNN, self).__init__()

        # Demodulation block
        self.cn_layer1 = ConvLayer_ZoomCNN(1, 16, pool_size=15, kernel_size=25)
        # Periodic signal analysis block
        self.cn_layer2 = ConvLayer_ZoomCNN(16, 32, kernel_size=25)
        self.cn_layer3 = ConvLayer_ZoomCNN(32, 64, kernel_size=25)
        self.cn_layer4 = ConvLayer_ZoomCNN(64, 64, kernel_size=25)
        # self.cn_layer5 = ConvLayer_ZoomCNN(64, 64, kernel_size=25)
        # self.cn_layer6 = ConvLayer_ZoomCNN(64, 64, kernel_size=25)
        # self.cn_layer7 = ConvLayer_ZoomCNN(64, 64, kernel_size=25)
        self.cn_layer8 = ConvLayer_ZoomCNN(64, 64, kernel_size=25, pool=False)

        # Classifier
        self.fc1 = nn.Linear(
            64,
            3,
        )

    def forward(self, x):
        verbose = False

        if verbose:
            print(x.shape)

        # Conv layers

        out = self.cn_layer1(x)
        if verbose:
            print(out.shape)
        # print(self.cn_layer1.conv.weight)

        out = self.cn_layer2(out)
        if verbose:
            print(out.shape)

        out = self.cn_layer3(out)
        if verbose:
            print(out.shape)

        out = self.cn_layer4(out)
        if verbose:
            print(out.shape)

        # out = self.cn_layer5(out)
        # if verbose:
        #     print(out.shape)

        # out = self.cn_layer6(out)
        # if verbose:
        #     print(out.shape)

        # out = self.cn_layer7(out)
        # if verbose:
        #     print(out.shape)

        out = self.cn_layer8(out)
        if verbose:
            print(out.shape)

        out = F.avg_pool1d(out, kernel_size=out.shape[-1]).squeeze(2)
        if verbose:
            print("GAP:", out.shape)

        # Match to class num
        out = out.squeeze()
        out = self.fc1(out)
        if verbose:
            print("FC:", out.shape)

        if verbose:
            quit()

        return out


class ConvLayer_WDCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        dropout=0.0,
    ):
        super(ConvLayer_WDCNN, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.dropout = None
        if dropout > 0.0:
            self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.pool(out)

        return out
