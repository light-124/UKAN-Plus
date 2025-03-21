import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict


class SKKANAttention(nn.Module):
    def __init__(self, in_channels=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super(SKKANAttention, self).__init__()
        self.d = max(L, in_channels // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(in_channels)),
                    ('relu', nn.ReLU())
                ]))
            )

        # self.fc = nn.Linear(in_channels, self.d)
        self.fc = KAN([in_channels, self.d])
        # self.fc = KAN([in_channels, self.d, self.d, self.d, self.d])
        self.fcs = nn.ModuleList([])

        for i in range(len(kernels)):
            self.fcs.append(KAN([self.d, in_channels]))
        self.softmax = nn.Softmax(dim=0)


    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k, bs, channel, h, w

        ### fuse
        U = sum(conv_outs)  # bs, c, h, w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs, c
        Z = self.fc(S)  # bs, d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs, channel, 1, 1
        attention_weights = torch.stack(weights, 0)  # k, bs, channel, 1, 1
        attention_weights = self.softmax(attention_weights)  # k, bs, channel, 1, 1

        ### fuse
        V = (attention_weights * feats).sum(0)
        return V



class SpatialAugAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):          # x.shape = torch.Size([50, 512, 7, 7])
        max_result, _ = torch.max(x, dim=1, keepdim=True)                  # torch.Size([50, 1, 7, 7])
        avg_result = torch.mean(x, dim=1, keepdim=True)                    # torch.Size([50, 1, 7, 7])
        con_result = self.conv1(x)                                         # torch.Size([50, 1, 7, 7])

        result = torch.cat([max_result, avg_result, con_result], 1)        # torch.Size([50, 2, 7, 7])

        output = self.conv(result)
        output = self.sigmoid(output)
        return output



class KSCS(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        # self.ca = SKAttention(channel=channel, reduction=reduction)
        self.ca = SKKANAttention(in_channels=channel, reduction=reduction)

        # self.sa = SpatialAttention(in_channels=channel, kernel_size=kernel_size)
        self.sa = SpatialAugAttention(in_channels=channel, kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x

        out = self.ca(x)
        out1 = x * self.sa(x)
        return out + residual + out1




if __name__ == '__main__':
    import time
    t0 = time.time()
    input = torch.randn(50, 512, 7, 7)

    kscs = KSCS(channel=512, reduction=16, kernel_size=7)
    output = kscs(input)
    print(output.shape)

    t1 = time.time()
    print(t1 - t0)

