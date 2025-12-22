import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    n, c, h, w = x.size()
    if c % groups != 0:
        return x
    x = x.view(n, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(n, c, h, w)
    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super().__init__()
        self.stride = int(stride)
        self.groups = int(groups)
        mid_channels = out_channels // 4
        out_channels_proj = out_channels - in_channels if self.stride == 2 else out_channels
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.dw_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=self.stride, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels_proj, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False),
            nn.BatchNorm2d(out_channels_proj),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1) if self.stride == 2 else None

    def forward(self, x):
        out = self.compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.dw_conv(out)
        out = self.expand(out)
        if self.stride == 1:
            out = F.relu(out + x, inplace=True)
            return out
        x_proj = self.avgpool(x)
        out = torch.cat((x_proj, out), dim=1)
        out = F.relu(out, inplace=True)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, num_classes=43, groups=3, stage_repeats=(4, 8, 4), stage_out_channels=(240, 480, 960)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        in_channels = 24
        self.stage2 = self._make_stage(in_channels, stage_out_channels[0], stage_repeats[0], groups)
        in_channels = stage_out_channels[0]
        self.stage3 = self._make_stage(in_channels, stage_out_channels[1], stage_repeats[1], groups)
        in_channels = stage_out_channels[1]
        self.stage4 = self._make_stage(in_channels, stage_out_channels[2], stage_repeats[2], groups)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_out_channels[2], num_classes)

    def _make_stage(self, in_channels, out_channels, repeats, groups):
        blocks = [ShuffleUnit(in_channels, out_channels, stride=2, groups=groups)]
        for _ in range(repeats - 1):
            blocks.append(ShuffleUnit(out_channels, out_channels, stride=1, groups=groups))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
