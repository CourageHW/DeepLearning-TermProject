"""
A1W1ResNet18v2 + dependencies — inference-only definitions,
extracted from train.ipynb cells 8 and 9.

Just enough to load best_student.pth and run forward.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Binary primitives
# ============================================================
class BinaryActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.ge(0).to(x.dtype).mul(2).sub(1)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1).to(g.dtype)


class BinaryWeightSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        return w.ge(0).to(w.dtype).mul(2).sub(1)
    @staticmethod
    def backward(ctx, g):
        return g


class RSignActivation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.threshold = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        return BinaryActivationSTE.apply(x - self.threshold.to(x.dtype))


class ShortcutDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.pool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.pad_ch = out_ch - in_ch
    def forward(self, x):
        x = self.pool(x)
        if self.pad_ch <= 0:
            return x
        return F.pad(x, (0, 0, 0, 0, 0, self.pad_ch))


# ============================================================
# A1W1ResNet18v2
# ============================================================
class RPReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.prelu = nn.PReLU(num_parameters=channels)
    def forward(self, x):
        return self.prelu(x - self.gamma) + self.beta


class A1W1BinaryConv2dV2(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, bias=False):
        super().__init__()
        self.stride, self.padding = stride, padding
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, k, k))
        self.bias   = nn.Parameter(torch.zeros(out_ch)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        w_b   = BinaryWeightSTE.apply(self.weight)
        alpha = self.weight.detach().abs().mean(dim=[1, 2, 3]).view(1, -1, 1, 1)
        out   = F.conv2d(x, w_b, self.bias, self.stride, self.padding)
        return out * alpha.to(out.dtype)


class A1W1BasicBlockV2(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, use_rprelu=True, use_double_skip=True):
        super().__init__()
        self.use_double_skip = use_double_skip
        self.bn1    = nn.BatchNorm2d(in_planes)
        self.rsign1 = RSignActivation(in_planes)
        self.conv1  = A1W1BinaryConv2dV2(in_planes, planes, 3, stride=stride, padding=1)
        self.bn2    = nn.BatchNorm2d(planes)
        self.act1   = RPReLU(planes) if use_rprelu else nn.Identity()

        self.bn3    = nn.BatchNorm2d(planes)
        self.rsign2 = RSignActivation(planes)
        self.conv2  = A1W1BinaryConv2dV2(planes, planes, 3, padding=1)
        self.bn4    = nn.BatchNorm2d(planes)
        self.act2   = RPReLU(planes) if use_rprelu else nn.Identity()

        self.downsample_a = None
        if stride != 1 or in_planes != planes:
            self.downsample_a = ShortcutDownsample(in_planes, planes, stride)

    def forward(self, x):
        res_a = x if self.downsample_a is None else self.downsample_a(x)
        out = self.act1(self.bn2(self.conv1(self.rsign1(self.bn1(x)))))
        out = out + res_a
        res_b = out if self.use_double_skip else 0
        out2 = self.act2(self.bn4(self.conv2(self.rsign2(self.bn3(out)))))
        return out2 + res_b


class A1W1ResNet18v2(nn.Module):
    def __init__(self, num_classes=100, use_rprelu=True, use_double_skip=True):
        super().__init__()
        self.use_rprelu = use_rprelu
        self.use_double_skip = use_double_skip
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.act_in = RPReLU(64) if use_rprelu else nn.PReLU(64)
        self.layer1 = self._make_layer(64,  64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128,256, 2, stride=2)
        self.layer4 = self._make_layer(256,512, 2, stride=2)
        self.bn_out = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_p, p, n, stride):
        layers = [A1W1BasicBlockV2(in_p, p, stride, self.use_rprelu, self.use_double_skip)]
        for _ in range(1, n):
            layers.append(A1W1BasicBlockV2(p, p, 1, self.use_rprelu, self.use_double_skip))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act_in(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.bn_out(x); x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def load_student(ckpt_path, device='cpu'):
    """Build A1W1ResNet18v2 and load weights from `ckpt_path`."""
    model = A1W1ResNet18v2(num_classes=100).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model
