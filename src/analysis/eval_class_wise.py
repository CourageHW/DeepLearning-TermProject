#!/usr/bin/env python3
"""
Per-class accuracy evaluation of A1W1ResNet18v2 student on CIFAR-100 test set.
Outputs worst-10 / best-10 classes + most common confusions.
"""
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as T

# ============================================================
# Model definitions (copied from train.ipynb cell 6)
# ============================================================
class BinaryActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return g * (x.abs() <= 1).to(g.dtype)


class BinaryWeightSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        return torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
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
        out = self.act1(self.bn2(self.conv1(self.rsign1(self.bn1(x))))) + res_a
        res_b = out if self.use_double_skip else 0
        out2 = self.act2(self.bn4(self.conv2(self.rsign2(self.bn3(out))))) + res_b
        return out2


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


# Class names come from torchvision's dataset (testset.classes) below.

# ============================================================
# Eval
# ============================================================
def main():
    device = torch.device("cpu")
    ckpt_path = ".agents/best_student.pth"

    # Load checkpoint
    model = A1W1ResNet18v2(num_classes=100, use_rprelu=True, use_double_skip=True)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:  print("  first missing:", missing[:3])
    if unexpected: print("  first unexpected:", unexpected[:3])
    model.eval().to(device)

    # CIFAR-100 test (ImageNet normalize stats — matches §6.1)
    MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=MEAN, std=STD),
    ])
    testset = datasets.CIFAR100(root="/home/yonggi/data", train=False, download=False, transform=transform)
    classes = testset.classes
    assert len(classes) == 100, len(classes)
    loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

    correct = [0] * 100
    total   = [0] * 100
    confusion = defaultdict(int)

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            pred   = logits.argmax(dim=1)
            for t, p in zip(y.tolist(), pred.tolist()):
                total[t] += 1
                if p == t: correct[t] += 1
                else:      confusion[(t, p)] += 1

    overall = sum(correct) / sum(total) * 100
    print(f"\nOverall top-1: {overall:.2f} %  ({sum(correct)}/{sum(total)})")

    # Per-class
    accs = [(c, correct[c] / total[c] * 100, total[c]) for c in range(100)]
    accs.sort(key=lambda x: x[1])

    print("\n=== Worst 10 classes ===")
    for c, a, t in accs[:10]:
        print(f"  {classes[c]:<18}  {a:5.1f} %   ({correct[c]}/{t})")

    print("\n=== Best 10 classes ===")
    for c, a, t in accs[-10:][::-1]:
        print(f"  {classes[c]:<18}  {a:5.1f} %   ({correct[c]}/{t})")

    print("\n=== Top 10 confusions (true → pred, count) ===")
    confs = sorted(confusion.items(), key=lambda x: -x[1])[:10]
    for (t, p), n in confs:
        print(f"  {classes[t]:<18} → {classes[p]:<18}  ×{n}")

    # Class-name groupings (for narrative)
    print("\n=== Worst 10 — semantic clusters ===")
    worst_names = [classes[c] for c, _, _ in accs[:10]]
    print("  raw list:", worst_names)


if __name__ == "__main__":
    main()
