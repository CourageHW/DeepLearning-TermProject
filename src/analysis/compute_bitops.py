#!/usr/bin/env python3
"""
Compute FP32 MACs vs Binary OPs for teacher (ResNet18 CIFAR-adapted, FP32) and
student (A1W1ResNet18v2: FP32 stem+fc + 16 binary convs + parameter-free shortcuts).

BitOPs convention (Bi-Real Net / ReActNet): 1 FP32 MAC ≈ 64 BitOPs in HW
(a 64-bit XNOR-POPCNT pair handles 64 channel pairs at once).
So FP32-equivalent FLOPs of student = FP32 part + (binary MACs / 64).
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18
from thop import profile

# ---------- Teacher: FP32 ResNet18 with CIFAR-adapted stem ----------
teacher = resnet18(num_classes=100)
teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
teacher.maxpool = nn.Identity()
teacher.eval()

x = torch.randn(1, 3, 32, 32)
teacher_macs, teacher_params = profile(teacher, inputs=(x,), verbose=False)

# Per-component count: stem, body convs, downsample convs, fc
def conv_macs(c_in, c_out, k_h, k_w, h_out, w_out, groups=1):
    return (c_in / groups) * c_out * k_h * k_w * h_out * w_out

# Spatial sizes for CIFAR ResNet18: 32 -> 32 (layer1) -> 16 (layer2) -> 8 (layer3) -> 4 (layer4)
stem_macs = conv_macs(3, 64, 3, 3, 32, 32)

# layer1 (64ch, 32x32, no downsample): 2 blocks * 2 convs each
body_macs_layer1 = 2 * 2 * conv_macs(64, 64, 3, 3, 32, 32)
# layer2 (64->128, 16x16, downsample at first block): 1 block (stride 2) + 1 block (same)
body_macs_layer2 = (conv_macs(64, 128, 3, 3, 16, 16) + conv_macs(128, 128, 3, 3, 16, 16)) \
                 + 2 * conv_macs(128, 128, 3, 3, 16, 16)
# layer3 (128->256, 8x8)
body_macs_layer3 = (conv_macs(128, 256, 3, 3, 8, 8) + conv_macs(256, 256, 3, 3, 8, 8)) \
                 + 2 * conv_macs(256, 256, 3, 3, 8, 8)
# layer4 (256->512, 4x4)
body_macs_layer4 = (conv_macs(256, 512, 3, 3, 4, 4) + conv_macs(512, 512, 3, 3, 4, 4)) \
                 + 2 * conv_macs(512, 512, 3, 3, 4, 4)
body_macs_total = body_macs_layer1 + body_macs_layer2 + body_macs_layer3 + body_macs_layer4

# Downsample 1x1 convs (3 transitions, present only in teacher)
ds_macs = conv_macs(64, 128, 1, 1, 16, 16) + conv_macs(128, 256, 1, 1, 8, 8) + conv_macs(256, 512, 1, 1, 4, 4)

# Final FC (Linear: 512 -> 100)
fc_macs = 512 * 100

teacher_calc_total = stem_macs + body_macs_total + ds_macs + fc_macs

# ---------- Student: same body MAC count, but binary; no downsample conv ----------
# Student binary conv MAC count = body_macs_total (16 convs in BasicBlock body)
student_binary_macs = body_macs_total
# Student FP32 MACs = stem + fc (downsample is parameter-free in student → 0)
student_fp32_macs = stem_macs + fc_macs

# Bit-equivalent conversion: 1 FP32 MAC ≈ 64 BitOPs (Bi-Real Net / ReActNet convention)
# To put student on the FP32-equivalent scale:
student_fp32_equiv = student_fp32_macs + student_binary_macs / 64.0

# Also report raw BitOPs (1 binary MAC = 1 XNOR + accumulate)
student_bitops = student_binary_macs  # in "binary MAC" units

# ---------- Report ----------
print("=" * 64)
print("Teacher (FP32 ResNet18 CIFAR-adapted)")
print("=" * 64)
print(f"  thop MACs (verified)        : {teacher_macs/1e6:8.2f} M")
print(f"  manual MACs (cross-check)   : {teacher_calc_total/1e6:8.2f} M")
print(f"    - stem                    : {stem_macs/1e6:8.2f} M")
print(f"    - body (16 BasicBlock conv): {body_macs_total/1e6:8.2f} M")
print(f"    - downsample (3 × 1x1)    : {ds_macs/1e6:8.2f} M")
print(f"    - fc                      : {fc_macs/1e6:8.2f} M")
print(f"  Params                      : {teacher_params/1e6:8.2f} M")
print()
print("=" * 64)
print("Student (A1W1ResNet18v2: FP32 stem + fc, 16 binary convs, no DS conv)")
print("=" * 64)
print(f"  FP32 MACs                   : {student_fp32_macs/1e6:8.2f} M")
print(f"    - stem                    : {stem_macs/1e6:8.2f} M")
print(f"    - fc                      : {fc_macs/1e6:8.2f} M")
print(f"  Binary MACs (= BitOPs / 64) : {student_binary_macs/1e6:8.2f} M")
print(f"  Raw BitOPs (1 XNOR per MAC) : {student_binary_macs/1e6:8.2f} M binary OPs")
print()
print(f"  FP32-equivalent FLOPs       : {student_fp32_equiv/1e6:8.2f} M")
print(f"    = stem ({stem_macs/1e6:.2f}M) + fc ({fc_macs/1e6:.2f}M) + body/64 ({student_binary_macs/64/1e6:.2f}M)")
print()
print("=" * 64)
print("Compression ratio")
print("=" * 64)
print(f"  Memory  ratio (44 MB → 1.4 MB)         : 31.4 × (reported in §6.4)")
print(f"  MAC     ratio (teacher / student-FP32) : {teacher_macs/student_fp32_macs:.1f} × ← if treating binary as FP32")
print(f"  FLOPs   ratio (teacher / student-equiv): {teacher_macs/student_fp32_equiv:.1f} ×")
print(f"  Binary share of body MACs              : {student_binary_macs/teacher_calc_total*100:.1f} %")
print()
print(f"  → Student replaces {student_binary_macs/1e6:.1f}M FP32 MACs with binary ops")
print(f"  → That portion contributes only {student_binary_macs/64/1e6:.2f}M FP32-equiv ops")
print(f"  → {teacher_macs/student_fp32_equiv:.1f}× theoretical FLOPs compression on top of 32× memory compression")
