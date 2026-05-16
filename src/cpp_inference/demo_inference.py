"""
Visual per-sample CPU inference demo for the trained W1A1 student.

Each test image is processed one at a time, printing:
    [   idx/total] [O/X] pred=YY truth=ZZ   N.Nms   acc=AA.A%

Usage:
    # 1) Place best_student.pth next to this file (or pass --ckpt)
    # 2) Run on CPU:
    python demo_inference.py
    python demo_inference.py --ckpt /path/to/best_student.pth
    python demo_inference.py --max 100             # only first 100 samples
    python demo_inference.py --data ~/data         # cache CIFAR100 here
    python demo_inference.py --color               # ANSI colour output

CIFAR-100 test set is downloaded via torchvision (10000 images).
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as T

from models import load_student


# Match training normalization (ImageNet stats, since pretrained teacher used them).
MEAN = [0.4850, 0.4560, 0.4060]
STD  = [0.2290, 0.2240, 0.2250]

GREEN = '\033[32m'
RED   = '\033[31m'
RESET = '\033[0m'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='best_student.pth')
    ap.add_argument('--data', default=os.path.expanduser('~/data'))
    ap.add_argument('--max',  type=int, default=0, help='stop after N samples (0 = all 10000)')
    ap.add_argument('--color', action='store_true', help='colour the O/X marks')
    ap.add_argument('--quiet', action='store_true', help='only print summary line')
    args = ap.parse_args()

    torch.set_num_threads(1)
    device = 'cpu'

    print(f"loading checkpoint: {args.ckpt}")
    model = load_student(args.ckpt, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model ready — {n_params:,} parameters")

    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=MEAN, std=STD),
    ])
    print(f"loading CIFAR-100 test set (cache dir: {args.data})")
    testset = datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform)
    N = len(testset) if args.max <= 0 else min(args.max, len(testset))
    loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    print(f"running {N} samples on CPU\n")

    correct = 0
    total_latency = 0.0
    latencies = []

    with torch.inference_mode():
        for i, (image, label) in enumerate(loader):
            if i >= N: break
            image = image.to(device); label = int(label.item())

            t0 = time.perf_counter()
            out = model(image)
            pred = int(out.argmax(dim=1).item())
            ms = (time.perf_counter() - t0) * 1000.0

            ok = (pred == label)
            correct += ok
            total_latency += ms
            latencies.append(ms)

            mark = 'O' if ok else 'X'
            if args.color:
                mark = (GREEN if ok else RED) + mark + RESET

            if not args.quiet:
                acc = correct * 100.0 / (i + 1)
                print(f"[{i:>5d}/{N:>5d}] [{mark}] pred={pred:3d} truth={label:3d}  {ms:6.1f}ms  acc={acc:5.2f}%")

    avg_ms = total_latency / N
    latencies.sort()
    p50 = latencies[N // 2]
    p95 = latencies[int(N * 0.95)]

    print('\n' + '=' * 60)
    print(f"  total samples : {N}")
    print(f"  correct       : {correct}  ({correct * 100.0 / N:.2f}%)")
    print(f"  avg latency   : {avg_ms:.2f} ms / sample")
    print(f"  p50 / p95     : {p50:.2f} / {p95:.2f} ms")
    print(f"  throughput    : {1000.0 / avg_ms:.1f} images / sec")
    print('=' * 60)


if __name__ == '__main__':
    main()
