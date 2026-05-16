"""
Export A1W1ResNet18v2 weights from .pth → custom binary format for C inference.

Format (little-endian):
    Header
        4 bytes : magic "BNN0"
        uint32  : num_tensors
        uint32  : reserved
    For each tensor:
        uint32  : name_length
        bytes   : name (UTF-8)
        uint32  : dtype  (0=float32, 1=uint64_packed, 2=int32)
        uint32  : ndim
        uint32[ndim] : shape
        bytes   : raw data (little-endian)

Binary conv weights are pre-binarized + packed:
    sign(w) ∈ {-1,+1}  →  bit 1 = positive, 0 = negative
    pack contiguous along (C_in, K, K) flattened
    shape: (C_out, K_words) uint64
    also store per-channel alpha = mean(|w|) as separate float32 tensor.

Usage:
    python export_weights.py best_student.pth   # writes best_student.bnn
"""
import argparse
import os
import struct
import sys
import numpy as np
import torch

MAGIC = b'BNN0'
DTYPE_F32, DTYPE_U64, DTYPE_I32 = 0, 1, 2


def _write_tensor(f, name, dtype_code, arr):
    name_b = name.encode('utf-8')
    f.write(struct.pack('<I', len(name_b)))
    f.write(name_b)
    f.write(struct.pack('<I', dtype_code))
    f.write(struct.pack('<I', arr.ndim))
    for d in arr.shape:
        f.write(struct.pack('<I', d))
    f.write(arr.tobytes(order='C'))


def pack_binary_weight(w_fp32):
    """Pack a float weight tensor (C_out, C_in, K, K) into uint64 bits.

    Flatten (C_in, K, K) -> (K_bits = C_in*K*K), pack along that axis.
    bit value = 1 if w >= 0, else 0.
    """
    C_out, C_in, K, _ = w_fp32.shape
    K_bits = C_in * K * K
    K_words = (K_bits + 63) // 64
    flat = w_fp32.reshape(C_out, K_bits)
    bits = (flat >= 0).astype(np.uint8)              # (C_out, K_bits)
    packed = np.zeros((C_out, K_words), dtype=np.uint64)
    for kw in range(K_words):
        lo = kw * 64
        hi = min(lo + 64, K_bits)
        word = np.zeros(C_out, dtype=np.uint64)
        for b in range(hi - lo):
            word |= (bits[:, lo + b].astype(np.uint64)) << b
        packed[:, kw] = word
    return packed, K_bits


def compute_alpha(w_fp32):
    """Per-output-channel scaling: alpha = mean(|w|) along (C_in, K, K)."""
    return np.abs(w_fp32).mean(axis=(1, 2, 3)).astype(np.float32)


# Names of all A1W1BinaryConv2dV2 conv weights inside our model.
def _binary_conv_keys():
    keys = []
    for stage in range(1, 5):                    # layer1..layer4
        for blk in range(2):                     # 2 blocks each
            for c in (1, 2):                     # conv1, conv2
                keys.append(f'layer{stage}.{blk}.conv{c}.weight')
    return keys


def export(ckpt_path, out_path, model_type=None):
    """Auto-detect model type:
        'student' — A1W1ResNet18v2 (has rsign/RPReLU keys, binary convs)
        'teacher' — FP32 ResNet18 with CIFAR stem (standard torchvision keys)
    """
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    if model_type is None:
        has_rsign = any('rsign1.threshold' in k for k in sd.keys())
        model_type = 'student' if has_rsign else 'teacher'
    print(f"detected model_type={model_type}")

    if model_type == 'student':
        binary_keys = set(_binary_conv_keys())
    else:
        binary_keys = set()   # everything FP32 for teacher

    tensors = []
    for k, v in sd.items():
        arr = v.detach().cpu().numpy()
        if k in binary_keys:
            assert arr.ndim == 4
            packed, K_bits = pack_binary_weight(arr.astype(np.float32))
            alpha = compute_alpha(arr.astype(np.float32))
            tensors.append((k + '.packed', DTYPE_U64, packed))
            tensors.append((k + '.alpha',  DTYPE_F32, alpha))
            tensors.append((k + '.shape',  DTYPE_I32, np.array(arr.shape, dtype=np.int32)))
            tensors.append((k + '.K_bits', DTYPE_I32, np.array([K_bits], dtype=np.int32)))
        else:
            tensors.append((k, DTYPE_F32, arr.astype(np.float32)))
    tensors.append(('__model_type', DTYPE_I32,
                    np.array([0 if model_type == 'student' else 1], dtype=np.int32)))

    with open(out_path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', len(tensors)))
        f.write(struct.pack('<I', 0))            # reserved
        for name, dtype, arr in tensors:
            _write_tensor(f, name, dtype, arr)

    total_bytes = sum(t[2].nbytes for t in tensors)
    print(f"wrote {out_path}: {len(tensors)} tensors, {total_bytes/1024/1024:.2f} MB")
    return tensors


def load_tensors(path):
    """Read back the binary file. Returns dict name -> np.ndarray."""
    out = {}
    with open(path, 'rb') as f:
        if f.read(4) != MAGIC:
            raise ValueError("bad magic")
        num_tensors, _ = struct.unpack('<II', f.read(8))
        for _ in range(num_tensors):
            (nl,) = struct.unpack('<I', f.read(4))
            name = f.read(nl).decode('utf-8')
            (dtype,) = struct.unpack('<I', f.read(4))
            (ndim,) = struct.unpack('<I', f.read(4))
            shape = struct.unpack('<%dI' % ndim, f.read(4 * ndim))
            np_dtype = {DTYPE_F32: np.float32, DTYPE_U64: np.uint64, DTYPE_I32: np.int32}[dtype]
            size = int(np.prod(shape)) if shape else 1
            data = np.frombuffer(f.read(size * np_dtype().nbytes), dtype=np_dtype).reshape(shape)
            out[name] = data.copy()                 # mutable copy
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', help='path to a .pth checkpoint')
    ap.add_argument('-o', '--out', default=None, help='output .bnn path')
    ap.add_argument('--type', choices=['student', 'teacher', None], default=None,
                    help='force model type (defaults to auto-detect)')
    args = ap.parse_args()
    out = args.out or os.path.splitext(args.ckpt)[0] + '.bnn'
    tensors = export(args.ckpt, out, model_type=args.type)
    print("\nFirst 15 tensors:")
    for name, dt, arr in tensors[:15]:
        print(f"  {name:40s} dtype={dt} shape={tuple(arr.shape)}")
