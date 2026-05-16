"""
C-backed inference engine for A1W1ResNet18v2.

Loads exported weights (.bnn) and orchestrates the forward pass by calling
primitives in xnor_kernel.so via ctypes.

Usage:
    # 1) Export weights once:
    python export_weights.py best_student.pth -o best_student.bnn

    # 2) Build the C kernel:
    bash build.sh

    # 3a) Validate against PyTorch (single forward, compare outputs):
    python inference_c.py --validate --ckpt best_student.pth --weights best_student.bnn

    # 3b) Run visual per-sample demo on CIFAR-100 test:
    python inference_c.py --demo --weights best_student.bnn
    python inference_c.py --demo --weights best_student.bnn --max 100 --color
"""
import argparse
import ctypes
import os
import sys
import time
from collections import OrderedDict

import numpy as np

from export_weights import load_tensors


# ---- CIFAR-100 class names (standard torchvision order) -------------
CIFAR100_CLASSES = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle',
    'bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel',
    'can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud',
    'cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant',
    'flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
    'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree',
    'motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter',
    'palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine',
    'possum','rabbit','raccoon','ray','road','rocket','rose','sea',
    'seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
    'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television',
    'tiger','tractor','train','trout','tulip','turkey','turtle','wardrobe',
    'whale','willow_tree','wolf','woman','worm',
]


# ---- Visualization helpers ------------------------------------------
def cls(idx): return CIFAR100_CLASSES[int(idx)]


def print_memory_banner(student_path=None, teacher_path=None):
    """Print storage / compression banner from .bnn file sizes."""
    print("┌" + "─" * 70 + "┐")
    print(f"│  {'Model':<15} {'Storage':>12}     {'Params (full prec)':>22}     {'Compress':>9}   │")
    print("├" + "─" * 70 + "┤")
    t_bytes = os.path.getsize(teacher_path) if teacher_path and os.path.exists(teacher_path) else 0
    s_bytes = os.path.getsize(student_path) if student_path and os.path.exists(student_path) else 0
    if t_bytes:
        print(f"│  {'Teacher FP32':<15} {t_bytes/1024/1024:>9.2f} MB     "
              f"{'11.2 M × 32-bit':>22}     {'1.00×':>9}   │")
    if s_bytes:
        ratio = (t_bytes / max(s_bytes, 1)) if t_bytes else 0
        compr = f'{ratio:5.2f}×' if t_bytes else '(no T)'
        print(f"│  {'Student W1A1':<15} {s_bytes/1024/1024:>9.2f} MB     "
              f"{'11.0 M × 1-bit':>22}     {compr:>9}   │")
    print("└" + "─" * 70 + "┘")


def latency_bar(ms, max_ms, width=30):
    """ASCII bar proportional to ms / max_ms."""
    n = max(1, min(width, int(round(ms / max(max_ms, 1e-6) * width))))
    return '▇' * n + '·' * (width - n)


def ascii_thumb(img_chw, height=6, width=16):
    """Render a (3, 32, 32) tensor as a tiny grayscale ASCII block."""
    g = img_chw.mean(axis=0).astype(np.float32)
    gmin, gmax = float(g.min()), float(g.max())
    if gmax - gmin < 1e-6:
        return [' ' * width for _ in range(height)]
    g = (g - gmin) / (gmax - gmin)
    H, W = g.shape
    bh = max(1, H // height); bw = max(1, W // width)
    chars = ' ░▒▓█'
    out = []
    for i in range(height):
        row = []
        for j in range(width):
            block = g[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            v = float(block.mean())
            row.append(chars[min(len(chars) - 1, int(v * len(chars)))])
        out.append(''.join(row))
    return out


def top_k(logits, k=3):
    """Return list of (class_idx, prob) for top-k predictions."""
    e = np.exp(logits - float(logits.max()))
    probs = e / float(e.sum())
    idx = np.argsort(-logits)[:k]
    return [(int(i), float(probs[i])) for i in idx]


class ClassStats:
    """Track per-class accuracy + confusion pairs for a single engine."""
    def __init__(self, num_classes=100):
        self.num_classes = num_classes
        self.correct = [0] * num_classes
        self.total   = [0] * num_classes
        self.confusion = {}

    def update(self, true, pred):
        self.total[true] += 1
        if pred == true:
            self.correct[true] += 1
        else:
            self.confusion[(true, pred)] = self.confusion.get((true, pred), 0) + 1

    def print_breakdown(self, label="", topn=5):
        accs = [(c, self.correct[c] / max(1, self.total[c]), self.total[c])
                for c in range(self.num_classes) if self.total[c] > 0]
        accs.sort(key=lambda x: x[1])
        print(f"\n  ── {label}: worst {topn} classes ──")
        for c, a, t in accs[:topn]:
            print(f"     {cls(c):<18}  {a*100:5.1f}%   ({self.correct[c]}/{t})")
        print(f"  ── {label}: best {topn} classes ──")
        for c, a, t in accs[-topn:][::-1]:
            print(f"     {cls(c):<18}  {a*100:5.1f}%   ({self.correct[c]}/{t})")
        confs = sorted(self.confusion.items(), key=lambda x: -x[1])[:topn]
        if confs:
            print(f"  ── {label}: top {topn} confusions (true → pred, count) ──")
            for (t, p), n in confs:
                print(f"     {cls(t):<18} → {cls(p):<18}  ×{n}")


# ---- ctypes setup ----------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))


def _make_lib(path):
    lib = ctypes.CDLL(path)

    f32p = ctypes.POINTER(ctypes.c_float)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    i32  = ctypes.c_int

    lib.fp32_conv2d.argtypes = [
        f32p, f32p, f32p,                          # input, weight, bias
        i32, i32, i32, i32, i32, i32, i32,         # C_in,H,W, C_out,K,stride,padding
        f32p,                                      # output
    ]
    lib.batch_norm_2d.argtypes = [
        f32p, f32p, f32p, f32p, f32p,              # input, mean, var, weight, bias
        i32, i32, i32, ctypes.c_float,             # C,H,W,eps
        f32p,                                      # output
    ]
    lib.rsign.argtypes = [f32p, f32p, i32, i32, i32, f32p]
    lib.rprelu.argtypes = [f32p, f32p, f32p, f32p, i32, i32, i32, f32p]
    lib.binary_conv2d.argtypes = [
        f32p, u64p, f32p,                          # input_signed, weight_packed, alpha
        i32, i32, i32, i32, i32, i32, i32, i32,    # K_bits,C_in,H,W,C_out,K,stride,padding
        f32p,                                      # output
    ]
    lib.shortcut_downsample.argtypes = [
        f32p, i32, i32, i32, i32, i32, f32p,
    ]
    lib.adaptive_avgpool_1.argtypes = [f32p, i32, i32, i32, f32p]
    lib.linear_fp32.argtypes = [f32p, f32p, f32p, i32, i32, f32p]
    lib.add_inplace.argtypes = [f32p, f32p, i32]
    lib.relu_inplace.argtypes = [f32p, i32]
    lib.cpu_set_num_threads.argtypes = [i32]
    lib.cpu_set_num_threads.restype  = None
    lib.cpu_get_max_threads.argtypes = []
    lib.cpu_get_max_threads.restype  = i32

    return lib


def _f32(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _u64(a):
    a = np.ascontiguousarray(a, dtype=np.uint64)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


# ---- Engine ----------------------------------------------------------
class BinaryInferenceEngine:
    BN_EPS = 1e-5

    def __init__(self, weights_path, lib_path=None):
        if lib_path is None:
            lib_path = os.path.join(_DIR, 'xnor_kernel.so')
        self.lib = _make_lib(lib_path)
        self.weights = load_tensors(weights_path)

        # Convert and hold strong references so ctypes pointers stay valid.
        self.refs = {}
        for k, v in self.weights.items():
            if v.dtype == np.float32:
                arr, _ = _f32(v); self.refs[k] = arr
            elif v.dtype == np.uint64:
                arr, _ = _u64(v); self.refs[k] = arr
            else:
                self.refs[k] = v.copy()

    # ---- small helpers --------------------------------------------------
    def _w(self, name): return self.refs[name]
    def _ptr(self, name, kind='f32'):
        a = self.refs[name]
        if kind == 'f32':
            return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if kind == 'u64':
            return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        raise ValueError(kind)

    def _alloc(self, shape):
        return np.zeros(shape, dtype=np.float32)

    def _fp_conv(self, x, prefix, C_in, H, W, C_out, K, stride, padding):
        out = self._alloc((C_out, (H + 2*padding - K)//stride + 1, (W + 2*padding - K)//stride + 1))
        bias_ptr = self._ptr(prefix + '.bias') if (prefix + '.bias') in self.refs else None
        bias_ptr = bias_ptr if bias_ptr else ctypes.cast(0, ctypes.POINTER(ctypes.c_float))
        self.lib.fp32_conv2d(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._ptr(prefix + '.weight'), bias_ptr,
            C_in, H, W, C_out, K, stride, padding,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def _bn(self, x, prefix, C, H, W):
        out = self._alloc((C, H, W))
        self.lib.batch_norm_2d(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._ptr(prefix + '.running_mean'),
            self._ptr(prefix + '.running_var'),
            self._ptr(prefix + '.weight'),
            self._ptr(prefix + '.bias'),
            C, H, W, ctypes.c_float(self.BN_EPS),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def _rsign(self, x, prefix, C, H, W):
        out = self._alloc((C, H, W))
        # threshold stored in train code as (1, C, 1, 1) → flatten
        thr = self.refs[prefix + '.threshold'].reshape(-1)
        self.lib.rsign(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            thr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C, H, W,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def _rprelu(self, x, prefix, C, H, W):
        out = self._alloc((C, H, W))
        g = self.refs[prefix + '.gamma'].reshape(-1)
        b = self.refs[prefix + '.beta'].reshape(-1)
        a = self.refs[prefix + '.prelu.weight'].reshape(-1)
        self.lib.rprelu(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            g.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C, H, W,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def _binary_conv(self, x_signed, prefix, C_in, H, W, K, stride, padding):
        K_bits = int(self.refs[prefix + '.weight.K_bits'][0])
        shape  = self.refs[prefix + '.weight.shape']                 # (C_out,C_in,K,K)
        C_out  = int(shape[0])
        H_out  = (H + 2*padding - K)//stride + 1
        W_out  = (W + 2*padding - K)//stride + 1
        out = self._alloc((C_out, H_out, W_out))
        self.lib.binary_conv2d(
            x_signed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._ptr(prefix + '.weight.packed', 'u64'),
            self._ptr(prefix + '.weight.alpha'),
            K_bits, C_in, H, W, C_out, K, stride, padding,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def _shortcut(self, x, C_in, H, W, C_out, stride):
        H_out = H // stride; W_out = W // stride
        out = self._alloc((C_out, H_out, W_out))
        self.lib.shortcut_downsample(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C_in, H, W, C_out, stride,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def _block(self, x, prefix, in_planes, planes, H, W, stride):
        """Forward one A1W1BasicBlockV2."""
        # Branch 1
        h1 = self._bn(x, prefix + '.bn1', in_planes, H, W)
        b1 = self._rsign(h1, prefix + '.rsign1', in_planes, H, W)
        z1 = self._binary_conv(b1, prefix + '.conv1', in_planes, H, W, K=3, stride=stride, padding=1)
        H_mid = z1.shape[1]; W_mid = z1.shape[2]
        z1 = self._bn(z1, prefix + '.bn2', planes, H_mid, W_mid)
        p1 = self._rprelu(z1, prefix + '.act1', planes, H_mid, W_mid)
        # residual_a
        if stride != 1 or in_planes != planes:
            res_a = self._shortcut(x, in_planes, H, W, planes, stride)
        else:
            res_a = x
        self.lib.add_inplace(
            p1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            res_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            p1.size,
        )
        # Branch 2
        h3 = self._bn(p1, prefix + '.bn3', planes, H_mid, W_mid)
        b2 = self._rsign(h3, prefix + '.rsign2', planes, H_mid, W_mid)
        z2 = self._binary_conv(b2, prefix + '.conv2', planes, H_mid, W_mid, K=3, stride=1, padding=1)
        z2 = self._bn(z2, prefix + '.bn4', planes, H_mid, W_mid)
        p2 = self._rprelu(z2, prefix + '.act2', planes, H_mid, W_mid)
        # res_b = p1 (double-skip)
        self.lib.add_inplace(
            p2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            p1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            p2.size,
        )
        return p2, H_mid, W_mid

    # ---- full forward ---------------------------------------------------
    def forward(self, x):
        """x: numpy (3, 32, 32) float32. Returns logits (100,) float32."""
        assert x.shape == (3, 32, 32), f"expected (3,32,32), got {x.shape}"
        x = np.ascontiguousarray(x, dtype=np.float32)

        # Stem
        h = self._fp_conv(x, 'conv1', 3, 32, 32, 64, K=3, stride=1, padding=1)
        h = self._bn(h, 'bn1', 64, 32, 32)
        h = self._rprelu(h, 'act_in', 64, 32, 32)

        # Layers
        configs = [
            ('layer1', 64,  64,  1),
            ('layer2', 64,  128, 2),
            ('layer3', 128, 256, 2),
            ('layer4', 256, 512, 2),
        ]
        H, W = 32, 32
        in_planes = 64
        for stage, (name, in_p, planes, stage_stride) in enumerate(configs):
            for b in range(2):
                stride = stage_stride if b == 0 else 1
                prefix = f"{name}.{b}"
                # in_planes for first block is the stage's `in_p`, else planes
                ip = in_p if b == 0 else planes
                h, H, W = self._block(h, prefix, ip, planes, H, W, stride)
            in_planes = planes

        # Tail
        h = self._bn(h, 'bn_out', 512, H, W)
        pooled = self._alloc((512,))
        self.lib.adaptive_avgpool_1(
            h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            512, H, W,
            pooled.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        logits = self._alloc((100,))
        self.lib.linear_fp32(
            pooled.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._ptr('fc.weight'),
            self._ptr('fc.bias'),
            512, 100,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return logits


# ---- FP32 ResNet18 (teacher) engine ----------------------------------
class FP32ResNet18Engine:
    """Same C primitives as the binary engine, but every conv is fp32.
    Targets the CIFAR-adapted torchvision ResNet18 (conv1 stride=1, no maxpool).
    All weights kept as FP32 — no packing."""
    BN_EPS = 1e-5

    def __init__(self, weights_path, lib_path=None, use_blas=False):
        if lib_path is None:
            lib_path = os.path.join(_DIR, 'xnor_kernel.so')
        self.lib = _make_lib(lib_path)
        # Optionally load BLAS-backed FP32 conv (im2col + cblas_sgemm).
        # Same C signature; we swap which symbol gets called in `_conv`.
        self.use_blas = use_blas
        self._conv_fn = self.lib.fp32_conv2d
        if use_blas:
            blas_path = os.path.join(_DIR, 'fp32_blas_kernel.so')
            if not os.path.exists(blas_path):
                raise FileNotFoundError(
                    f"--blas requested but {blas_path} not built. Run build.sh first."
                )
            self.blas_lib = ctypes.CDLL(blas_path)
            self.blas_lib.fp32_conv2d_blas.argtypes = self.lib.fp32_conv2d.argtypes
            self.blas_lib.fp32_conv2d_blas.restype  = self.lib.fp32_conv2d.restype
            self._conv_fn = self.blas_lib.fp32_conv2d_blas
        self.weights = load_tensors(weights_path)
        self.refs = {}
        for k, v in self.weights.items():
            if v.dtype == np.float32:
                arr, _ = _f32(v); self.refs[k] = arr
            else:
                self.refs[k] = v.copy()

    def _ptr(self, name):
        return self.refs[name].ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def _alloc(self, shape):
        return np.zeros(shape, dtype=np.float32)

    def _conv(self, x, prefix, C_in, H, W, C_out, K, stride, padding):
        H_out = (H + 2*padding - K)//stride + 1
        W_out = (W + 2*padding - K)//stride + 1
        out = self._alloc((C_out, H_out, W_out))
        bias_ptr = self._ptr(prefix + '.bias') if (prefix + '.bias') in self.refs else \
                   ctypes.cast(0, ctypes.POINTER(ctypes.c_float))
        self._conv_fn(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._ptr(prefix + '.weight'), bias_ptr,
            C_in, H, W, C_out, K, stride, padding,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out, H_out, W_out

    def _bn(self, x, prefix, C, H, W):
        out = self._alloc((C, H, W))
        self.lib.batch_norm_2d(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._ptr(prefix + '.running_mean'),
            self._ptr(prefix + '.running_var'),
            self._ptr(prefix + '.weight'),
            self._ptr(prefix + '.bias'),
            C, H, W, ctypes.c_float(self.BN_EPS),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def _relu(self, x):
        self.lib.relu_inplace(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x.size)
        return x

    def _block(self, x, prefix, in_ch, planes, H, W, stride, has_downsample):
        """Standard torchvision ResNet BasicBlock."""
        identity = x
        out, H_mid, W_mid = self._conv(x, prefix + '.conv1', in_ch, H, W,
                                       planes, K=3, stride=stride, padding=1)
        out = self._bn(out, prefix + '.bn1', planes, H_mid, W_mid)
        out = self._relu(out)
        out, _, _ = self._conv(out, prefix + '.conv2', planes, H_mid, W_mid,
                               planes, K=3, stride=1, padding=1)
        out = self._bn(out, prefix + '.bn2', planes, H_mid, W_mid)
        if has_downsample:
            d, _, _ = self._conv(x, prefix + '.downsample.0', in_ch, H, W,
                                 planes, K=1, stride=stride, padding=0)
            identity = self._bn(d, prefix + '.downsample.1', planes, H_mid, W_mid)
        self.lib.add_inplace(
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            identity.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.size,
        )
        out = self._relu(out)
        return out, H_mid, W_mid

    def forward(self, x):
        """x: (3, 32, 32) float32 — normalized. Returns logits (100,)."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        # Stem: 7x7 conv (torchvision default kept), stride=1, padding=3 (CIFAR-adapted)
        conv1_w = self.refs['conv1.weight']
        K_stem = conv1_w.shape[-1]   # could be 7 (torchvision default) or 3 (CIFAR custom)
        pad_stem = K_stem // 2        # 3 for K=7, 1 for K=3
        h, H, W = self._conv(x, 'conv1', 3, 32, 32, 64, K=K_stem, stride=1, padding=pad_stem)
        h = self._bn(h, 'bn1', 64, H, W)
        h = self._relu(h)
        # maxpool is Identity in our CIFAR adaptation -> skip

        configs = [
            ('layer1', 64,  64,  1, False),
            ('layer2', 64,  128, 2, True),
            ('layer3', 128, 256, 2, True),
            ('layer4', 256, 512, 2, True),
        ]
        in_ch = 64
        for name, in_p, planes, stage_stride, stage_downsample in configs:
            for b in range(2):
                stride = stage_stride if b == 0 else 1
                has_down = stage_downsample if b == 0 else False
                prefix = f"{name}.{b}"
                ip = in_p if b == 0 else planes
                h, H, W = self._block(h, prefix, ip, planes, H, W, stride, has_down)
            in_ch = planes

        # avgpool, fc
        pooled = self._alloc((512,))
        self.lib.adaptive_avgpool_1(
            h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            512, H, W,
            pooled.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        logits = self._alloc((100,))
        self.lib.linear_fp32(
            pooled.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._ptr('fc.weight'),
            self._ptr('fc.bias'),
            512, 100,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return logits


# ---- validation against PyTorch -------------------------------------
def validate(ckpt_path, weights_path, lib_path=None):
    import torch
    sys.path.insert(0, _DIR)
    from models import load_student

    print(f"== validation ==")
    print(f"  ckpt    : {ckpt_path}")
    print(f"  weights : {weights_path}")
    model_pt = load_student(ckpt_path, device='cpu')
    model_pt.eval()
    engine   = BinaryInferenceEngine(weights_path, lib_path)

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((1, 3, 32, 32)).astype(np.float32)
    x_t  = torch.from_numpy(x_np)

    with torch.inference_mode():
        logits_pt = model_pt(x_t).numpy().squeeze(0)
    logits_c = engine.forward(x_np[0])

    diff = np.abs(logits_pt - logits_c)
    print(f"  PyTorch top-5 indices : {np.argsort(-logits_pt)[:5]}")
    print(f"  C       top-5 indices : {np.argsort(-logits_c)[:5]}")
    print(f"  argmax match          : pt={logits_pt.argmax()} c={logits_c.argmax()}")
    print(f"  max |diff|            : {diff.max():.4f}")
    print(f"  mean |diff|           : {diff.mean():.4f}")
    if diff.max() < 0.1:
        print("  ✓ outputs match within tolerance")
    else:
        print("  ⚠ outputs differ — check padding handling, alpha, or dtype")
    return diff.max()


# ---- visual demo on CIFAR-100 test ---------------------------------
def demo(weights_path, data_dir, max_samples, color, lib_path=None, show_image=False, use_blas=False):
    from torchvision import datasets
    from torchvision.transforms import v2 as T
    import torch

    MEAN = np.array([0.4850, 0.4560, 0.4060], dtype=np.float32).reshape(3, 1, 1)
    STD  = np.array([0.2290, 0.2240, 0.2250], dtype=np.float32).reshape(3, 1, 1)

    # Auto-detect engine type from the exported weights.
    tmp = load_tensors(weights_path)
    is_teacher = ('__model_type' in tmp and int(tmp['__model_type'][0]) == 1)
    if is_teacher:
        print(f"loaded weights from {weights_path}  →  FP32 ResNet18 (teacher)")
        engine = FP32ResNet18Engine(weights_path, lib_path, use_blas=use_blas)
        label_tag = "Teacher" + (" (BLAS)" if use_blas else "")
    else:
        print(f"loaded weights from {weights_path}  →  A1W1ResNet18v2 (student)")
        engine = BinaryInferenceEngine(weights_path, lib_path)
        label_tag = "Student"
    del tmp
    print(f"loading CIFAR-100 test set (cache: {data_dir})")

    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    N = len(testset) if max_samples <= 0 else min(max_samples, len(testset))
    print(f"running C inference on {N} samples\n")

    GREEN, RED, RESET = '\033[32m', '\033[31m', '\033[0m'
    correct = 0; total_ms = 0.0; lats = []
    stats = ClassStats()
    bar_max = 1.0

    for i in range(N):
        img_t, label = testset[i]
        img_np = (img_t.numpy() - MEAN) / STD
        t0 = time.perf_counter()
        logits = engine.forward(img_np)
        ms = (time.perf_counter() - t0) * 1000.0
        pred = int(logits.argmax())
        ok = (pred == label)
        correct += ok; total_ms += ms; lats.append(ms)
        stats.update(int(label), pred)
        bar_max = max(bar_max, ms)
        mark = ('O' if ok else 'X')
        if color:
            mark = (GREEN if ok else RED) + mark + RESET
        acc = correct * 100.0 / (i + 1)
        top3 = top_k(logits, 3)
        top3_str = ", ".join(f"{cls(c)[:10]:<10} {p*100:4.1f}%" for c, p in top3)
        print(f"[{i:>5d}/{N:>5d}] [{mark}] {cls(label):<14}  {ms:6.1f}ms {latency_bar(ms, bar_max, 12)}  "
              f"top3=[{top3_str}]  acc={acc:5.2f}%")
        if show_image:
            for row in ascii_thumb(img_t.numpy(), height=6, width=18):
                print(f"               {row}")

    lats.sort()
    print('\n' + '=' * 70)
    print(f"  model         : {label_tag}")
    print(f"  total samples : {N}")
    print(f"  correct       : {correct}  ({correct * 100.0 / N:.2f}%)")
    print(f"  avg latency   : {total_ms / N:.2f} ms / sample (C inference)")
    print(f"  p50 / p95     : {lats[N // 2]:.2f} / {lats[int(N * 0.95)]:.2f} ms")
    print(f"  throughput    : {1000.0 / (total_ms / N):.1f} images / sec")
    print('=' * 70)
    stats.print_breakdown(label_tag)


# ---- PyTorch CPU engine (baseline using BLAS/MKL) -------------------
class PyTorchTeacherEngine:
    """Loads the FP32 ResNet18 teacher (same CIFAR-adapted layout as build_teacher
    in train.ipynb) and runs forward via standard PyTorch CPU backend (BLAS)."""
    def __init__(self, ckpt_path, num_threads=1):
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18
        torch.set_num_threads(num_threads)
        m = resnet18(weights=None)
        m.conv1.stride = (1, 1)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, 100)
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        sd = {k.replace('module.', ''): v for k, v in sd.items() if 'num_batches_tracked' not in k or k in sd}
        m.load_state_dict(sd, strict=False)
        m.eval()
        self.model = m
        self._torch = torch

    def forward(self, x_np):
        x_t = self._torch.from_numpy(x_np).unsqueeze(0)
        with self._torch.inference_mode():
            return self.model(x_t).numpy().squeeze(0)


# ---- Side-by-side compare: student (binary) vs teacher (FP32), both in C ---
def compare(student_weights, teacher_weights, data_dir, max_samples, color,
            lib_path=None, use_blas=False):
    from torchvision import datasets
    from torchvision.transforms import v2 as T
    import torch

    MEAN = np.array([0.4850, 0.4560, 0.4060], dtype=np.float32).reshape(3, 1, 1)
    STD  = np.array([0.2290, 0.2240, 0.2250], dtype=np.float32).reshape(3, 1, 1)

    print(f"loading student (W1A1) : {student_weights}")
    stu = BinaryInferenceEngine(student_weights, lib_path)
    print(f"loading teacher (FP32{', BLAS' if use_blas else ''}) : {teacher_weights}")
    tch = FP32ResNet18Engine(teacher_weights, lib_path, use_blas=use_blas)

    print(f"loading CIFAR-100 test set (cache: {data_dir})")
    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    N = len(testset) if max_samples <= 0 else min(max_samples, len(testset))
    print_memory_banner(student_weights, teacher_weights)
    print(f"\nrunning both engines on {N} samples\n")

    GREEN, RED, RESET = '\033[32m', '\033[31m', '\033[0m'
    s_correct = t_correct = 0
    s_total = t_total = 0.0
    s_lats = []; t_lats = []
    s_stats = ClassStats(); t_stats = ClassStats()
    # Track a typical max for the latency bar (use teacher as the slow side).
    bar_max = 1.0

    for i in range(N):
        img_t, label = testset[i]
        img_np = (img_t.numpy() - MEAN) / STD
        # student
        ts0 = time.perf_counter(); stu_logits = stu.forward(img_np)
        sms = (time.perf_counter() - ts0) * 1000.0
        # teacher
        tt0 = time.perf_counter(); tch_logits = tch.forward(img_np)
        tms = (time.perf_counter() - tt0) * 1000.0

        sp = int(stu_logits.argmax()); tp = int(tch_logits.argmax())
        s_ok = (sp == label); t_ok = (tp == label)
        s_correct += s_ok; t_correct += t_ok
        s_total += sms; t_total += tms
        s_lats.append(sms); t_lats.append(tms)
        s_stats.update(label, sp); t_stats.update(label, tp)
        bar_max = max(bar_max, tms)

        sm = ('O' if s_ok else 'X'); tm = ('O' if t_ok else 'X')
        if color:
            sm = (GREEN if s_ok else RED) + sm + RESET
            tm = (GREEN if t_ok else RED) + tm + RESET
        speedup = tms / sms if sms > 0 else 0.0
        sacc = s_correct * 100.0 / (i + 1)
        tacc = t_correct * 100.0 / (i + 1)
        truth = cls(label)
        sp_lbl = cls(sp); tp_lbl = cls(tp)

        # Pretty per-sample line: latency bars + class names.
        print(f"[{i:>5d}/{N:>5d}] {truth:<14} "
              f"Stu [{sm}] {sms:6.1f}ms {latency_bar(sms, bar_max, 14)}  "
              f"Tch [{tm}] {tms:6.1f}ms {latency_bar(tms, bar_max, 14)}  "
              f"{speedup:5.2f}× | S→{sp_lbl[:12]:<12} T→{tp_lbl[:12]:<12} | "
              f"acc S={sacc:5.2f}% T={tacc:5.2f}%")

    s_lats.sort(); t_lats.sort()
    print('\n' + '=' * 96)
    print(f"  samples              : {N}")
    print(f"  Student acc (W1A1)   : {s_correct*100.0/N:.2f}%")
    print(f"  Teacher acc (FP32)   : {t_correct*100.0/N:.2f}%")
    print(f"  Student avg latency  : {s_total/N:.2f} ms (p50 {s_lats[N//2]:.2f}, p95 {s_lats[int(N*0.95)]:.2f})")
    print(f"  Teacher avg latency  : {t_total/N:.2f} ms (p50 {t_lats[N//2]:.2f}, p95 {t_lats[int(N*0.95)]:.2f})")
    print(f"  speedup (T/S)        : {t_total/s_total:.2f}×")
    print(f"  throughput Student   : {1000.0/(s_total/N):.1f} img/sec")
    print(f"  throughput Teacher   : {1000.0/(t_total/N):.1f} img/sec")
    print('=' * 96)

    s_stats.print_breakdown("Student")
    t_stats.print_breakdown("Teacher")


# ---- Three-way compare: PyTorch FP32 + C Student (W1A1) + C Teacher (FP32) ---
def compare3(student_weights, teacher_weights, teacher_ckpt, data_dir,
             max_samples, color, threads, lib_path=None, use_blas=False):
    from torchvision import datasets
    from torchvision.transforms import v2 as T
    import torch

    MEAN = np.array([0.4850, 0.4560, 0.4060], dtype=np.float32).reshape(3, 1, 1)
    STD  = np.array([0.2290, 0.2240, 0.2250], dtype=np.float32).reshape(3, 1, 1)

    print(f"loading C engines ...")
    stu = BinaryInferenceEngine(student_weights, lib_path)
    tch = FP32ResNet18Engine(teacher_weights, lib_path, use_blas=use_blas)
    print(f"loading PyTorch teacher (BLAS, threads={threads}) ...")
    pyt = PyTorchTeacherEngine(teacher_ckpt, num_threads=threads)

    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    N = len(testset) if max_samples <= 0 else min(max_samples, len(testset))
    print_memory_banner(student_weights, teacher_weights)
    print(f"\nrunning 3 engines on {N} samples (PyTorch=BLAS, C-Student=binary, C-Teacher=naive FP32)\n")

    GREEN, RED, RESET = '\033[32m', '\033[31m', '\033[0m'
    s_correct = t_correct = p_correct = 0
    s_total = t_total = p_total = 0.0
    s_stats = ClassStats(); t_stats = ClassStats(); p_stats = ClassStats()
    bar_max = 1.0

    for i in range(N):
        img_t, label = testset[i]
        img_np = (img_t.numpy() - MEAN) / STD
        # warm-up first run of each engine pre-loop would be nicer; we accept run-1 jitter.
        ts0 = time.perf_counter(); sl = stu.forward(img_np);  sms = (time.perf_counter() - ts0) * 1000.0
        tt0 = time.perf_counter(); tl = tch.forward(img_np);  tms = (time.perf_counter() - tt0) * 1000.0
        tp0 = time.perf_counter(); pl = pyt.forward(img_np);  pms = (time.perf_counter() - tp0) * 1000.0

        sp = int(sl.argmax()); tp = int(tl.argmax()); pp = int(pl.argmax())
        s_ok = (sp == label); t_ok = (tp == label); p_ok = (pp == label)
        s_correct += s_ok; t_correct += t_ok; p_correct += p_ok
        s_total += sms; t_total += tms; p_total += pms
        s_stats.update(label, sp); t_stats.update(label, tp); p_stats.update(label, pp)
        bar_max = max(bar_max, tms, pms)

        sm = 'O' if s_ok else 'X'; tm = 'O' if t_ok else 'X'; pm = 'O' if p_ok else 'X'
        if color:
            sm = (GREEN if s_ok else RED) + sm + RESET
            tm = (GREEN if t_ok else RED) + tm + RESET
            pm = (GREEN if p_ok else RED) + pm + RESET
        sacc = s_correct*100/(i+1); tacc = t_correct*100/(i+1); pacc = p_correct*100/(i+1)
        print(f"[{i:>5d}/{N:>5d}] {cls(label):<14} "
              f"PyT [{pm}] {pms:6.1f}ms {latency_bar(pms, bar_max, 12)} | "
              f"C-Tch [{tm}] {tms:6.1f}ms {latency_bar(tms, bar_max, 12)} | "
              f"C-Stu [{sm}] {sms:6.1f}ms {latency_bar(sms, bar_max, 12)}  "
              f"acc P={pacc:5.2f}% T={tacc:5.2f}% S={sacc:5.2f}%")

    print('\n' + '=' * 96)
    print(f"  samples              : {N}")
    print(f"  PyTorch FP32 (BLAS)  : acc {p_correct*100/N:.2f}%  avg {p_total/N:.2f} ms")
    print(f"  C Teacher FP32       : acc {t_correct*100/N:.2f}%  avg {t_total/N:.2f} ms")
    print(f"  C Student W1A1       : acc {s_correct*100/N:.2f}%  avg {s_total/N:.2f} ms")
    print(f"  Student faster than C-Teacher : {t_total/s_total:6.2f}×  (algorithm only — both naive C)")
    print(f"  Student faster than PyTorch   : {p_total/s_total:6.2f}×  (vs production FP32 BLAS)")
    print(f"  PyTorch faster than C-Teacher : {t_total/p_total:6.2f}×  (BLAS+multi-thread vs naive C)")
    print('=' * 96)
    s_stats.print_breakdown("C Student W1A1")


# ---- CLI ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true', help='compare PyTorch vs C (binary student)')
    ap.add_argument('--demo',     action='store_true', help='single-model visual demo (auto: student or teacher)')
    ap.add_argument('--compare',  action='store_true', help='side-by-side W1A1 vs FP32 teacher, both in C')
    ap.add_argument('--compare3', action='store_true', help='three-way: PyTorch + C teacher + C student')
    ap.add_argument('--ckpt',     default='best_student.pth')
    ap.add_argument('--teacher_ckpt', default='best_teacher.pth')
    ap.add_argument('--weights',  default='best_student.bnn')
    ap.add_argument('--teacher_weights', default='best_teacher.bnn')
    ap.add_argument('--lib',      default=os.path.join(_DIR, 'xnor_kernel.so'))
    ap.add_argument('--blas',     action='store_true',
                    help='use im2col+SGEMM (fp32_blas_kernel.so) for FP32 teacher convs')
    ap.add_argument('--data',     default=os.path.expanduser('~/data'))
    ap.add_argument('--max',      type=int, default=0)
    ap.add_argument('--color',    action='store_true')
    ap.add_argument('--show-image', action='store_true',
                    help='print 6×18 ASCII thumbnail of each input image (demo mode)')
    ap.add_argument('--threads',  type=int, default=1,
                    help='OpenMP thread count for C kernels (default: 1)')
    args = ap.parse_args()

    # Configure C-side thread count once.
    _tmp_lib = _make_lib(args.lib)
    _tmp_lib.cpu_set_num_threads(args.threads)
    print(f"C threads: {args.threads}  (available cores: {_tmp_lib.cpu_get_max_threads()})")

    if args.validate:
        validate(args.ckpt, args.weights, lib_path=args.lib)
    elif args.demo:
        demo(args.weights, args.data, args.max, args.color, lib_path=args.lib,
             show_image=args.show_image, use_blas=args.blas)
    elif args.compare:
        compare(args.weights, args.teacher_weights, args.data, args.max, args.color,
                lib_path=args.lib, use_blas=args.blas)
    elif args.compare3:
        compare3(args.weights, args.teacher_weights, args.teacher_ckpt,
                 args.data, args.max, args.color, args.threads, lib_path=args.lib,
                 use_blas=args.blas)
    else:
        print("specify --validate, --demo, --compare, or --compare3")
        sys.exit(1)


if __name__ == '__main__':
    main()
