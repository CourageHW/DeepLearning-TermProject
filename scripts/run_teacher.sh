#!/usr/bin/env bash
# Run FP32 teacher (ResNet18) C inference baseline on CIFAR-100.
# Run from repo root. Requires src/cpp_inference/fp32_blas_kernel.so
# (built via `bash src/cpp_inference/build.sh`) and CIFAR-100 data at ~/data.
python3 src/cpp_inference/inference_c.py --demo --thread 4 \
      --weights checkpoints/best_teacher.bnn \
      --data ~/data --max 100
