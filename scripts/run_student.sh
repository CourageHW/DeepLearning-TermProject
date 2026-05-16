#!/usr/bin/env bash
# Run student (1-bit BNN) C inference demo on CIFAR-100.
# Run from repo root. Requires src/cpp_inference/{xnor_kernel.so,fp32_blas_kernel.so}
# (built via `bash src/cpp_inference/build.sh`) and CIFAR-100 data at ~/data.
python3 src/cpp_inference/inference_c.py --demo --thread 4 \
      --weights checkpoints/best_student.bnn \
      --data ~/data --max 1000
