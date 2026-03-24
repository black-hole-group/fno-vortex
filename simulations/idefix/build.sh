#!/bin/bash
# Build Idefix for GPU (RTX 3090 Ti = Ampere sm_86)
# Run this from the simulations/idefix/ directory.
# Requires: $IDEFIX_DIR to point to the Idefix source tree.

set -e

if [ -z "$IDEFIX_DIR" ]; then
  echo "Error: IDEFIX_DIR is not set. Please set it to the Idefix source directory."
  exit 1
fi

cmake "$IDEFIX_DIR" \
  -DIdefix_MHD=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE86=ON

make -j"$(nproc)"

echo "Build complete. Idefix binary: $(pwd)/idefix"
