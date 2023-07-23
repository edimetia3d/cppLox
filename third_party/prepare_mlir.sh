#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

BUILD_DIR=$SCRIPT_DIR/llvm-project/build
INSTALL_DIR=$SCRIPT_DIR/llvm-project/install

if [ -d "llvm-project" ]; then
  echo "llvm-project directory exists, auto clone is skipped"
else
  git clone -n https://github.com/llvm/llvm-project.git
fi

mkdir -p $BUILD_DIR
rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR
git checkout 993bdb047c90e9b85fb91578349a9faf4f6a853d # latest master for now

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_USE_SPLIT_DWARF=ON \
  #-DMLIR_ENABLE_CUDA_RUNNER=ON \
  #-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DLLVM_INSTALL_UTILS=ON
cmake --build . --target install