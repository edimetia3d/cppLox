#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

CMAKE_EXTRA_ARGS=${@:1}

if [ -d "llvm-project" ]; then
  echo "llvm-project directory exists, auto clone is skipped"
else
  git clone https://github.com/llvm/llvm-project.git
fi

mkdir llvm-project/build
cd llvm-project/build
git checkout 75e33f71c2dae584b13a7d1186ae0a038ba98838 # LLVM 13.0.1

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  $CMAKE_EXTRA_ARGS

cmake --build . --target mlir-opt
