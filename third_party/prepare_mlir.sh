#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

BUILD_DIR=$SCRIPT_DIR/llvm-project/build
INSTALL_DIR=$SCRIPT_DIR/llvm-project/install

if [ -d "llvm-project" ]; then
  echo "llvm-project directory exists, auto clone is skipped"
else
  git clone https://github.com/llvm/llvm-project.git
fi

mkdir -p $BUILD_DIR
rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR
git checkout 75e33f71c2dae584b13a7d1186ae0a038ba98838 # LLVM 13.0.1

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_SHARED_LIBS=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_RTTI=ON

cmake --build . --target install -j2
