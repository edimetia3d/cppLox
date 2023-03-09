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
git checkout db0c7dde6b18035bd7d2022f3ea939d8323e72d5 # latest master for now
# disable the test to avoid some build issue
sed -i 's/add_subdirectory(ExceptionDemo)/#dd_subdirectory(ExceptionDemo)/g' $SCRIPT_DIR/llvm-project/llvm/examples/CMakeLists.txt
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_INSTALL_UTILS=ON
cmake --build . --target install
# recover
sed -i 's/#dd_subdirectory(ExceptionDemo)/add_subdirectory(ExceptionDemo)/g' $SCRIPT_DIR/llvm-project/llvm/examples/CMakeLists.txt
