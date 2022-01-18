#!/bin/sh
SCRIPT_DIR=$(dirname "$0")

if [ "$#" -lt 2 ]; then
  SRC_DIR=$SCRIPT_DIR/..
else
  SRC_DIR=$2
fi

if [ "$#" -lt 1 ]; then
  mkdir -p test_cache
  TEST_CACHE_DIR=$SCRIPT_DIR/test_cache
else
  TEST_CACHE_DIR=$1
fi

cd $TEST_CACHE_DIR

mkdir -p build
pushd build
cmake $SRC_DIR -DCMAKE_CXX_COMPILER=clang++-13 -DCMAKE_BUILD_TYPE=Release
make -j4
BINARY_PATH=$PWD/bin/lox
popd

git clone https://github.com/edimetia3d/craftinginterpreters.git --depth=1

mkdir -p dart_pub_cache
export PUB_CACHE=$PWD/dart_pub_cache
pushd craftinginterpreters/tool
dart pub get
popd

pushd craftinginterpreters
echo "Testing with virtual machine"
dart tool/bin/test.dart clox operator/ --interpreter "$BINARY_PATH" --loose_mode
echo "Testing with tree walker"
dart tool/bin/test.dart clox operator/add.lox --interpreter "$BINARY_PATH" --loose_mode --arguments --backend="TreeWalker"
popd