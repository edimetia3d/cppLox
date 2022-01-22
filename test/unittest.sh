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
cmake $SRC_DIR -DCMAKE_CXX_COMPILER=clang++-13 -DCMAKE_BUILD_TYPE=Release -DUPSTREAM_STYLE_SYNCHRONIZE=ON
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
dart tool/bin/test.dart clox operator --interpreter "$BINARY_PATH" --loose_mode
dart tool/bin/test.dart clox assignment --interpreter "$BINARY_PATH" --loose_mode
dart tool/bin/test.dart clox block --interpreter "$BINARY_PATH" --loose_mode
dart tool/bin/test.dart clox bool --interpreter "$BINARY_PATH" --loose_mode
dart tool/bin/test.dart clox call --interpreter "$BINARY_PATH" --loose_mode
dart tool/bin/test.dart clox class --interpreter "$BINARY_PATH" --loose_mode
dart tool/bin/test.dart clox closure --interpreter "$BINARY_PATH" --loose_mode
dart tool/bin/test.dart clox comments --interpreter "$BINARY_PATH" --loose_mode
dart tool/bin/test.dart clox constructor --interpreter "$BINARY_PATH" --loose_mode
dart tool/bin/test.dart clox expressions --interpreter "$BINARY_PATH" --loose_mode
dart tool/bin/test.dart clox field --interpreter "$BINARY_PATH" --loose_mode
dart tool/bin/test.dart clox for --interpreter "$BINARY_PATH" --loose_mode
dart tool/bin/test.dart clox function --interpreter "$BINARY_PATH" --loose_mode
echo "Testing with tree walker"
dart tool/bin/test.dart clox operator/add.lox --interpreter "$BINARY_PATH" --loose_mode --arguments --backend="TreeWalker"
popd