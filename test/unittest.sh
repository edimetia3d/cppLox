#!/bin/sh
SCRIPT_DIR=`realpath $(dirname "$0")`
cd $SCRIPT_DIR
if [ "$#" -lt 2 ]; then
  SRC_DIR=$SCRIPT_DIR/..
else
  SRC_DIR=$2
fi

if [ "$#" -lt 1 ]; then
  TEST_CACHE_DIR=$SCRIPT_DIR/test_cache
else
  TEST_CACHE_DIR=$1
fi
echo CACHE_DIR: $TEST_CACHE_DIR
mkdir -p $TEST_CACHE_DIR
cd $TEST_CACHE_DIR

mkdir -p build
pushd build
cmake $SRC_DIR -DCMAKE_CXX_COMPILER=clang++-13 -DCMAKE_BUILD_TYPE=Release -DUPSTREAM_STYLE_ERROR_MSG=ON
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
dart tool/bin/test.dart clox --interpreter "$BINARY_PATH"

echo "Testing with tree walker and pratt parser"
dart tool/bin/test.dart jlox --interpreter "$BINARY_PATH"  --loose_mode --arguments --backend="TreeWalker" --arguments --parser="PrattParser"

echo "Testing with tree walker and recursive descent parser"
dart tool/bin/test.dart jlox --interpreter "$BINARY_PATH"  --loose_mode --arguments --backend="TreeWalker" --arguments --parser="RecursiveDescent"
popd
