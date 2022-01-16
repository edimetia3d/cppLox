#!/bin/sh

TEST_CACHE_DIR=$1
SRC_DIR=$2

cd $TEST_CACHE_DIR

mkdir -p build
pushd build
cmake $SRC_DIR -DCMAKE_CXX_COMPILER=clang++-13 -DCMAKE_BUILD_TYPE=Release
make -j4
BINARY_PATH=$PWD/bin/lox
popd

git clone https://github.com/munificent/craftinginterpreters.git --depth=1

mkdir -p dart_pub_cache
export PUB_CACHE=$PWD/dart_pub_cache
pushd craftinginterpreters/tool
dart pub get
popd

pushd craftinginterpreters
dart tool/bin/test.dart clox operator/add.lox --interpreter "$BINARY_PATH"
popd