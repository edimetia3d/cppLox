#!/bin/sh
SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR

./docker/build_image.sh cpp_lox_test_env

mkdir -p test_cache
docker run -it --rm -v $PWD/..:/workspace/cpplox -v $PWD/test_cache:/workspace/test_cache cpp_lox_test_env \
 bash /workspace/cpplox/test/unittest.sh /workspace/test_cache /workspace/cpplox