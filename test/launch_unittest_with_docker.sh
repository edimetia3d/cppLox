#!/bin/sh
SCRIPT_DIR=`realpath $(dirname "$0")`
cd $SCRIPT_DIR

./docker/build_image.sh cpp_lox_test_env

docker run -it --rm -v $SCRIPT_DIR/..:/workspace/cpplox cpp_lox_test_env \
 bash /workspace/cpplox/test/unittest.sh
