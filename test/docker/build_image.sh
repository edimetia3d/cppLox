#!/bin/sh
SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR

docker build  -t $1 -f ./test_env.Dockerfile .