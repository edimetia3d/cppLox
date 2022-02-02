#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_FILE=$1
OUTPUT_FILE=$2
TMP_DIR=$3

if [ -d $TMP_DIR/venv ]; then
  source $TMP_DIR/venv/bin/activate
else
  echo Creating virtual environment
  python3 -m venv $TMP_DIR/venv
  source $TMP_DIR/venv/bin/activate
  python3 -m pip install -r $SCRIPT_DIR/requirements.txt
fi

python3 $SCRIPT_DIR/nodes_code_writer.py $INPUT_FILE $TMP_DIR/gen.tmp
DIFF=$(diff <(tail -n +4 $OUTPUT_FILE) <(tail -n +4 $TMP_DIR/gen.tmp))
if [ "$DIFF" != "" ]; then
  cp $TMP_DIR/gen.tmp $OUTPUT_FILE
fi
