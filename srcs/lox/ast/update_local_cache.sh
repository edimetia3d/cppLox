#!/bin/bash
DIFF=$(diff <(tail -n +3 $1) <(tail -n +3 $2))
if [ "$DIFF" != "" ]
then
    cp "$1" "$2"
fi
