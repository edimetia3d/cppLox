#!/bin/bash
DIFF=$(diff <(tail -n +4 $1) <(tail -n +4 $2))
if [ "$DIFF" != "" ]
then
    cp "$1" "$2"
fi
