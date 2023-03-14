#!/bin/bash

if [ -z "$1" ]; then
    echo "Please provide the path to the input root folder. Example: sandbox/maps/benchmark/furnished"
    exit 1
fi

ROOT=$1
DEBUG=${2:-"--debug"}

base=$(basename ${ROOT})
echo "################################################################"
echo "  .. processing: ${ROOT}"
echo "  .. writing outputs to: sandbox/out/${base}"
echo "  .. using config file: $ROOT/config.yml"
echo "################################################################"

logsdir="logs/$base"
rm -rf $logsdir
mkdir -p $logsdir
echo "Writing logs to $logsdir"

find $ROOT/* -type d | parallel --bar --jobs 12 python -m py_floor_plan_segmenter -c $ROOT/config.yml -i {} -p sandbox/out/$base $DEBUG ">" $logsdir/{/}.log
