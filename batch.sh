#!/bin/bash

for d in sandbox/maps/$1/*/ ; do
    echo "$d"
    python -m py_floor_plan_segmenter -i $d -p sandbox/out/$1 --debug
    echo "============================"
    echo
done
