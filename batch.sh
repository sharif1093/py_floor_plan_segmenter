#!/bin/bash

for d in sandbox/maps/$1/*/ ; do
    echo "$d"
    time python -m py_floor_plan_segmenter -i $d -p sandbox/out/$1 --debug
done
