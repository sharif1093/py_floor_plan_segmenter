#!/bin/bash

if [ -z "$1" ]; then
    echo "Please provide the path to the output root folder. Example: sandbox/out/furnished"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Please provide the path to the ground-truth root folder. Example: sandbox/maps/benchmark/groundtruth"
    exit 1
fi

OUT=$1
GROUNDTRUTH=$2
TYPE=$(basename ${OUT})
OUT_DIR=sandbox/eval/${TYPE}
CSV_FILE=${OUT_DIR}/results.csv

echo "################################################################"
echo "  .. processing: ${OUT}"
echo "  .. writing output results to ${OUT_DIR}"
echo "  .. the output csv file is ${CSV_FILE}"
echo "################################################################"
mkdir -p ${OUT_DIR}

echo "base_name,n_clusters_true,n_clusters_pred,over_painting,under_painting,over_segmentation_score,completeness_score,over_mixing_score,homogeneity_score,v_score,ami_score,nmi_score,ari_score,fmi_score,iou_score" > ${CSV_FILE}

for d in $OUT/*/ ; do
    echo "  .. processing: $d"
    BUILDING=$(basename ${d})
    python -m py_floor_plan_segmenter.evaluate -i sandbox/out/$TYPE/$BUILDING/sigma=1.0,0.5 -g sandbox/maps/benchmark/groundtruth/$BUILDING -p sandbox/eval/$TYPE >> ${CSV_FILE}
done

