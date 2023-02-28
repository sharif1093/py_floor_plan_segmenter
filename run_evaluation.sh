#!/bin/bash

TYPE=$1

OUT_DIR=sandbox/eval/${TYPE}
CSV_FILE=${OUT_DIR}/results.csv

echo "Writing output results to ${OUT_DIR}"

mkdir -p ${OUT_DIR}
echo "base_name,n_clusters_true,n_clusters_pred,over_painting,under_painting,over_segmentation_score,completeness_score,over_mixing_score,homogeneity_score,v_score,ami_score,nmi_score,ari_score,fmi_score" > ${CSV_FILE}

for d in sandbox/out/benchmark/$1/*/ ; do
    echo "  .. processing: $d"
    BUILDING=$(basename ${d})

    python -m py_floor_plan_segmenter.evaluate -i sandbox/out/benchmark/$TYPE/$BUILDING/sigma=1.0,0.5 -p sandbox/eval/$TYPE -g sandbox/maps/benchmark/groundtruth/$BUILDING >> ${CSV_FILE}
done
