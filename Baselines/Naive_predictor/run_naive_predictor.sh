#!/bin/bash

start_time=$SECONDS

echo "running pre-trained model......"

python main.py --dataroot ../../../datasets/cell_viability/CCLE/curveCurator/matrixes_raw/ \
--outroot results_ec50_curated/ \
--foldtype all \
--avg_by both

echo "finished run"
elapsed=$(( SECONDS - start_time ))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
