#!/bin/bash

start_time=$SECONDS

echo "running pre-trained model......"

python main.py --dataroot ../../input_data/ \
--outroot results/ \
--pathway KEGG \
--foldtype pair \
--avg_by drug

echo "finished run"
elapsed=$(( SECONDS - start_time ))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
