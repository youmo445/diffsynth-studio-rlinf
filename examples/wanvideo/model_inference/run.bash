#!/bin/bash

# 8张卡，每张卡负责一个seed (0-7)
for i in {0..7}
do
    echo "Starting inference for seed $i on GPU $i"
    CUDA_VISIBLE_DEVICES=$i python /mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/examples/wanvideo/model_inference/Wan2.2-5B-eval-ref.py --seed $i --device cuda:0 &
done

wait
echo "All inference tasks completed."
