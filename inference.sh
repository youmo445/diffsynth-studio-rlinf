#!/bin/bash

python examples/wanvideo/model_inference/Wan2.2-5B-4eval_w_fixidx_w_ref_img_wo_blockattn.py --seed 0 --device cuda:0 &
python examples/wanvideo/model_inference/Wan2.2-5B-4eval_w_fixidx_w_ref_img_wo_blockattn.py --seed 1 --device cuda:1 &
python examples/wanvideo/model_inference/Wan2.2-5B-4eval_w_fixidx_w_ref_img_wo_blockattn.py --seed 2 --device cuda:2 &
python examples/wanvideo/model_inference/Wan2.2-5B-4eval_w_fixidx_w_ref_img_wo_blockattn.py --seed 3 --device cuda:3 &
python examples/wanvideo/model_inference/Wan2.2-5B-4eval_w_fixidx_w_ref_img_wo_blockattn.py --seed 4 --device cuda:4 &
python examples/wanvideo/model_inference/Wan2.2-5B-4eval_w_fixidx_w_ref_img_wo_blockattn.py --seed 5 --device cuda:5 &
python examples/wanvideo/model_inference/Wan2.2-5B-4eval_w_fixidx_w_ref_img_wo_blockattn.py --seed 6 --device cuda:6 &
python examples/wanvideo/model_inference/Wan2.2-5B-4eval_w_fixidx_w_ref_img_wo_blockattn.py --seed 7 --device cuda:7 &
wait

