CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train_rlinf.py \
  --height 256 \
  --width 256 \
  --num_frames 13 \
  --dataset_repeat 1 \
  --model_paths '[
    ["/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/diffusion_pytorch_model-00001-of-00003.safetensors",
     "/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/diffusion_pytorch_model-00002-of-00003.safetensors",
     "/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/diffusion_pytorch_model-00003-of-00003.safetensors"],
    "/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/Wan2.2_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 100000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "outputs/only4test" \
  --trainable_models "dit" \
  --context_noise_sigma 0.0 \
  --static_video_prob 0.05 \
  --extra_inputs "input_image,action" \
  --val_interval 50 \
  --save_epochs 50 \
  --dataset RLinfNpyDataset \
  --dataset_base_path /mnt/project_rlinf/jzn/workspace/latest/RLinf/dataset_for_posttrain_worldmodel_libero_spatial/base_policy_rollout