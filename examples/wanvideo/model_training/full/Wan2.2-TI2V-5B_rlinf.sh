CUDA_VISIBLE_DEVICES=1,6,7 accelerate launch \
  --config_file /data/zsq/diffsynth-studio-rlinf/examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  /data/zsq/diffsynth-studio-rlinf/examples/wanvideo/model_training/train_rlinf.py \
  --height 480 \
  --width 640 \
  --num_frames 13 \
  --dataset_repeat 1 \
  --model_paths '[
    ["/data/zsq/Wan_model/diffusion_pytorch_model-00001-of-00003.safetensors",
     "/data/zsq/Wan_model/diffusion_pytorch_model-00002-of-00003.safetensors",
     "/data/zsq/Wan_model/diffusion_pytorch_model-00003-of-00003.safetensors"],
    "/data/zsq/Wan_model/Wan2.2_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 100000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/data/zsq/diffsynth-studio-rlinf/outputs/AgiBotWCAbs" \
  --trainable_models "dit" \
  --context_noise_sigma 0.0 \
  --static_video_prob 0.05 \
  --extra_inputs "input_image,action" \
  --val_interval 50 \
  --save_epochs 1 \
  --dataset AgiBotWCAbsDataset \
  --dataset_base_path /data/zsq/AgiBotWorldChallenge2026/WorldModel