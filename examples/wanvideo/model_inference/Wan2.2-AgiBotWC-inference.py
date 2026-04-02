import torch
import os
import cv2
import h5py
import numpy as np
import time
import gc
from PIL import Image, ImageDraw
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ================= Config =================
# TODO: Update your model checkpoint path
model_ckpt_path = "/data/zsq/diffsynth-studio-rlinf/outputs/only4test/epoch-61.safetensors" # Placeholder, update to your 5B RLinf model
vae_ckpt_path = "/data/zsq/Wan_model/Wan2.2_VAE.pth"
dataset_base_path = "/data/zsq/AgiBotWorldChallenge2026/WorldModel/mnt/public/chenshengcong/dataset/iros_challenge_2025_acwm/train"

# Inference parameters
inference_configs = [
    (5, 0.0, 1.0, 5.0), # steps, tea_cache_l1_thresh, cfg_scale, shift
]

window = 9 # Number of frames per chunk
# ==========================================

def get_pipeline(lora_path=None):
    print(f"Loading pipeline...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        model_configs=[
            ModelConfig(path=model_ckpt_path, offload_device="cpu"),
            ModelConfig(path=vae_ckpt_path, offload_device="cpu"),
        ],
    )
    if lora_path:
        print(f"Loading LoRA: {lora_path}")
        pipe.load_lora(pipe.dit, lora_path, alpha=1)
    
    pipe.dit.to("cuda")
    pipe.vae.to("cuda")
    return pipe

def load_agibot_episode(base_path, ep_name):
    ep_path = os.path.join(base_path, ep_name)
    # Load video
    video_path = os.path.join(ep_path, "head_color.mp4")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    # Load actions/proprio
    h5_path = os.path.join(ep_path, "proprio_stats.h5")
    with h5py.File(h5_path, 'r') as f:
        pos = f['state/end/position'][:]       # [T, 2, 3]
        quat = f['state/end/orientation'][:]   # [T, 2, 4]
        grip = f['state/effector/position'][:] # [T, 2]
    
    T = min(len(frames), pos.shape[0])
    frames = frames[:T]
    pos = pos[:T]
    quat = quat[:T]
    grip = grip[:T]

    # Calculate actions as deltas (matching AgiBotWCDataset logic)
    F = T
    actions = np.zeros((F, 14), dtype=np.float32)
    for t in range(1, F):
        for arm in range(2):
            # position delta
            dp = pos[t, arm] - pos[t-1, arm]
            # rotation delta
            R_t = Rotation.from_quat(quat[t, arm]).as_matrix()
            R_prev = Rotation.from_quat(quat[t-1, arm]).as_matrix()
            R_delta = R_t @ R_prev.T
            rpy = Rotation.from_matrix(R_delta).as_euler("xyz")
            # gripper (normalized by 120.0 as in dataset)
            g = grip[t, arm] / 120.0
            actions[t, arm*7:(arm+1)*7] = np.concatenate([dp, rpy, [g]])
    # First action is zero
    actions[0] = 0
    return frames, torch.from_numpy(actions).to(dtype=torch.bfloat16, device="cuda:0")

def add_label(img: Image.Image, text: str):
    W, H = img.size
    bar_h = 120 
    new_img = Image.new("RGB", (W, H + bar_h), color=(255, 255, 255))
    new_img.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(new_img)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        draw.text((10, 5 + i * 20), line, fill=(0, 0, 0))
    return new_img

def run_inference():
    # Select first episode for test
    episodes = [d for d in sorted(os.listdir(dataset_base_path)) if os.path.isdir(os.path.join(dataset_base_path, d))]
    target_ep = episodes[0]
    print(f"Testing on episode: {target_ep}")
    
    gt_frames, action_full = load_agibot_episode(dataset_base_path, target_ep)
    total_frames = len(gt_frames)
    num_iters = (total_frames - 1) // (window - 1)
    print(f"Total frames: {total_frames}, Rolling chunks: {num_iters}")

    pipe = get_pipeline()
    
    for steps, tea_thresh, cfg_scale, shift in inference_configs:
        print(f"\n=== Processing | steps={steps}, cfg={cfg_scale}, shift={shift} ===")
        start_time = time.time()
        generated_frames = []

        input_image = gt_frames[0]
        input_image4 = [input_image] * 4

        for i in tqdm(range(num_iters)):
            start = i * (window - 1)                   
            end = start + window 
            
            # Indexing logic from baseline
            if start == 0:
                idx = [0] * 5 + list(range(1, window))
            else:
                idx = [0] + list(range(start - 3, end))
            idx = np.array(idx)
            
            # Ensure index within bounds
            idx = np.clip(idx, 0, total_frames - 1)
            
            action = action_full[idx]
            print(f"Chunk {i+1}/{num_iters} | Frames: {idx} | Action shape: {action.shape}")
            kwargs = {
                "seed": 0,
                "tiled": False,
                "input_image": input_image,
                "input_image4": input_image4,
                "action": action,
                "height": 480, "width": 640,
                "num_frames": window + 4,
                "num_inference_steps": steps,
                "cfg_scale": cfg_scale,
                "sigma_shift": shift,
                "bs_1": True,
            }

            out_video = pipe(**kwargs)[0]

            if i == 0:
                generated_frames.extend([out_video[0]] + out_video[-8:])
            else:
                generated_frames.extend(out_video[5:])
            
            input_image4 = out_video[-4:]

        elapsed_time = time.time() - start_time
        print(f"Generated {len(generated_frames)} frames. Time: {elapsed_time:.2f}s")
        
        # Save comparison video
        concat_frames = []
        for t in range(min(total_frames, len(generated_frames))):
            gt_f = add_label(gt_frames[t], f"FRAME: {t}\nGROUND TRUTH")
            gen_f = add_label(generated_frames[t], f"FRAME: {t}\nPREDICTION\nsteps={steps}, cfg={cfg_scale}")
            
            combined = Image.new("RGB", (gt_f.width * 2, gt_f.height))
            combined.paste(gt_f, (0, 0))
            combined.paste(gen_f, (gt_f.width, 0))
            concat_frames.append(combined)
            
        output_name = f"agibot_inference_result.mp4"
        save_video(concat_frames, output_name, fps=30, quality=5)
        print(f"Saved {output_name}")

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    run_inference()
