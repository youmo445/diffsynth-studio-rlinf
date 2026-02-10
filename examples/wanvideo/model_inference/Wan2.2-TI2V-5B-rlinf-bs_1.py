import torch
try:
    torch.serialization.add_safe_globals(['set', 'OrderedDict', 'builtins.set'])
except AttributeError:
    pass
import time
from PIL import Image, ImageDraw
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import numpy as np
import gc




model_ckpt_path = "/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio_zsq/outputs/0201-object/epoch-2199.safetensors"
lora_rows = [
    (None, "Full Model"),
]

# (steps, tea_cache_l1_thresh, cfg_scale, shift)
inference_configs = [
    (5, 0.0, 1.0, 5.0),
]


window = 9
total_frames = 505


rgb = np.load("/mnt/project_rlinf/jzn/workspace/latest/RLinf/dataset_for_posttrain_worldmodel_libero_object/base_policy_rollout/val_data/step_0/seed_1/rgb.npy")
video_np = rgb[:, 0]  # [num_frames, 3, H, W]
print(f'video:{video_np.shape}')
gt_video_full = []
for frame_img in video_np:
    if frame_img.max() <= 1.0:
        frame_img = (frame_img * 255).clip(0, 255)
    frame_img = frame_img.astype(np.uint8)
    gt_video_full.append(Image.fromarray(frame_img))
action_full = np.load("/mnt/project_rlinf/jzn/workspace/latest/RLinf/dataset_for_posttrain_worldmodel_libero_object/base_policy_rollout/val_data/step_0/seed_1/actions.npy")[:,0]
action_full = torch.from_numpy(action_full).float()
gt_gripper_all = action_full[:, -1].clone() 

if isinstance(action_full, torch.Tensor):
    action_full = action_full.to(dtype=torch.bfloat16, device="cuda:0")

def add_label(img: Image.Image, text: str):
    W, H = img.size
    bar_h = 100 
    new_img = Image.new("RGB", (W, H + bar_h), color=(255, 255, 255))
    new_img.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(new_img)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        draw.text((10, 5 + i * 22), line, fill=(0, 0, 0))
    return new_img


def to_numpy_img(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if isinstance(img, Image.Image):
        img = np.array(img)
    return img.astype(np.uint8)

def get_pipeline(lora_path):
    print(f"Loading pipeline with LoRA: {lora_path}")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        model_configs=[
            ModelConfig(path=model_ckpt_path, offload_device="cpu"),
            ModelConfig(path="/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/Wan2.2_VAE.pth", offload_device="cpu"),
        ],
    )
    if lora_path:
        pipe.load_lora(pipe.dit, lora_path, alpha=1)
    
    pipe.dit.to("cuda")
    pipe.vae.to("cuda")
    return pipe

all_results = []

num_iters = (total_frames - 1) // (window - 1)
print("Total rolling chunks:", num_iters)

for lora_path, lora_name in lora_rows:
    pipe = get_pipeline(lora_path)
    
    row_results = []
    for steps, tea_thresh, cfg_scale, shift in inference_configs:
        print(f"\n=== Processing {lora_name} | steps={steps}, tea={tea_thresh}, cfg={cfg_scale}, shift={shift} ===")
        start_time = time.time()
        generated_frames = []

        input_image = gt_video_full[0]
        input_image4 = [input_image] * 4

        for i in range(num_iters):
            print(f"  Chunk {i+1}/{num_iters}")

            start = i * (window - 1)                   
            end = start + window 
            if start == 0:

                idx = [0] * 5 + list(range(1, window))
            else:
                idx = [0] + list(range(start - 3, end))
            idx = np.array(idx)
            print(f"context_idx:{idx[:5]}, predict_idx:{idx[5:]}")
            action_full[0] = 0
            action_full[0, -1] = -1
            action = action_full[idx]
            if isinstance(action, torch.Tensor):
                action = action.to(dtype=torch.bfloat16, device="cuda:0")



            kwargs = {
                "seed": 0,
                "tiled": False,
                "input_image": input_image,
                "input_image4": input_image4,
                "action": action,
                "height": 256, "width": 256,
                "num_frames": window + 4,
                "num_inference_steps": steps,
                "cfg_scale": cfg_scale,
                "sigma_shift": shift,
                "bs_1": True,
            }
            if tea_thresh > 0:
                raise NotImplementedError('已不支持')

            out_video = pipe(**kwargs)[0]

            if i == 0:
                generated_frames.extend(
                    [out_video[0]] + out_video[-8:]
                )
            else:
                generated_frames.extend(out_video[5:])
            
            if i == num_iters - 1:
                final_ref_frame = out_video[0]
            input_image4 = out_video[-4:]


        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Generated {len(generated_frames)} frames, trimming to {total_frames} frames.")
        generated_frames = generated_frames[:total_frames]
        
        row_results.append({
            "frames": generated_frames,
            "time": elapsed_time,
            "steps": steps,
            "tea": tea_thresh,
            "cfg": cfg_scale,
            "shift": shift,
        })
        print(f"Done. Time: {elapsed_time:.2f}s")
        
    all_results.append((lora_name, row_results))
    
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


concat_frames = []

chunk_end_frames = [(i + 1) * (window - 1) for i in range(num_iters)]
chunk_end_frames = [x for x in chunk_end_frames if x < total_frames]


for t in range(total_frames):
    
    rows_imgs = []
    
    for lora_name, row_res in all_results:
        gt_grip = gt_gripper_all[t].item()
        gt_status = "CLOSE" if gt_grip > 0 else "OPEN"
        row_imgs = [add_label(gt_video_full[t], f"FRAME: {t}\nGT GRIPPER: {gt_status}\nValue: {gt_grip:.1f}")]
        
        for res in row_res:
            frame = res["frames"][t]
            steps = res["steps"]
            tea = res["tea"]
            cfg = res["cfg"]
            time_cost = res["time"]
            shift = res["shift"]
            curr_grip = action_full[t, -1].item()
            curr_status = "CLOSE" if curr_grip > 0 else "OPEN"
            label = f"FRAME: {t} | GRIPPER: {curr_status}\nValue: {curr_grip:.1f}\n{lora_name} (s={steps}, cfg={cfg}, shift={shift})"
            row_imgs.append(add_label(frame, label))
            
        row_np = [to_numpy_img(im) for im in row_imgs]
        min_h = min(arr.shape[0] for arr in row_np)
        row_np = [arr[:min_h] for arr in row_np]
        merged_row = np.concatenate(row_np, axis=1)
        rows_imgs.append(merged_row)

    min_w = min(arr.shape[1] for arr in rows_imgs)
    rows_imgs = [arr[:, :min_w] for arr in rows_imgs]
    
    final_frame_np = np.concatenate(rows_imgs, axis=0)
    final_frame_img = Image.fromarray(final_frame_np)

    concat_frames.append(final_frame_img)


save_video(concat_frames, "concat_rows_5B.mp4", fps=30, quality=5)
print("\n===> Done! Saved concat_lora_rows_5B.mp4")

