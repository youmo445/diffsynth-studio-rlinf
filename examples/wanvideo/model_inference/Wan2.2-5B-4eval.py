import os
import torch
import numpy as np
from PIL import Image
from diffsynth.pipelines.wan_video_new_bs1 import WanVideoPipeline, ModelConfig
from diffsynth import save_video
from tqdm import tqdm
import multiprocessing
import argparse


# =========================================================
# 0. 参数解析
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda:1")
args_cli = parser.parse_args()

# =========================================================
# 1. 世界模型加载
# =========================================================
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=args_cli.device,
    model_configs=[
        ModelConfig(
            path="/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/epoch-250.safetensors",
            offload_device="cpu"
        ),
        ModelConfig(
            path="/mnt/project_rlinf/jzn/workspace/DiffSynth-Studio/ckpt/Wan2.2_VAE.pth",
            offload_device="cpu"
        ),
    ],
)
# pipe.load_lora(
#     pipe.dit,
#     "/opt/zsq/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_lora_action_num_frames_13_enhance_begin_more_noise/epoch-416.safetensors",
#     alpha=1,
# )
# pipe.enable_vram_management()
pipe.dit.to(args_cli.device)
pipe.vae.to(args_cli.device)

# =========================================================
# 2. Helpers
# =========================================================
def load_gt_npy_folder(folder):
    rgb = np.load(os.path.join(folder, "rgb.npy"))   # [T,3,H,W]
    ak = np.load(os.path.join(folder, "actions.npy"))  # [T,action_dim]

    video = []
    for frame in rgb:
        img = frame
        if img.max() <= 1:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        img = np.transpose(img, (1, 2, 0))
        video.append(Image.fromarray(img))

    return video, ak


def save_frames(frames, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, frame in enumerate(tqdm(frames, desc="Saving images")):
        frame.save(os.path.join(save_path, f"frame_{i:04d}.png"))


# =========================================================
# 3. 自回归生成（严格输出 action_len 帧）
# =========================================================
def generate_sequence(rgb_list, actions, window=9, steps=5):
    action_len = len(actions)
    action_window = 8
    print(f"动作帧数 = {action_len}")

    # chunk 数：覆盖 action_len
    num_iters = action_len  // (window - 1)
    if num_iters < 1:
        num_iters = 1

    print(f"Rolling chunks = {num_iters}")

    generated_frames = []
    input_image = rgb_list[0]
    input_image4 = [input_image] * 4

    for i in range(num_iters):
        print(f'\n--- Chunk {i+1}/{num_iters} ---')
        start = i * (window - 1) # 0, 8, 16, 
        end = start + window # 9, 17, 25, 

        if i == 0:
            # 序号就是0,0,0,0,0,  1,2,3,4,5,6,7,8
            idx = [0] * 5 + list(range(1, window))
        else:
            idx = [0] + list(range(start - 3, end))
        idx = np.array(idx)

        # 使用 idx 提取动作，处理越界情况（使用最后一帧动作填充）
        clamped_idx = np.clip(idx, 0, len(actions) - 1)
        actions[0] = 0
        actions[0, -1] = -1
        act_win = actions[clamped_idx]
        act_win = torch.from_numpy(act_win).to(dtype=torch.bfloat16, device=args_cli.device)

        out_video = pipe(
            seed=0,
            tiled=False,
            input_image=input_image,
            input_image4=input_image4,
            action=act_win,
            height=256,
            width=256,
            num_frames=window + 4,
            num_inference_steps=steps,
            cfg_scale=1.0,
            idx=idx,
        )

        gen_video = out_video[0]

        # 第一个 chunk：加入全 window
        if len(generated_frames) == 0:
            generated_frames.extend([gen_video[0]] + gen_video[-8:])
        else:
            # 后续 chunk：去掉第一帧（避免重复）
            generated_frames.extend(gen_video[5:])

        # 下一个 chunk 的 context
        input_image4 = gen_video[-4:]

    # 输出严格等于动作长度
    generated_frames = generated_frames[:action_len]
    print(f"最终生成帧数 = {len(generated_frames)}")

    return generated_frames


# =========================================================
# 4. 处理单条序列
# =========================================================
def process_one_sequence(gt_root, step, seed, traj, out_root, window=9, steps=5):

    folder = os.path.join(gt_root, f"step_{step}_seed_{seed}_traj_{traj}")
    print(f"\n=== 处理 {folder} ===")

    rgb_list, actions = load_gt_npy_folder(folder)

    # 生成
    gen_frames = generate_sequence(rgb_list, actions, window=window, steps=steps)

    # 输出目录：https://规范结构
    out_dir = os.path.join(out_root, f"step_{step}_seed_{seed}_traj_{traj}")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(out_dir, exist_ok=True)

    # 保存 PNG
    save_frames(gen_frames, img_dir)

    # 保存视频
    video_path = os.path.join(out_dir, "video.mp4")
    save_video(gen_frames, video_path, fps=30, quality=5)

    print(f"保存完成: {video_path}")


# =========================================================
# 5. 批处理
# =========================================================
if __name__ == "__main__":

    gt_root = "/mnt/project_rlinf/jzn/workspace/latest/RLinf/dataset_for_posttrain_worldmodel_libero_spatial/base_policy_rollout/val_data_processed"
    # out_root = "/opt/zsq/DiffSynth-Studio/models/train/5B-TI2V2ckpt-enhance_0118_w_ref_image_w_block_attn/epoch-99.safetensors"
    # out_root = "outputs/wan2.2-5b-full-epoch250-jzn-w_blockattn_w_contextnoise"
    out_root = "outputs/wan2.2-5b-full-epoch250-jzn-wo_blockattn"
    os.makedirs(out_root, exist_ok=True)
    trajs = [0]
    seeds = [0,1,2,3,4,5,6,7]
    # seed = args_cli.seed
    steps = 5
    for traj in trajs:
        for seed in seeds:
            process_one_sequence(gt_root, 0, seed, traj, out_root, steps=steps)
