import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.zsq_single_npy_dataset import MyNpyDatasetnew
from PIL import Image, ImageDraw

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(path="/opt/zsq/Wan2.1/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
    ],
)

dataset = MyNpyDatasetnew(num_frames=13)
batch = dataset[0]
gt_video = batch["video"]
gt_video = gt_video + gt_video 

pipe.vae.to(device=pipe.device, dtype=pipe.torch_dtype).eval()
input_video = pipe.preprocess_video(gt_video)
print(f'Input video shape: {input_video.shape}, dtype: {input_video.dtype}, device: {input_video.device}')
input_latents = pipe.vae.encode(input_video, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
print(f'Input latents shape: {input_latents.shape}, dtype: {input_latents.dtype}, device: {input_latents.device}')
timesteps = pipe.scheduler.timesteps.to(dtype=pipe.torch_dtype, device=pipe.device)
timesteps = timesteps[-10:]
print(f'timesteps: {timesteps.shape}')


shape = input_latents.shape
seed = 42
noise = pipe.generate_noise(shape, seed=seed, rand_torch_dtype=pipe.torch_dtype,rand_device=pipe.device)

# Store all rows of images here
image_rows = []
row_labels = []

# 1. Decode original latents (GT)
rec_video = pipe.vae.decode(input_latents, device=pipe.device)
rec_images = pipe.vae_output_to_video(rec_video)
print(f'Reconstructed images from original latents.')
input('x')
image_rows.append(rec_images)
row_labels.append("GT")

# 2. Add noise and decode at different timesteps
# Select indices to visualize: e.g., 5 evenly spaced steps
num_steps = len(timesteps)
indices_to_visualize = torch.linspace(0, num_steps - 1, steps=10).long().tolist()

for idx in indices_to_visualize:
    t = timesteps[idx]
    print(f"Processing timestep index {idx}, value {t.item():.2f}")
    
    noisy_latents = pipe.scheduler.add_noise(input_latents, noise, t).to(dtype=pipe.torch_dtype, device=pipe.device)
    noisy_video = pipe.vae.decode(noisy_latents, device=pipe.device)
    noisy_images = pipe.vae_output_to_video(noisy_video)
    
    image_rows.append(noisy_images)
    row_labels.append(f"t={t.item():.2f}")

# 3. Create long image
if image_rows and image_rows[0]:
    frame_width, frame_height = image_rows[0][0].size
    num_frames = len(image_rows[0])
    num_rows = len(image_rows)
    
    total_width = frame_width * num_frames
    total_height = frame_height * num_rows
    
    new_im = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(new_im)
    
    for row_idx, (images, label) in enumerate(zip(image_rows, row_labels)):
        for col_idx, im in enumerate(images):
            x = col_idx * frame_width
            y = row_idx * frame_height
            new_im.paste(im, (x, y))
            
            # Draw label on the first frame of each row
            if col_idx == 0:
                draw.text((x + 10, y + 10), label, fill="red")

    output_filename = "test_VAE_multistep.png"
    new_im.save(output_filename)