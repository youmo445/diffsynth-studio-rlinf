import torch, os, json
import numpy as np
from PIL import Image
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.zsq_single_npy_dataset import MyNpyDatasetnew
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Patch Start: 允许加载包含 set 的权重文件 ---
try:
    torch.serialization.add_safe_globals(['set', 'OrderedDict', 'builtins.set'])
except AttributeError:
    pass
# --- Patch End ---



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        context_noise_sigma=0.0, # 新增参数
        static_video_prob=0.0, # 新增参数
        use_wow_checkpoint=False, # 新增参数
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(model_id=audio_processor_config.split(":")[0], origin_file_pattern=audio_processor_config.split(":")[1])
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, audio_processor_config=audio_processor_config)
        
        # --- Patch Start: 加载 WoW 权重 ---
        if use_wow_checkpoint:
            wow_ckpt_path = "/opt/zsq/wow-world-model/dit_models/checkpoints/WoW-1-Wan-14B-600k/WoW_video_dit.pt"
            if os.path.exists(wow_ckpt_path):
                print(f"🎯 Overwriting with WoW checkpoint: {wow_ckpt_path}")
                # 使用 weights_only=False 避开报错
                state_dict = torch.load(wow_ckpt_path, map_location="cpu", weights_only=False)
                # 覆盖 pipe.dit 的权重
                msg = self.pipe.dit.load_state_dict(state_dict, strict=False)
                print(f"✅ WoW checkpoint loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
            else:
                print(f"⚠️ WoW checkpoint not found at {wow_ckpt_path}, skipping.")
        else:
            print("🎯 Not using WoW checkpoint.")
        # --- Patch End ---
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.context_noise_sigma = context_noise_sigma # 保存参数
        self.static_video_prob = static_video_prob # 保存参数

        
    def forward_preprocess(self, data):
        # === 新增：静态样本增强 ===
        # 如果启用，随机将当前样本变为“完全静止”，强迫模型学习背景保持
        if self.training and self.static_video_prob > 0 and np.random.rand() < self.static_video_prob:
            first_frame = data["video"][0]
            data["video"] = [first_frame] * len(data["video"])
            if "action" in data:
                data["action"] = torch.zeros_like(data["action"])
                data["action"][:,-1] = -1  # 最后一维设为 -1
        # ============================
        # CFG-sensitive parameters
        # inputs_posi = {"prompt": data["prompt"]}
        inputs_posi = {}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "idx":data["idx"] if "idx" in data else None,
        }
        
        # Extra inputs
        # print(f'====WanTrainingModule forward_preprocess extra_inputs: {self.extra_inputs}====')
        # control_video, reference_image, etc.
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                # 在这里给 context frame 加噪
                if self.context_noise_sigma > 0:
                    img_arr = np.array(data["video"][0]).astype(np.float32)
                    noise = np.random.normal(0, self.context_noise_sigma, img_arr.shape)
                    img_noisy = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
                    inputs_shared["input_image"] = Image.fromarray(img_noisy)
                else:
                    inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--context_noise_sigma", type=float, default=10, help="Sigma of Gaussian noise added to the context frame")
    parser.add_argument("--static_video_prob", type=float, default=0.15, help="Probability of replacing the sample with a static video (action=0)")
    parser.add_argument("--use_wow_checkpoint", action="store_true",help="Whether to load the WoW checkpoint to overwrite the base model weights.")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval in epochs")
    parser.add_argument("--dataset",type=str,default="MyNpyDataset",help="Dataset type for training")
    args = parser.parse_args()

    if args.dataset == "MyNpyDatasetnew":
        dataset = MyNpyDatasetnew(
            base_path='/mnt/project_rlinf/jzn/workspace/latest/RLinf/dataset_for_posttrain_worldmodel_libero_object/base_policy_rollout/train_data',
            repeat=args.dataset_repeat,
            num_frames=args.num_frames,
        )
        val_dataset = MyNpyDatasetnew(
            base_path='/mnt/project_rlinf/jzn/workspace/latest/RLinf/dataset_for_posttrain_worldmodel_libero_object/base_policy_rollout/val_data',
            repeat=1, # 验证集不需要重复
            num_frames=args.num_frames # 保持与训练一致
        )
    else:
        raise NotImplementedError('this dataset type not implemented')
    # ----------------------
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        context_noise_sigma=args.context_noise_sigma, 
        static_video_prob=args.static_video_prob, 
        use_wow_checkpoint=args.use_wow_checkpoint, 
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, val_dataset, model, model_logger, args=args)
