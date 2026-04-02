import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
from functools import partial

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..models.wan_video_mot import MotWanModel

from ..schedulers.flow_match import FlowMatchScheduler
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
import time  
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns


class L1AnalysisCollector:
    """一个用于在推理过程中收集数据并计算相对L1距离的辅助类。"""
    def __init__(self):
        self.history = {}
        # self.distances 存储当前正在进行的这一次推理的数据
        self.distances = {}
        # self.all_sessions 存储所有推理的历史数据： {"t_mod": [[run1_step1, ...], [run2_step1, ...]], ...}
        self.all_sessions = {}
        self.clear()

    def clear(self):
        self.history = {
            "hidden_states_before_head": None,
            "t_mod": None,
            "model_residual": None,
        }
        self.distances = {
            "hidden_states_before_head": [],
            "t_mod": [],
            "model_residual": [],
        }
        self.all_sessions = {}
    def start_new_session(self):
        """开始一次新的推理记录，将上一次的数据归档。"""
        if any(len(v) > 0 for v in self.distances.values()):
            for key, val in self.distances.items():
                if key not in self.all_sessions:
                    self.all_sessions[key] = []
                self.all_sessions[key].append(val)
        
        self.history = {
            "hidden_states_before_head": None,
            "t_mod": None,
            "model_residual": None,
        }
        self.distances = {
            "hidden_states_before_head": [],
            "t_mod": [],
            "model_residual": [],
        }
    def update(self, step_name: str, current_tensor: torch.Tensor):
        """
        更新一个张量的状态，如果历史记录存在，则计算并存储L1距离。
        """
        if self.history.get(step_name) is not None:
            previous_tensor = self.history[step_name]
            
            # 确保张量在同一设备上
            current_tensor_cpu = current_tensor.detach().cpu().float()
            previous_tensor_cpu = previous_tensor.detach().cpu().float()

            # 计算相对L1距离
            # 公式: ||curr - prev||_1 / ||prev||_1
            # 这等价于: mean(abs(curr - prev)) / mean(abs(prev))
            l1_diff = torch.linalg.norm((current_tensor_cpu - previous_tensor_cpu).flatten(), ord=1)
            l1_norm_prev = torch.linalg.norm(previous_tensor_cpu.flatten(), ord=1)
            
            # 防止除以零
            relative_l1 = (l1_diff / (l1_norm_prev + 1e-6)).item()
            
            # 【修改】如果 key 不存在，先初始化列表
            if step_name not in self.distances:
                self.distances[step_name] = []
            self.distances[step_name].append(relative_l1)

        # 更新历史记录为当前张量，为下一步做准备
        self.history[step_name] = current_tensor.detach().clone()
    def get_average_distances(self):
        """计算多次推理的平均值和标准差。"""
        # 先把当前正在进行的这次（如果还没归档）也归档进去
        temp_sessions = self.all_sessions.copy()
        if any(len(v) > 0 for v in self.distances.values()):
            for key, val in self.distances.items():
                if key not in temp_sessions:
                    temp_sessions[key] = []
                temp_sessions[key].append(val)
        
        avg_results = {}
        std_results = {}
        
        for key, sessions in temp_sessions.items():
            if not sessions:
                continue
            # 找出最小长度，对齐数据
            min_len = min(len(s) for s in sessions)
            # 截断并转换为 numpy 数组: [num_sessions, num_steps]
            data_matrix = np.array([s[:min_len] for s in sessions])
            
            avg_results[key] = np.mean(data_matrix, axis=0)
            std_results[key] = np.std(data_matrix, axis=0)
            
        return avg_results, std_results
def plot_l1_distances(collector: L1AnalysisCollector, num_steps: int, save_path="l1_distance_analysis.png"):
    """根据收集到的距离数据绘图并保存。支持多次推理平均。"""
    avg_data, std_data = collector.get_average_distances()
    
    if not avg_data:
        print("No data to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    first_key = list(avg_data.keys())[0]
    actual_steps = len(avg_data[first_key])
    diffusion_process_steps = np.linspace(0, 100, num=actual_steps)

    for name, mean_vals in avg_data.items():
        std_vals = std_data[name]
        ax.plot(diffusion_process_steps, mean_vals, label=f"{name} (mean)", marker='o', markersize=3, linestyle='-')
        ax.fill_between(diffusion_process_steps, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)

    ax.set_xlabel("Diffusion Process (%)")
    ax.set_ylabel("Relative L1 Distance")
    ax.set_title(f"Relative L1 Distance (Averaged over {len(collector.all_sessions.get(first_key, [])) + 1} runs)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"L1 distance analysis plot saved to: {save_path}")



class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None, model_type="Wan2.1-T2V-14B"):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.model_type = model_type
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        # self.action_embedder = ActionEmbedder(action_dim=8, embed_dim=1536)
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.vap: MotWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter", "vap")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter", "vap")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            # WanVideoUnit_ActionEmbedder(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_TeaCache(),
        ]
        self.model_fn = model_fn_wan_video
        #【新增】初始化分析收集器
        self.analysis_collector = L1AnalysisCollector()

        
    def training_loss(self, **inputs):
        # print(f'====WanVideoPipeline training_loss')
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)

        if (self.dit.TI2V1 or self.dit.TI2V2 or self.dit.TI2V3) and self.dit.one_frame_condition:
            raise NotImplementedError("已弃用，请迁移至[five_frame_condition]")
            rand_idx = torch.tensor([max(0, len(self.scheduler.timesteps) - 50)])
            small_timestep = self.scheduler.timesteps[rand_idx].to(dtype=self.torch_dtype, device=self.device)
            
            first_frame_input = inputs["input_latents"][:, :, 0:1]
            first_frame_noise = inputs["noise"][:, :, 0:1]
            
            first_frame_noisy = self.scheduler.add_noise(first_frame_input, first_frame_noise, small_timestep)
            
            inputs["latents"][:, :, 0:1] = first_frame_noisy
        # ==============================================================
        if (self.dit.TI2V1 or self.dit.TI2V2 or self.dit.TI2V3) and self.dit.five_frame_condition:
            inputs["latents"][:, :, 0:1] = inputs["input_latents"][:, :, 0:1]
            

            rand_idx = torch.tensor([max(0, len(self.scheduler.timesteps) - 50)])
            small_timestep = self.scheduler.timesteps[rand_idx].to(dtype=self.torch_dtype, device=self.device)

            
            context_frames_input = inputs["input_latents"][:, :, 1:2]
            context_frames_noise = inputs["noise"][:, :, 1:2]
            
            context_frames_noisy = self.scheduler.add_noise(context_frames_input, context_frames_noise, small_timestep)
            inputs["latents"][:, :, 1:2] = context_frames_noisy


        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        for name, t in [
            ("input_latents", inputs["input_latents"]),
            ("noise", inputs["noise"]),
            ("latents", inputs["latents"]),
            ("training_target", training_target),
        ]:
            assert torch.isfinite(t).all(), f"{name} has NaN/Inf, min={t.min()}, max={t.max()}"
        noise_pred = self.model_fn(**inputs, timestep=timestep)
        assert torch.isfinite(noise_pred).all(), f"noise_pred has NaN/Inf, min={noise_pred.min()}, max={noise_pred.max()}"

        if (self.dit.TI2V1 or self.dit.TI2V2 or self.dit.TI2V3):
            loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction='none')
            
            mask = torch.zeros_like(loss)
            mask[:, :, -2:] = 1.0
            
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        # ========================================================

        loss = loss * self.scheduler.training_weight(timestep)
        return loss

    
    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            from ..models.longcat_video_dit import LayerNorm_FP32, RMSNorm_FP32
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                    LayerNorm_FP32: AutoWrappedModule,
                    RMSNorm_FP32: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.audio_encoder is not None:
            # TODO: need check
            dtype = next(iter(self.audio_encoder.parameters())).dtype
            enable_vram_management(
                self.audio_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        audio_processor_config: ModelConfig = None,
        model_type: str = "Wan2.2-TI2V-5B",
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        print(f'====WanVideoPipeline from_pretrained====')
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "/opt/zsq/Wan2.1/Wan2.1-I2V-14B-480P",
                "Wan2.1_VAE.pth": "/opt/zsq/Wan2.1/Wan2.1-I2V-14B-480P",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "/opt/zsq/Wan2.1/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)

        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            # print(f'path: {model_config.path}')
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")


        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2


        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str = None,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        input_image4: Optional[list[Image.Image]] = None,
        idx: Optional[np.array] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        action = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # VAP
        vap_video: Optional[list[Image.Image]] = None,
        vap_prompt: Optional[str] = " ",
        negative_vap_prompt: Optional[str] = " ",
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=9,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 1.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # LongCat-Video
        longcat_video: Optional[list[Image.Image]] = None,
        # VAE tiling
        tiled: Optional[bool] = False,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        # 【新增】一个用于控制分析的参数
        analyze_l1_distance: bool = False,
        accumulate_analysis: bool = False, # 新增：是否累积多次推理数据
        # B
        batch_size: Optional[int] = None, 
        bs_1: bool = True,
    ):


        # 【新增】如果需要分析，则清空收集器
        if analyze_l1_distance:
            if not accumulate_analysis:
                # 如果不累积（单次模式），则完全清空
                self.analysis_collector.clear()
            else:
                # 如果累积（多次模式），则开始新会话（归档旧数据）
                self.analysis_collector.start_new_session()

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "action": action,
            "vap_prompt": vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
            # 【新增】传入 analysis_collector，仅在 positive pass 中记录
            "analysis_collector": self.analysis_collector if analyze_l1_distance else None,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "action": torch.zeros_like(action) if action is not None else None,
            "negative_vap_prompt": negative_vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image, "input_image4": input_image4, "batch_size": batch_size, "bs_1": bs_1,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image, "idx": idx, 
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "longcat_video": longcat_video,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
            "vap_video": vap_video, 
        }


        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)


        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        # Denoise
        self.load_models_to_device(self.in_iteration_models)

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
  
            if cfg_scale != 1.0:
                print(f'Performing classifier-free guidance with scale {cfg_scale}...')
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
      
            if "first_frame_latents" in inputs_shared:
                ff = inputs_shared["first_frame_latents"]   # [C, T0, H, W] 或 [B, C, T0, H, W]
                if ff.dim() == 4:
                    ff = ff.unsqueeze(0)  # -> [1, C, T0, H, W]

                T0 = ff.shape[2]

                inputs_shared["latents"][:, :, :T0] = ff
        

        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        if analyze_l1_distance:
            # 【修改】绘制并保存 L1 距离分析图
            if analyze_l1_distance:
                # 定义保存路径
                mode_str = "accumulated" if accumulate_analysis else "single"
                save_plot_path = f"teacache_analysis_{mode_str}.png"
                
                
                plot_l1_distances(
                    collector=self.analysis_collector, # 传入 collector 对象
                    num_steps=num_inference_steps,
                    save_path=save_plot_path
                )
 
        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image", "batch_size"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image, batch_size):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        if batch_size != None:
            shape = (batch_size, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        # print(f'===== WanVideoUnit_InputVideoEmbedder: input_video is None: {input_video is None} =====')
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        if prompt is None or pipe.text_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or pipe.image_encoder is None:
            input('x')
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]

        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}


class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    # 这是把第一帧放在latents的第一个位置上
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride","input_image4"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride,input_image4):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            # print(f'===== WanVideoUnit_ImageEmbedderFused: input_image is None or not fuse_vae_embedding_in_latents =====')
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if input_image4 is None:
            image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
            z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            latents[:, :, 0: 1] = z
        else:
            # input_image:      List[PIL]        length = B
            # input_image4:     List[List[PIL]]  shape = B x 4

            if isinstance(input_image, list):
                input_image5 = [
                    [img] + img4
                    for img, img4 in zip(input_image, input_image4)
                ]
            else:
                input_image5 = [input_image] + input_image4
                input_image5 = [input_image5]
            input_video5 = pipe.preprocess_video(input_image5)
            z = pipe.vae.encode(input_video5, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            latents[:, :, 0: 2] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}


    

class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        print(f'type of reference_image: {type(reference_image)}')
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            print(f'FunReference Unit: image_encoder is None, returning only reference_latents')
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        print(f'dtype of clip_feature: {clip_feature.dtype}, device: {clip_feature.device}')
        print(f'clip_feature shape: {clip_feature.shape}')
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}




class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
            "Wan2.1-I2V-14B-action": [ -1.78542477e+03, 6.25367494e+01, 2.96112008e+01, 6.08822239e-01, 2.34992620e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    action = None,
    idx = None,
    reference_latents = None,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    pose_latents=None,
    face_pixel_values=None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    model_type="Wan2.1-T2V-14B",
    **kwargs,
):
    bs_1 = True if len(action.shape) == 2 else False


    analysis_collector = kwargs.get("analysis_collector", None)
    x = latents


    # Image Embedding
    
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        if context is not None:
            context = torch.cat([clip_embdding, context], dim=1)
        else:
            context = clip_embdding

    action_raw = action.clone()
    if dit.I2V or dit.TI2V2 or dit.TI2V1:
        # get action embedding in shape [B, T, D]
        # action embedding for text like cross-attn
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        action = action.to(dtype=x.dtype, device=x.device)
        if action.dim() == 2:
            action = action.unsqueeze(0)
        action_emb = dit.action_mlp1(action)
        if context is not None:
            context = torch.cat([context, action_emb], dim=1)
        else:
            context = action_emb

    if context is None and dit.TI2V3:
        # get zero context if TI2V3 but no context
        context = torch.zeros(x.shape[0], 1, dit.dim, dtype=x.dtype, device=x.device)
    


    assert torch.isfinite(latents).all(), "latents has NaN/Inf"
    assert torch.isfinite(timestep).all(), "timestep has NaN/Inf"
    if y is not None:
        assert torch.isfinite(y).all(), "y has NaN/Inf"
    if clip_feature is not None:
        assert torch.isfinite(clip_feature).all(), "clip_feature has NaN/Inf"
    
    # Camera control
    # downsample x
    x = dit.patchify(x, control_camera_latents_input)


    if dit.TI2V2 or dit.TI2V3:
        # AdaLN for action embedding
        action = action_raw
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        action = action.to(dtype=context.dtype, device=context.device)

        if not bs_1:
            # batch inference
            B, T, D = action.shape

            first = action[:, 0:1, :]          # [B, 1, D]
            first = first.repeat(1, 3, 1)      # [B, 3, D]

            action = torch.cat([first, action], dim=1)  # [B, T+3, D]
            B, T4, D4 = action.shape
            action = action.reshape(B, T4 // 4, D4 * 4)
            action_emb = dit.action_mlp2(action)

        elif bs_1:
            # single inference or training
            action = torch.cat([action[0:1].repeat(3,1), action], dim=0)
            action = action.reshape(x.shape[2],-1)
            assert torch.isfinite(action).all(), f"action input has NaN/Inf. Shape: {action.shape}, min: {action.min()}, max: {action.max()}"
            action_emb = dit.action_mlp2(action)

    if dit.seperated_timestep and fuse_vae_embedding_in_latents and dit.one_frame_condition:
        raise NotImplementedError("已弃用，请迁移至[five_frame_condition]")
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))

        if dit.TI2V2 or dit.TI2V3:
            action_emb = action_emb.unsqueeze(0) # [1, T, 3072]
            action_emb = action_emb.unsqueeze(2).repeat(1,1,64,1).flatten(1,2)
            t = t + action_emb

        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
        # print(f't_mod shape: {t_mod.shape}')
    elif dit.seperated_timestep and fuse_vae_embedding_in_latents and dit.five_frame_condition:
        if not bs_1:
            timestep = torch.concat([
                torch.zeros((2, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
                torch.ones((latents.shape[2] - 2, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
            ]).flatten()
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0)).repeat(B, 1, 1)
            if dit.TI2V2 or dit.TI2V3:
                action_emb = action_emb.unsqueeze(2).repeat(1,1,300,1).flatten(1,2)
                t = t + action_emb
            t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
        elif bs_1:
            timestep = torch.concat([
                torch.zeros((2, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
                torch.ones((latents.shape[2] - 2, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
            ]).flatten()
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))

            if dit.TI2V2 or dit.TI2V3:
                action_emb = action_emb.unsqueeze(0) # [1, T, 3072]
                action_emb = action_emb.unsqueeze(2).repeat(1,1,300,1).flatten(1,2)
                t = t + action_emb
            print(f'the range of t and action_emb before projection, t: min {t.min().item()}, max {t.max().item()}, action_emb: min {action_emb.min().item()}, max {action_emb.max().item()}')
            t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if analysis_collector is not None:
        analysis_collector.update("t_mod", t_mod)

    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    x_input_for_residual = x
    if analysis_collector is not None:
        x_input_for_residual = x.clone()

    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    if dit.one_frame_condition:
        raise NotImplementedError("已弃用，请迁移至[five_frame_condition]")
        freqs = torch.cat([
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    elif dit.five_frame_condition:
        freqs = torch.cat([
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    assert torch.isfinite(x).all(), "x before transformer blocks has NaN/Inf"
    assert torch.isfinite(context).all(), "context before transformer blocks has NaN/Inf"
    assert torch.isfinite(t_mod).all(), "t_mod before transformer blocks has NaN/Inf"



    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        

    # blocks

    if tea_cache_update:
        print('use tea cache to skip computation')
        x = tea_cache.update(x)
    else:
        # print('normal forward')
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        def create_custom_forward_vap(block, vap):
            def custom_forward(*inputs):
                return vap(block, *inputs)
            return custom_forward
        for block_id, block in enumerate(dit.blocks):
            t_layer_start = time.time()
            # Block
            if use_gradient_checkpointing_offload:
                # print(f'=== gradient checkpointing offload block forward, block id: {block_id} ===')
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                # print(f'=== gradient checkpointing block forward, block id: {block_id} ===')
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                # print(f'normal block forward, block id: {block_id}')
                if dit.TI2V2 or dit.TI2V1 or dit.TI2V3:
                    x = block(x, context, t_mod, freqs)
                elif dit.I2V:
                    raise NotImplementedError("已弃用，请迁移至[TI2V]")
            
            

            t_layer_end = time.time()
            if (t_layer_end - t_layer_start) > 0.5: 
                print(f"  [ModelFn] Block {block_id} slow: {t_layer_end - t_layer_start:.4f}s")
        if tea_cache is not None:
            tea_cache.store(x)
    if analysis_collector is not None:
        print(f'x_input_for_residual shape: {x_input_for_residual.shape}, x shape: {x.shape}')
        model_residual = x - x_input_for_residual
        analysis_collector.update("model_residual", model_residual)
        analysis_collector.update("hidden_states_before_head", x)
    
    x = dit.head(x, t)

    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    
    
    x = dit.unpatchify(x, (f, h, w))
    return x



