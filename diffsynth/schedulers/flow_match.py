import torch, math



class FlowMatchScheduler():

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003/1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None, dynamic_shift_len=None, exponential_shift_mu=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            if exponential_shift_mu is not None:
                mu = exponential_shift_mu
            elif dynamic_shift_len is not None:
                mu = self.calculate_shift(dynamic_shift_len)
            else:
                mu = self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False


    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    
    # def add_noise(self, original_samples, noise, timestep):
    #     if isinstance(timestep, torch.Tensor):
    #         timestep = timestep.cpu()
    #     timestep_id = torch.argmin((self.timesteps - timestep).abs())
    #     sigma = self.sigmas[timestep_id]
    #     sample = (1 - sigma) * original_samples + sigma * noise
    #     return sample
    
    def add_noise(self, original_samples, noise, timestep):
        # timestep: shape [T]
        # original_samples: [B, C, T, H, W]
        if timestep.dim() == 1:
            # fallback: original behaviour
            timestep_id = torch.argmin((self.timesteps - timestep.cpu()).abs())
            sigma = self.sigmas[timestep_id]
            return (1 - sigma) * original_samples + sigma * noise

        # --- new behaviour: per-frame timestep ---
        # timestep shape: [T]
        timestep = timestep.cpu()

        # 找到每一帧对应的 timestep_id
        # timesteps: [num_steps]
        # timestep: [T]
        abs_diff = (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs()
        timestep_ids = torch.argmin(abs_diff, dim=0)   # shape [T]

        # sigma_per_frame: shape [T]
        sigma_per_frame = self.sigmas[timestep_ids].to(original_samples.device)

        # reshape 为 [1,1,T,1,1] 以便广播
        sigma = sigma_per_frame.view(1, 1, -1, 1, 1)

        # broadcast: [B,C,T,H,W]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample


    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
    

    def training_weight(self, timestep):
        if timestep.dim() == 1:
            timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
            weights = self.linear_timesteps_weights[timestep_id]
            return weights
            # ---- Case 2：向量 timestep（per-frame），timestep shape = [T] ----
        # 扩展维度以便比较： self.timesteps: [S], timestep: [T]
        # 计算 |timesteps - timestep| → shape [S, T]
        abs_diff = (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs()

        # 求每帧最近的 timestep index → shape [T]
        timestep_ids = torch.argmin(abs_diff, dim=0)

        # 取出对应的 weight → shape [T]
        weights = self.linear_timesteps_weights[timestep_ids].to(timestep.device)

        # 变成可广播到 [B, C, T, H, W] 的形状
        # [T] → [1,1,T,1,1]
        weights = weights.view(1, 1, -1, 1, 1)

        return weights
    
    
    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
