import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
class MyNpyDatasetnew(Dataset):
    """
    dataset_base_path/  
        rgb.npy  : [T, N, 3, H, W]
        traj.npy : [T, N, 3, H, W] 或 [T, N, ...]，需保证每帧能转成图片
    """

    def __init__(self, base_path='/opt/zsq/rlinf_dataset_1114_split/train_data', num_frames=9, repeat=1):
        self.base_path = base_path
        self.num_frames = num_frames
        self.repeat = repeat
        self.load_from_cache = False

        self.data_paths = []
        for step_name in sorted(os.listdir(base_path)):
            # step_path = os.path.join(base_path, step_name, "video/eval")
            step_path = os.path.join(base_path, step_name)
            if not os.path.isdir(step_path):
                continue
            for seed_name in sorted(os.listdir(step_path)):
                seed_path = os.path.join(step_path, seed_name)
                if os.path.isdir(seed_path):
                    self.data_paths.append(seed_path)
        if len(self.data_paths) == 0:
            raise ValueError(f"No valid data paths found under {base_path}")
        print(f'Found {len(self.data_paths)} data paths under {base_path}.')

        self.T_list = []
        self.N_list = []

        for p in self.data_paths:
            rgb_shape = np.load(os.path.join(p, "rgb.npy"), mmap_mode='r').shape
            T, N = rgb_shape[0], rgb_shape[1]
            self.T_list.append(T)
            self.N_list.append(N)

        self.total_env = sum(self.N_list)
        self.length = self.total_env * repeat 

        self.cum_N = np.cumsum(self.N_list)


    def __len__(self):
        return self.length
    def _locate_env(self, global_env_id):
        path_idx = np.searchsorted(self.cum_N, global_env_id, side='right')
        if path_idx == 0:
            env_id = global_env_id
        else:
            env_id = global_env_id - self.cum_N[path_idx - 1]
        return path_idx, env_id
    def __getitem__(self, idx):
        global_env_id = idx % self.total_env

        path_idx, env_id = self._locate_env(global_env_id)
        data_path = self.data_paths[path_idx]
        T = self.T_list[path_idx]
        rgb = np.load(os.path.join(data_path, "rgb.npy"), mmap_mode='r')
        actions = np.load(os.path.join(data_path, "actions.npy"), mmap_mode='r')


        if T > (self.num_frames-1):
            if np.random.rand() < 0.95:
                # 95% 随机采样
                if T >256:
                    start_idx = np.random.randint(0, 250)
                else:
                    start_idx = np.random.randint(0, T - self.num_frames + 2)

                consecutive_ids = np.arange(start_idx, start_idx + self.num_frames - 1)
                frame_ids = np.concatenate([[0], consecutive_ids])
            else:
                # 5% 使用硬编码 frame_ids
                frame_ids = np.array([0,0,0,0,0,1,2,3,4,5,6,7,8])
        else:
            raise ValueError(f"T={T} is too small for num_frames={self.num_frames}")
        

        video_np = rgb[frame_ids, env_id]  # shape [num_frames, 3, H, W]
        video_list = []
        for frame in video_np:
            if "/opt/zsq/rlinf_dataset_1114_split" in data_path:
                img = np.transpose(frame, (1, 2, 0))  # CHW → HWC
            else:
                img = frame
            if img.max() <= 1.0:
                img = (img * 255).clip(0, 255)
            video_list.append(Image.fromarray(img.astype(np.uint8)))

        # Action → Tensor
        action_np = actions[frame_ids, env_id]  # [num_frames, action_dim]
        action_tensor = torch.from_numpy(action_np).float()
        action_tensor[0] = torch.tensor([0., 0., 0., 0., 0., 0., -1.],
                                  dtype=action_tensor.dtype,
                                  device=action_tensor.device)
        return {
            "video": video_list,
            "prompt": "机械臂根据控制轨迹进行相应的移动",
            "reference_image": [video_list[0]],
            "action": action_tensor,
        }