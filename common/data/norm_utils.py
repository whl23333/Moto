import os
import h5py
import numpy as np
import torch
import re
from glob import glob


def compute_norm_stats_from_hdf5(hdf5_dir, qpos_key="observations/qpos", use_robot_base=False, num_episodes=None):
    """
    从 HDF5 数据集计算 qpos 的归一化统计信息
    
    Args:
        hdf5_dir: HDF5 数据目录（可能包含 train/val 子目录）
        qpos_key: qpos 在 HDF5 中的 key
        use_robot_base: 是否包含 base_action
        num_episodes: 使用多少个 episode 计算统计（None 表示全部）
    
    Returns:
        dict: {"qpos_mean": torch.Tensor, "qpos_std": torch.Tensor, 
               "action_mean": torch.Tensor, "action_std": torch.Tensor}
    """
    # 收集所有 episode 文件
    train_dir = os.path.join(hdf5_dir, "train")
    if os.path.isdir(train_dir):
        search_dir = train_dir
    else:
        search_dir = hdf5_dir
    
    episode_files = []
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if re.match(r"episode_\d+\.hdf5$", f):
                episode_files.append(os.path.join(root, f))
    
    episode_files.sort()
    
    if num_episodes is not None:
        episode_files = episode_files[:num_episodes]
    
    if len(episode_files) == 0:
        raise ValueError(f"No episode files found in {search_dir}")
    
    print(f"Computing normalization stats from {len(episode_files)} episodes...")
    
    all_qpos_data = []
    
    for ep_file in episode_files:
        with h5py.File(ep_file, "r") as f:
            qpos = f[qpos_key][()]  # (T, qpos_dim)
            if use_robot_base and '/base_action' in f:
                base_action = f['/base_action'][()]
                qpos = np.concatenate((qpos, base_action), axis=1)
            all_qpos_data.append(torch.from_numpy(qpos))
    
    # all_qpos_data = torch.stack(all_qpos_data)  # (num_episodes, T, qpos_dim)
    all_qpos_data = torch.cat(all_qpos_data, dim=0)  # (total_timesteps, qpos_dim)
    
    # 计算统计信息（最后 7 维作为 action）
    # qpos_mean = all_qpos_data[..., -7:].mean(dim=[0, 1], keepdim=False).float()
    # qpos_std = all_qpos_data[..., -7:].std(dim=[0, 1], keepdim=False).float()
    qpos_mean = all_qpos_data[..., -7:].mean(dim=0, keepdim=False).float()
    qpos_std = all_qpos_data[..., -7:].std(dim=0, keepdim=False).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)
    
    stats = {
        "qpos_mean": qpos_mean,  # (7,)
        "qpos_std": qpos_std,    # (7,)
        "action_mean": qpos_mean,
        "action_std": qpos_std,
    }
    
    print(f"Normalization stats computed:")
    print(f"  qpos_mean: {qpos_mean}")
    print(f"  qpos_std: {qpos_std}")
    
    return stats


def save_norm_stats(stats, save_path):
    """保存归一化统计信息"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(stats, save_path)
    print(f"Normalization stats saved to {save_path}")


def load_norm_stats(load_path):
    """加载归一化统计信息"""
    stats = torch.load(load_path, map_location='cpu')
    print(f"Normalization stats loaded from {load_path}")
    return stats


def denormalize_actions(actions, norm_stats):
    """
    将归一化的 actions 恢复到原始尺度
    
    Args:
        actions: torch.Tensor, shape (..., 7)
        norm_stats: dict with "qpos_mean" and "qpos_std"
    
    Returns:
        torch.Tensor: denormalized actions
    """
    qpos_mean = norm_stats["qpos_mean"]
    qpos_std = norm_stats["qpos_std"]
    
    # Ensure tensors are on the same device
    if isinstance(qpos_mean, torch.Tensor):
        qpos_mean = qpos_mean.to(actions.device)
    if isinstance(qpos_std, torch.Tensor):
        qpos_std = qpos_std.to(actions.device)
    
    return actions * qpos_std + qpos_mean


def normalize_actions(actions, norm_stats):
    """
    将 actions 归一化
    
    Args:
        actions: torch.Tensor, shape (..., 7)
        norm_stats: dict with "qpos_mean" and "qpos_std"
    
    Returns:
        torch.Tensor: normalized actions
    """
    qpos_mean = norm_stats["qpos_mean"]
    qpos_std = norm_stats["qpos_std"]
    
    # Ensure tensors are on the same device
    if isinstance(qpos_mean, torch.Tensor):
        qpos_mean = qpos_mean.to(actions.device)
    if isinstance(qpos_std, torch.Tensor):
        qpos_std = qpos_std.to(actions.device)
    
    return (actions - qpos_mean) / qpos_std
