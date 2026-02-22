import h5py
from torch import tensor
import torch
file_path = "/media/disk3/WHL/aloha_single_traj/train/cup_plate_regular_pink_blue/episode_0.hdf5"
f = h5py.File(file_path, "r")
qpos = f["observations/qpos"][()]  # (T, qpos_dim)
print("qpos shape:", qpos.shape)
# print every 10 timesteps
for i in range(0, qpos.shape[0]):
    print(f"timestep {i}: {tensor(qpos[i][-7:])}")

norm_stats_path = "/home/hlwang/Moto/norm_stats/cup_plate_single_traj/norm_stats.pt"
norm_stats = torch.load(norm_stats_path)
print("Loaded normalization stats:", norm_stats)