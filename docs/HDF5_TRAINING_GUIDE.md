# HDF5 Dataset Training for MotoGPT with Normalization

本指南说明如何使用 HDF5 数据集训练 MotoGPT，预测归一化的 qpos chunk。

## 修改的文件

### 1. 核心修改
- **`common/data/hdf5_datasets.py`**: 
  - 添加了 `norm_stats` 和 `use_robot_base` 参数
  - 修改 `_read_qpos_f` 方法支持 qpos 归一化

### 2. 新增文件
- **`common/data/norm_utils.py`**: 归一化工具函数
  - `compute_norm_stats_from_hdf5()`: 计算归一化统计
  - `save_norm_stats()` / `load_norm_stats()`: 保存/加载统计
  - `normalize_actions()` / `denormalize_actions()`: 归一化/反归一化

- **`configs/hdf5_aloha_config.yaml`**: 训练配置文件

- **`scripts/train_hdf5_aloha.py`**: 训练脚本

- **`scripts/compute_norm_stats.py`**: 计算归一化统计的独立脚本

### 3. Trainer 修改
- **`moto_gpt/src/trainers/moto_gpt_trainer_aloha.py`**:
  - `calculate_loss` 方法支持多视角数据键名（`rgb_initial_static` 或 `rgb_initial`）

## 使用步骤

### 步骤 1: 准备 HDF5 数据集

确保你的数据集结构如下：
```
/path/to/hdf5/data/
├── train/
│   ├── task1/
│   │   ├── episode_0.hdf5
│   │   ├── episode_1.hdf5
│   │   ├── ...
│   │   └── instr.txt (可选，包含语言指令)
│   └── task2/
│       └── ...
└── val/
    └── ...
```

每个 HDF5 文件应包含：
- `observations/images/cam_high`: 主相机图像 (H, W, C)
- `observations/images/cam_right_wrist`: 夹爪相机图像
- `observations/images/cam_left_wrist`: 左侧相机图像
- `observations/qpos`: 关节位置，最后 7 维为 action (T, qpos_dim)

### 步骤 2: 计算归一化统计

```bash
cd /home/hlwang/Moto

# 方法 1: 使用独立脚本
python scripts/compute_norm_stats.py \
    --hdf5_dir /path/to/your/hdf5/data \
    --output ./norm_stats/aloha_norm_stats.pt \
    --qpos_key observations/qpos \
    --num_episodes 100  # 可选，使用前100个episodes

# 方法 2: 训练时自动计算（首次运行）
# 在 config 中设置 compute_norm_stats: true
```

### 步骤 3: 修改配置文件

编辑 `configs/hdf5_aloha_config.yaml`:

```yaml
dataset:
  hdf5_dir: "/path/to/your/hdf5/data"  # 修改为实际路径
  use_normalization: true
  norm_stats_path: "./norm_stats/aloha_norm_stats.pt"
  compute_norm_stats: false  # 如果已经计算过
  
  # 根据你的数据集调整
  camera_key: "observations/images/cam_high"
  qpos_key: "observations/qpos"
  
training:
  batch_size_train: 32
  num_epochs: 100
  
paths:
  save_path: "./checkpoints/aloha_hdf5"
```

### 步骤 4: 运行训练

```bash
python scripts/train_hdf5_aloha.py --config configs/hdf5_aloha_config.yaml
```

## 数据流说明

### 输入数据格式（HDF5Dataset 输出）

```python
{
    "lang": str,                                    # 语言指令
    "rgb_initial_static": (1, 3, 224, 224),        # 初始帧 (static cam)
    "rgb_future_static": (T, 3, 224, 224),         # 未来帧 (static cam)
    "rgb_initial_gripper": (1, 3, 224, 224),       # 初始帧 (gripper cam)
    "rgb_future_gripper": (T, 3, 224, 224),        # 未来帧 (gripper cam)
    "rgb_initial_left": (1, 3, 224, 224),          # 初始帧 (left cam)
    "rgb_future_left": (T, 3, 224, 224),           # 未来帧 (left cam)
    "actions": (T, chunk_size, 7),                 # 归一化的 qpos actions
    "mask": (T, chunk_size),                       # action mask
    "latent_mask": (T,),                           # latent motion mask
    "idx": int,
    "delta_t": int,
    "start_local_step": int,
}
```

### 归一化公式

**训练时（Dataset）**:
```python
normalized_action = (action - qpos_mean) / qpos_std
```

**推理时（需要反归一化）**:
```python
from common.data.norm_utils import denormalize_actions

# 预测的归一化 actions
predicted_actions = model(...)  # shape: (B, T, 7)

# 反归一化到原始尺度
denorm_actions = denormalize_actions(predicted_actions, norm_stats)
```

## 多视角支持

Trainer 的 `calculate_loss` 方法会自动检测数据键名：
- 如果存在 `rgb_initial_static`，使用多视角数据（来自 HDF5Dataset）
- 否则使用 `rgb_initial`（来自 LMDB Dataset）

当前默认使用 `static` 相机，如需使用其他相机或融合多相机，需要修改 trainer。

## 注意事项

### 1. Action 维度
- 当前假设 action 维度为 7（前6维 arm，最后1维 gripper）
- 如果你的数据维度不同，需要修改：
  - `act_dim` 参数
  - `calculate_loss` 中的维度分割

### 2. Gripper Action
- 当前假设 gripper 是二值的（开/关）
- 使用 `binary_cross_entropy_with_logits` loss
- 如果是连续值，设置 `pred_binary_gripper_action: false`

### 3. 内存使用
- 多视角数据会占用更多内存
- 如果 OOM，可以：
  - 减小 `batch_size`
  - 减小 `sequence_length`
  - 只使用单个相机视角

### 4. Denormalization 推理
在推理脚本中需要加载 norm_stats 并反归一化：

```python
from common.data.norm_utils import load_norm_stats, denormalize_actions

# 加载 norm_stats
norm_stats = load_norm_stats("./norm_stats/aloha_norm_stats.pt")

# 推理
with torch.no_grad():
    pred = model(...)
    arm_actions_norm = pred['arm_action_preds']  # (B, T, 6)
    gripper_actions = pred['gripper_action_preds']  # (B, T, 1)
    
    # 反归一化 arm actions
    arm_actions = denormalize_actions(
        arm_actions_norm, 
        {"qpos_mean": norm_stats["qpos_mean"][:6],
         "qpos_std": norm_stats["qpos_std"][:6]}
    )
    
    # gripper 通常不需要反归一化（二值）
    gripper_actions = torch.sigmoid(gripper_actions)  # 如果用 BCE loss
```

## 调试

### 测试数据加载

```python
from common.data.hdf5_datasets import HDF5Dataset_for_MotoGPT_CALVINLike
from common.data.norm_utils import load_norm_stats

# 加载 norm_stats
norm_stats = load_norm_stats("./norm_stats/aloha_norm_stats.pt")

# 创建数据集
dataset = HDF5Dataset_for_MotoGPT_CALVINLike(
    hdf5_dir="/path/to/data",
    split="train",
    skip_frame=5,
    sequence_length=1,
    chunk_size=3,
    norm_stats=norm_stats,
    # ... 其他参数
)

# 测试采样
sample = dataset[0]
print(f"Actions shape: {sample['actions'].shape}")
print(f"Actions range: [{sample['actions'].min():.3f}, {sample['actions'].max():.3f}]")
print(f"Expected: roughly [-3, 3] if normalized correctly")
```

### 验证归一化

```python
# Actions 应该接近 N(0, 1) 分布
import matplotlib.pyplot as plt

actions = []
for i in range(min(100, len(dataset))):
    actions.append(dataset[i]['actions'])
actions = torch.cat([a.flatten() for a in actions])

plt.hist(actions.numpy(), bins=50)
plt.title("Normalized Actions Distribution")
plt.savefig("actions_dist.png")
print(f"Mean: {actions.mean():.3f}, Std: {actions.std():.3f}")
# 应该接近 Mean: 0.0, Std: 1.0
```

## 后续工作

1. **完善训练脚本**: 在 `train_hdf5_aloha.py` 中添加模型加载和 trainer 初始化
2. **推理脚本**: 创建包含 denormalization 的推理脚本
3. **多相机融合**: 如需使用多个相机视角，修改 trainer 的图像处理部分
4. **可视化**: 添加预测 qpos 的可视化工具

## 故障排除

### 问题: "No episode files found"
- 检查 `hdf5_dir` 路径是否正确
- 确保目录包含 `train/` 和 `val/` 子目录
- 确保文件名匹配 `episode_\d+\.hdf5` 格式

### 问题: "KeyError: 'rgb_initial_static'"
- 旧版 LMDB dataset 使用 `rgb_initial`
- 新版 HDF5 dataset 使用 `rgb_initial_static`
- Trainer 已修改为自动兼容两者

### 问题: Loss 很大或 NaN
- 检查归一化是否正确应用
- 验证 actions 的分布
- 检查学习率是否合适
