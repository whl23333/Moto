# HDF5 ALOHA 训练配置说明

## 配置文件结构对比

### 原始 CALVIN 训练结构
```
moto_gpt/configs/train/data_calvin-model_actPredFalse_motionPredTrue_visionMaeLarge_seq2_chunk5_maskProb0.5-train_lr0.0001_bs512-aug_shiftTrue_resizedCropFalse.yaml
├── moto_gpt_config_path → moto_gpt/configs/models/actPredFalse_motionPredTrue_visionMaeLarge_seq2_chunk5_maskProb0.5.yaml
├── dataset_config_path → moto_gpt/configs/data/calvin.yaml
├── latent_motion_tokenizer_path (预训练的 latent motion tokenizer)
├── dataloader_config
├── rgb_preprocessor_config
└── training_config
```

### HDF5 ALOHA 训练结构
```
moto_gpt/configs/train/hdf5_aloha_train.yaml
├── moto_gpt_config_path → moto_gpt/configs/models/actPredTrue_motionPredTrue_visionMaeLarge_seq1_chunk3_maskProb0.5_aloha.yaml
├── dataset_config_path → moto_gpt/configs/data/hdf5_aloha.yaml
├── latent_motion_tokenizer_path (可选，如果需要 latent motion prediction)
├── dataloader_config
├── rgb_preprocessor_config
└── training_config
```

## 关键差异

### 1. 数据集配置 (moto_gpt/configs/data/hdf5_aloha.yaml)
**新增内容：**
- `data_type: "hdf5_aloha"` (新的数据类型)
- `use_normalization: true` (启用归一化)
- `norm_stats_path` (归一化统计信息路径)
- `camera_key`, `camera_gripper_key`, `camera_left_key` (多相机支持)
- `qpos_key` (qpos 数据的 HDF5 key)
- `no_repeat_action`, `constant_action_atol` (过滤重复动作)

**与 CALVIN 相同：**
- `rgb_shape: [224, 224]`
- `skip_frame: 5`

### 2. 模型配置 (moto_gpt/configs/models/actPredTrue_motionPredTrue_visionMaeLarge_seq1_chunk3_maskProb0.5_aloha.yaml)
**与 CALVIN 的主要差异：**
- `sequence_length: 1` (CALVIN 用 2)
- `chunk_size: 3` (CALVIN 用 5)
- `act_pred: true` (CALVIN 用 false，因为只预测 latent motion)
- `latent_motion_pred: true` (两者都 true，但 ALOHA 也预测 action)

**相同部分：**
- 使用 MAE-Large 作为 vision encoder
- 使用 T5-base 作为 language encoder
- GPT2 作为 causal transformer
- `freeze_lang: true`, `freeze_vision: true`

### 3. 训练配置 (moto_gpt/configs/train/hdf5_aloha_train.yaml)
**主要差异：**
- `bs_per_gpu: 32` (CALVIN 用 128，因为 ALOHA 数据更复杂)
- `workers_per_gpu: 4` (CALVIN 用 7)
- `num_epochs: 100` (CALVIN 用 20)
- `num_warmup_epochs: 5` (CALVIN 用 1)
- `paired_loss: false` (CALVIN 用 true)

**相同部分：**
- `lr_max: 0.0001`
- `weight_decay: 0.0001`
- `gradient_accumulation_steps: 4`
- `pred_binary_gripper_action: true`
- RGB augmentation 设置

## 使用流程

### 步骤 1: 准备数据
确保你的 HDF5 数据在 `/media/disk3/WHL/aloha/` 下，结构为：
```
/media/disk3/WHL/aloha/
├── train/
│   ├── task1/
│   │   ├── episode_0.hdf5
│   │   └── ...
└── val/
    └── ...
```

### 步骤 2: 计算归一化统计（如果还没计算）
```bash
cd /home/hlwang/Moto

# 方法 1: 使用专用脚本（推荐）
python moto_gpt/train/compute_hdf5_norm_stats.py \
    --config_path moto_gpt/configs/data/hdf5_aloha.yaml

# 方法 2: 使用通用脚本
python scripts/compute_norm_stats.py \
    --hdf5_dir /media/disk3/WHL/aloha \
    --output /home/hlwang/Moto/norm_stats/aloha_norm_stats.pt \
    --qpos_key observations/qpos
```

### 步骤 3: 开始训练
```bash
cd /home/hlwang/Moto

# 使用默认配置
python moto_gpt/train/train_moto_gpt_hdf5_aloha.py

# 或指定配置文件
python moto_gpt/train/train_moto_gpt_hdf5_aloha.py \
    --config_path moto_gpt/configs/train/hdf5_aloha_train.yaml
```

## 配置修改建议

### 如果要使用 Latent Motion Prediction
1. 先训练或下载 latent motion tokenizer
2. 在 `hdf5_aloha_train.yaml` 中设置：
```yaml
latent_motion_tokenizer_path: "/path/to/latent_motion_tokenizer"
```

### 如果只想预测 Action（不预测 Latent Motion）
修改 `actPredTrue_motionPredTrue_visionMaeLarge_seq1_chunk3_maskProb0.5_aloha.yaml`：
```yaml
latent_motion_pred: false
act_pred: true
```

然后在 `hdf5_aloha_train.yaml` 中：
```yaml
latent_motion_tokenizer_path: null
```

### 调整 Batch Size 和 Workers
根据你的 GPU 内存和 CPU 核心数调整：
```yaml
dataloader_config:
  bs_per_gpu: 16  # 如果 OOM，减小这个值
  workers_per_gpu: 2  # 如果 CPU 负载高，减小这个值
```

### 修改 Sequence Length 和 Chunk Size
如果需要更长的预测序列：
```yaml
# 在 model config 中
sequence_length: 2  # 预测未来 2 帧
chunk_size: 5       # 每帧预测 5 个 action steps
```

## 代码修改说明

### 1. common/data/data_utils.py
**添加的功能：**
- 在 `data_type2dataset_cls` 中注册 `'hdf5_aloha'`
- 在 `load_dataset` 函数中添加 HDF5 数据集的 normalization 处理逻辑
- 自动加载或计算归一化统计信息

### 2. common/data/hdf5_datasets.py
**已有的修改（之前完成）：**
- `HDF5Dataset_for_MotoGPT_CALVINLike` 支持 normalization
- `_read_qpos_f` 方法应用归一化

### 3. moto_gpt/src/trainers/moto_gpt_trainer_aloha.py
**已有的修改（之前完成）：**
- `calculate_loss` 方法支持多视角数据键名
- 自动检测 `rgb_initial_static` 或 `rgb_initial`

## 数据流对比

### CALVIN (LMDB)
```
LMDB → LMDBDataset_for_MotoGPT_CALVIN → DataPrefetcher → Trainer
- rgb_initial: (B, 1, 3, 224, 224)
- rgb_future: (B, T, 3, 224, 224)
- actions: 直接从 LMDB 读取，无归一化
```

### ALOHA (HDF5)
```
HDF5 → HDF5Dataset_for_MotoGPT_CALVINLike → DataPrefetcher → Trainer
- rgb_initial_static: (B, 1, 3, 224, 224)
- rgb_future_static: (B, T, 3, 224, 224)
- rgb_initial_gripper: (B, 1, 3, 224, 224)  # 额外的相机
- rgb_future_gripper: (B, T, 3, 224, 224)
- actions: 从 qpos 提取 + 自动归一化
```

## 注意事项

### 1. Normalization 是必须的
HDF5 数据集中的 qpos 值范围差异很大，必须归一化才能训练。

### 2. 多相机数据
当前 trainer 默认只使用 `static` 相机。如果要使用多相机融合，需要修改 trainer。

### 3. 内存使用
HDF5 数据集会缓存打开的文件句柄，多相机数据会占用更多内存。如果遇到内存问题：
- 减小 `bs_per_gpu`
- 减小 `workers_per_gpu`
- 减小 `sequence_length`

### 4. 语言指令
HDF5 数据集支持从 `instr.txt` 读取语言指令，格式：
```
0 pick up the cube
1 place the cube on plate
...
```

### 5. 推理时需要反归一化
训练时 actions 是归一化的，推理时需要反归一化：
```python
from common.data.norm_utils import load_norm_stats, denormalize_actions

norm_stats = load_norm_stats("/home/hlwang/Moto/norm_stats/aloha_norm_stats.pt")
predicted_actions_norm = model(...)
predicted_actions = denormalize_actions(predicted_actions_norm, norm_stats)
```

## 完整文件清单

### 新创建的文件
```
moto_gpt/configs/data/hdf5_aloha.yaml                          # 数据集配置
moto_gpt/configs/models/actPredTrue_motionPredTrue_visionMaeLarge_seq1_chunk3_maskProb0.5_aloha.yaml  # 模型配置
moto_gpt/configs/train/hdf5_aloha_train.yaml                   # 训练配置
moto_gpt/train/train_moto_gpt_hdf5_aloha.py                    # 训练脚本
moto_gpt/train/compute_hdf5_norm_stats.py                      # 计算归一化统计
```

### 修改的文件
```
common/data/data_utils.py                                       # 添加 HDF5 支持
common/data/hdf5_datasets.py                                    # 之前已修改
common/data/norm_utils.py                                       # 之前已创建
moto_gpt/src/trainers/moto_gpt_trainer_aloha.py               # 之前已修改
```

## 与原始流程的一致性检查

✅ **相同的配置结构**: 分为 data config, model config, train config
✅ **相同的训练脚本逻辑**: 遵循 `train_moto_gpt.py` 的结构
✅ **相同的 dataloader 设置**: pin_memory, persistent_workers 等
✅ **相同的 trainer 调用**: 使用相同的参数传递方式
✅ **相同的 RGB preprocessor**: 使用 `get_rgb_preprocessor` 函数
✅ **相同的 language tokenizer**: AutoTokenizer from T5

🆕 **新增的内容**:
- Normalization 支持（HDF5 数据必需）
- 多相机支持
- 动作过滤（no_repeat_action）

## 快速开始

```bash
# 1. 计算归一化统计
python moto_gpt/train/compute_hdf5_norm_stats.py

# 2. 开始训练
python moto_gpt/train/train_moto_gpt_hdf5_aloha.py

# 3. 监控训练（另一个终端）
tensorboard --logdir /home/hlwang/Moto/moto_gpt/outputs/moto_gpt_aloha_hdf5/logs
```
