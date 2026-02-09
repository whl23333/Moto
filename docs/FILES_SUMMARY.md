# HDF5 ALOHA 训练配置 - 文件清单

## 按照原始训练流程创建的配置结构

本次配置完全遵循原始 CALVIN 训练的结构，分为独立的配置文件而非单一配置。

---

## 📁 新创建的文件

### 1. 配置文件

#### `/home/hlwang/Moto/moto_gpt/configs/data/hdf5_aloha.yaml`
**作用**: 数据集配置（对应 CALVIN 的 `calvin.yaml`）
**内容**:
- `data_type: "hdf5_aloha"`
- HDF5 数据路径和相机配置
- Normalization 配置
- 数据过滤配置

#### `/home/hlwang/Moto/moto_gpt/configs/models/actPredTrue_motionPredTrue_visionMaeLarge_seq1_chunk3_maskProb0.5_aloha.yaml`
**作用**: 模型配置（对应 CALVIN 的模型配置）
**内容**:
- MotoGPT 模型架构
- Vision encoder (MAE-Large)
- Language encoder (T5-base)
- GPT2 causal transformer
- `sequence_length: 1`, `chunk_size: 3`, `act_dim: 7`

#### `/home/hlwang/Moto/moto_gpt/configs/train/hdf5_aloha_train.yaml`
**作用**: 训练主配置（对应 CALVIN 的训练配置）
**内容**:
- 引用上述 model config 和 data config
- Dataloader 配置
- RGB preprocessor 配置
- Training 超参数

---

### 2. 训练脚本

#### `/home/hlwang/Moto/moto_gpt/train/train_moto_gpt_hdf5_aloha.py`
**作用**: 主训练脚本（对应 `train_moto_gpt.py`）
**特点**:
- 完全遵循原始 `train_moto_gpt.py` 的逻辑
- 使用 hydra 加载模型配置
- 使用 `load_dataset` 加载数据
- 自动处理 normalization

#### `/home/hlwang/Moto/moto_gpt/train/compute_hdf5_norm_stats.py`
**作用**: 计算归一化统计信息
**用法**: 在首次训练前运行一次

---

### 3. 辅助脚本

#### `/home/hlwang/Moto/scripts/train_hdf5_aloha_quickstart.sh`
**作用**: 一键启动训练的 shell 脚本
**功能**:
- 自动检查 normalization stats
- 如果不存在则自动计算
- 启动训练

---

### 4. 文档

#### `/home/hlwang/Moto/docs/HDF5_ALOHA_TRAINING_COMPARISON.md`
**作用**: 详细的对比文档
**内容**:
- 与原始 CALVIN 训练流程的详细对比
- 配置文件结构说明
- 使用流程和注意事项
- 快速开始指南

---

## 🔧 修改的文件

### `/home/hlwang/Moto/common/data/data_utils.py`
**修改内容**:
1. 导入 `HDF5Dataset_for_MotoGPT_CALVINLike`
2. 导入 normalization 工具函数
3. 在 `data_type2dataset_cls` 中注册 `'hdf5_aloha'`
4. 在 `load_dataset` 函数中添加 HDF5 数据集的 normalization 处理

**修改位置**:
- Line 11-12: 添加 import
- Line 21: 添加数据类型映射
- Line 30-50: 添加 normalization 处理逻辑

---

## ✅ 之前已完成的修改（保持不变）

### `/home/hlwang/Moto/common/data/hdf5_datasets.py`
- 添加 `norm_stats` 和 `use_robot_base` 参数
- 修改 `_read_qpos_f` 支持归一化

### `/home/hlwang/Moto/common/data/norm_utils.py`
- 创建完整的归一化工具函数库

### `/home/hlwang/Moto/moto_gpt/src/trainers/moto_gpt_trainer_aloha.py`
- `calculate_loss` 支持多视角数据键名

---

## 🔄 配置结构对比

### 原始 CALVIN 结构
```
configs/train/xxx.yaml
├── moto_gpt_config_path
├── dataset_config_path
├── latent_motion_tokenizer_path
├── dataloader_config
├── rgb_preprocessor_config
└── training_config

train_moto_gpt.py
├── Load configs
├── Instantiate model (hydra)
├── Load dataset (load_dataset)
├── Create dataloaders
└── Start training
```

### HDF5 ALOHA 结构（完全一致）
```
configs/train/hdf5_aloha_train.yaml
├── moto_gpt_config_path
├── dataset_config_path
├── latent_motion_tokenizer_path
├── dataloader_config
├── rgb_preprocessor_config
└── training_config

train_moto_gpt_hdf5_aloha.py
├── Load configs
├── Instantiate model (hydra)
├── Load dataset (load_dataset + norm)
├── Create dataloaders
└── Start training
```

**唯一区别**: 在 `load_dataset` 中添加了 normalization 处理

---

## 🎯 使用方法

### 方法 1: 使用快速启动脚本（推荐）
```bash
cd /home/hlwang/Moto
./scripts/train_hdf5_aloha_quickstart.sh
```

### 方法 2: 手动执行
```bash
cd /home/hlwang/Moto

# 步骤 1: 计算 normalization（如果还没有）
python moto_gpt/train/compute_hdf5_norm_stats.py

# 步骤 2: 开始训练
python moto_gpt/train/train_moto_gpt_hdf5_aloha.py
```

### 方法 3: 自定义配置
```bash
# 修改配置文件
vim moto_gpt/configs/train/hdf5_aloha_train.yaml

# 使用自定义配置训练
python moto_gpt/train/train_moto_gpt_hdf5_aloha.py \
    --config_path moto_gpt/configs/train/your_custom_config.yaml
```

---

## 📊 与原始流程的一致性

| 特性 | 原始 CALVIN | HDF5 ALOHA | 状态 |
|------|------------|------------|------|
| 配置文件结构 | 3个独立文件 | 3个独立文件 | ✅ 一致 |
| 训练脚本逻辑 | train_moto_gpt.py | train_moto_gpt_hdf5_aloha.py | ✅ 一致 |
| 模型加载方式 | hydra.utils.instantiate | hydra.utils.instantiate | ✅ 一致 |
| 数据加载方式 | load_dataset() | load_dataset() + norm | ✅ 一致（仅添加norm） |
| Dataloader 设置 | partial(DataLoader, ...) | partial(DataLoader, ...) | ✅ 一致 |
| Trainer 调用 | MotoGPT_Trainer(...) | MotoGPT_Trainer(...) | ✅ 一致 |
| RGB preprocessor | get_rgb_preprocessor() | get_rgb_preprocessor() | ✅ 一致 |

**结论**: 除了数据来源和必要的 normalization 处理外，完全遵循原始训练流程。

---

## 🆕 与之前单一配置文件的区别

### 之前（scripts/train_hdf5_aloha.py + configs/hdf5_aloha_config.yaml）
- ❌ 所有配置写在一个 YAML 文件中
- ❌ 训练脚本自己处理配置
- ❌ 与原始流程结构不一致

### 现在（train_moto_gpt_hdf5_aloha.py + 3个配置文件）
- ✅ 配置分为 data/model/train 三部分
- ✅ 训练脚本遵循原始逻辑
- ✅ 完全符合原始流程结构
- ✅ 易于维护和扩展

---

## 📝 配置参数说明

### 关键参数（与 CALVIN 不同的部分）

#### Model Config
```yaml
sequence_length: 1     # CALVIN: 2 (根据任务调整)
chunk_size: 3          # CALVIN: 5 (根据任务调整)
act_pred: true         # CALVIN: false (ALOHA 需要预测动作)
```

#### Data Config
```yaml
data_type: "hdf5_aloha"              # CALVIN: "calvin"
use_normalization: true              # 新增（必需）
norm_stats_path: "..."               # 新增（必需）
camera_key: "observations/images/..."  # 新增（HDF5 key）
```

#### Train Config
```yaml
bs_per_gpu: 32                # CALVIN: 128 (根据 GPU 内存调整)
num_epochs: 100               # CALVIN: 20 (根据数据量调整)
paired_loss: false            # CALVIN: true (根据需求调整)
```

---

## 🔍 验证配置正确性

运行以下命令验证配置是否正确：

```bash
# 1. 检查 normalization stats
python -c "
import torch
stats = torch.load('/home/hlwang/Moto/norm_stats/aloha_norm_stats.pt')
print('qpos_mean:', stats['qpos_mean'])
print('qpos_std:', stats['qpos_std'])
"

# 2. 测试数据加载
python -c "
import omegaconf
from common.data.data_utils import load_dataset

extra_config = {
    'sequence_length': 1,
    'chunk_size': 3,
    'act_dim': 7,
    'do_extract_future_frames': True,
    'do_extract_action': True
}

train_ds, eval_ds = load_dataset(
    '/home/hlwang/Moto/moto_gpt/configs/data/hdf5_aloha.yaml',
    extra_config
)
print(f'Train size: {len(train_ds)}')
print(f'Eval size: {len(eval_ds)}')
sample = train_ds[0]
print(f'Sample keys: {sample.keys()}')
print(f'Actions shape: {sample[\"actions\"].shape}')
"

# 3. 测试模型加载
python -c "
import omegaconf
import hydra

config = omegaconf.OmegaConf.load(
    '/home/hlwang/Moto/moto_gpt/configs/models/actPredTrue_motionPredTrue_visionMaeLarge_seq1_chunk3_maskProb0.5_aloha.yaml'
)
model = hydra.utils.instantiate(config)
print('Model loaded successfully!')
print(f'Model type: {type(model).__name__}')
"
```

---

## ❓ 常见问题

### Q1: 与之前创建的 `scripts/train_hdf5_aloha.py` 有什么区别？
A: 之前的脚本把所有配置放在一个文件中，与原始流程不一致。现在的脚本完全遵循原始的三文件配置结构。

### Q2: 我应该使用哪个训练脚本？
A: 使用 `moto_gpt/train/train_moto_gpt_hdf5_aloha.py`，这个与原始流程一致。

### Q3: 旧的配置文件还能用吗？
A: 旧的单一配置文件结构不同，建议使用新的三文件结构。

### Q4: 如何添加更多数据增强？
A: 修改 `hdf5_aloha_train.yaml` 中的 `rgb_preprocessor_config.vision_aug_config`

### Q5: 如何使用预训练的 latent motion tokenizer？
A: 在 `hdf5_aloha_train.yaml` 中设置 `latent_motion_tokenizer_path`

---

## 📁 完整文件树

```
Moto/
├── moto_gpt/
│   ├── configs/
│   │   ├── data/
│   │   │   ├── calvin.yaml
│   │   │   └── hdf5_aloha.yaml                    [新建]
│   │   ├── models/
│   │   │   ├── actPredFalse_motionPredTrue_...yaml
│   │   │   └── actPredTrue_motionPredTrue_...aloha.yaml  [新建]
│   │   └── train/
│   │       ├── data_calvin-model_...yaml
│   │       └── hdf5_aloha_train.yaml              [新建]
│   ├── train/
│   │   ├── train_moto_gpt.py
│   │   ├── train_moto_gpt_hdf5_aloha.py           [新建]
│   │   └── compute_hdf5_norm_stats.py             [新建]
│   └── src/trainers/
│       └── moto_gpt_trainer_aloha.py              [已修改]
├── common/
│   └── data/
│       ├── data_utils.py                          [已修改]
│       ├── hdf5_datasets.py                       [已修改]
│       └── norm_utils.py                          [已创建]
├── scripts/
│   ├── compute_norm_stats.py                      [已创建]
│   └── train_hdf5_aloha_quickstart.sh             [新建]
├── docs/
│   ├── HDF5_TRAINING_GUIDE.md                     [已创建]
│   ├── HDF5_ALOHA_TRAINING_COMPARISON.md          [新建]
│   └── FILES_SUMMARY.md                           [本文件]
└── norm_stats/
    └── aloha_norm_stats.pt                        [运行时生成]
```

---

## 🎉 总结

✅ **完全遵循原始训练流程结构**
✅ **配置文件清晰分离（data/model/train）**
✅ **训练脚本逻辑一致**
✅ **添加了必要的 normalization 支持**
✅ **保持了代码的可维护性和可扩展性**

现在可以开始训练了！🚀
