# MotoGPT qpos输入功能实现总结

## 概述
成功为MotoGPT模型添加了可选的qpos（机器人当前关节位置）输入功能。这是一种常见的机器人学习技术，通过提供机器人当前的本体感受状态来增强策略学习效果。

## 实现方案

### 设计思路
将qpos作为额外的condition token添加到模型输入序列中，类似于language、patch和obs的处理方式。

**输入序列格式**：
- 原始：`[lang, patch, obs, [LATENT_1], [ACT_1], ..., [LATENT_t], [ACT_t]]`
- 添加qpos后：`[lang, patch, obs, qpos, [LATENT_1], [ACT_1], ..., [LATENT_t], [ACT_t]]`

### 修改文件

#### 1. `/home/hlwang/Moto/common/data/hdf5_datasets.py`

**修改内容**：
- 添加了`_read_full_qpos_f()`方法来读取完整的qpos（14维，而不仅是最后7维的action）
- 对ALOHA双臂机器人（14维qpos）进行了特殊处理，使用相同的归一化统计量分别归一化左右臂
- 在`__getitem__()`中读取初始时刻的qpos并添加到返回的字典中

**关键代码**：
```python
def _read_full_qpos_f(self, f, idx):
    """Read full qpos (not just last 7 dims) for proprioceptive state input"""
    qpos = f[self.qpos_key][idx]
    qpos_tensor = torch.from_numpy(np.asarray(qpos, dtype=np.float32))
    
    # Apply normalization if stats are provided
    if self.norm_stats is not None:
        qpos_mean = self.norm_stats["qpos_mean"]
        qpos_std = self.norm_stats["qpos_std"]
        # For ALOHA: 14-dim qpos = [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        if len(qpos_tensor) == 14:
            qpos_tensor[:7] = (qpos_tensor[:7] - qpos_mean) / qpos_std
            qpos_tensor[7:] = (qpos_tensor[7:] - qpos_mean) / qpos_std
        else:
            qpos_tensor[-7:] = (qpos_tensor[-7:] - qpos_mean) / qpos_std
    
    return qpos_tensor
```

返回的数据字典添加了`'qpos_initial'`键。

#### 2. `/home/hlwang/Moto/moto_gpt/src/models/moto_gpt.py`

**修改内容**：
- 在`__init__()`中添加了两个新参数：
  - `use_qpos_input=False`：是否使用qpos输入（默认关闭）
  - `qpos_dim=14`：qpos维度（ALOHA默认14维）
- 当`use_qpos_input=True`时，创建`embed_qpos` linear层用于qpos embedding
- 在`forward()`方法中：
  - 添加`qpos`参数（可选，默认None）
  - 嵌入qpos并添加condition embedding
  - 将qpos token添加到condition tokens序列中
  - 更新`n_cond_tokens`计算以包含qpos token

**关键代码**：
```python
# In __init__:
self.use_qpos_input = use_qpos_input
self.qpos_dim = qpos_dim
if self.use_qpos_input:
    self.embed_qpos = torch.nn.Linear(self.qpos_dim, hidden_size)

# In forward:
qpos_embeddings = None
if self.use_qpos_input and qpos is not None:
    qpos_embeddings = self.embed_qpos(qpos.float())  # (b, qpos_dim) -> (b, h)
    qpos_embeddings = qpos_embeddings.unsqueeze(1)  # (b, 1, h)
    qpos_embeddings = qpos_embeddings + condition_embeddings

cond_stacked_inputs = torch.cat((lang_embeddings, patch_embeddings, obs_embeddings, qpos_embeddings), dim=1)
n_qpos_tokens = 1 if (qpos_embeddings is not None and qpos_embeddings.shape[0] > 0) else 0
n_cond_tokens = n_lang_tokens + n_patch_tokens + n_obs_tokens + n_qpos_tokens
```

#### 3. `/home/hlwang/Moto/moto_gpt/src/trainers/moto_gpt_trainer_aloha.py`

**修改内容**：
- 在所有3个`calculate_loss()`方法中（对应不同的训练模式）添加了qpos提取和传递逻辑
- 从batch中提取`qpos_initial`并传递给moto_gpt的forward方法

**关键代码**：
```python
# Extract qpos if use_qpos_input is enabled
qpos = None
if self.moto_gpt_config.get('use_qpos_input', False):
    qpos = batch.get('qpos_initial', None)  # (b, qpos_dim)

pred = self.moto_gpt(
    rgb=rgb_initial,
    language=batch['lang_input_ids'],
    attention_mask=attention_mask,
    latent_motion_ids=latent_motion_ids,
    latent_mask=batch['latent_mask'],
    train=True,
    lang_attention_mask=batch['lang_attention_mask'],
    qpos=qpos,  # (b, qpos_dim) or None
)
```

## 使用方法

### 1. 在配置文件中启用qpos输入

在你的模型配置YAML文件中添加以下参数：

```yaml
# 启用qpos输入
use_qpos_input: true  # 设置为true启用，false禁用（默认false）
qpos_dim: 14          # qpos维度（ALOHA为14，根据你的机器人调整）
```

### 2. 配置示例

完整的配置示例已保存在 `/home/hlwang/Moto/moto_gpt_qpos_usage_example.yaml`

### 3. 数据要求

- HDF5数据集必须包含`observations/qpos`键（默认路径）
- qpos维度应与配置中的`qpos_dim`匹配
- 数据集会自动读取初始时刻的qpos并进行归一化

## 技术细节

### qpos归一化
- 使用与action相同的归一化统计量（`qpos_mean`和`qpos_std`）
- ALOHA双臂机器人（14维）：分别归一化左臂（前7维）和右臂（后7维）
- 其他机器人：只归一化最后7维

### Token序列组织
- qpos作为单个token插入到condition tokens的末尾
- 位置：language tokens → patch tokens → obs token → qpos token
- 通过attention机制，qpos信息可以传递给所有后续的latent motion和action预测tokens

### 向后兼容性
- 当`use_qpos_input=False`（默认）时，模型行为与原始版本完全相同
- 旧的checkpoint可以继续使用而无需修改
- 如果启用qpos但batch中没有`qpos_initial`，qpos会被设为None（模型仍可运行）

## 预期效果

添加qpos输入后，模型应该能够：
1. 更好地理解机器人当前的配置状态
2. 生成更加context-aware的动作预测
3. 提高对不同起始位置的泛化能力
4. 可能减少位置相关的错误

## 测试建议

1. **消融实验**：对比`use_qpos_input=True`和`False`的性能
2. **可视化**：检查qpos embedding的学习情况
3. **泛化测试**：测试在不同起始位置的性能
4. **维度验证**：确认HDF5中的qpos维度与配置匹配

## 注意事项

1. **维度匹配**：确保`qpos_dim`配置与实际HDF5数据中的qpos维度一致
2. **归一化**：确保使用了正确的归一化统计量（通过`compute_norm_stats.py`计算）
3. **内存**：qpos添加一个额外的token，对内存影响很小
4. **推理时**：推理时也需要提供qpos_initial（从传感器读取当前关节位置）

## 文件修改总结

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `hdf5_datasets.py` | 功能添加 | 添加完整qpos读取和归一化 |
| `moto_gpt.py` | 功能添加 | 添加qpos embedding和forward处理 |
| `moto_gpt_trainer_aloha.py` | 功能添加 | 添加qpos提取和传递逻辑 |
| `moto_gpt_qpos_usage_example.yaml` | 新建 | 使用示例配置文件 |

## 后续工作建议

1. 如果需要，可以为qpos添加单独的归一化统计量（而不是复用action的统计量）
2. 可以尝试更复杂的qpos embedding方式（如MLP）
3. 可以添加qpos的时序信息（当前只使用初始时刻的qpos）
4. 可以在eval_latent_motion_gen中也添加qpos支持（用于可视化生成）

---

实现日期：2025年
实现者：GitHub Copilot (Claude Sonnet 4.5)
