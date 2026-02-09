# MotoGPT添加qpos输入 - 快速开始指南

## 我做了什么修改？

为MotoGPT模型添加了**可选的qpos（当前机器人关节位置）输入**功能。这样模型在预测动作时可以知道机器人当前的配置状态。

## 如何使用？

### 最简单的方式：

在你的模型配置YAML文件中添加两行：

```yaml
use_qpos_input: true  # 启用qpos输入
qpos_dim: 14          # ALOHA机器人的qpos维度是14
```

就这么简单！其他不需要修改。

### 完整示例

参考配置文件：`/home/hlwang/Moto/moto_gpt_qpos_usage_example.yaml`

## 修改了哪些文件？

1. **数据加载** (`hdf5_datasets.py`)：自动读取HDF5中的qpos数据
2. **模型** (`moto_gpt.py`)：添加qpos embedding层
3. **训练器** (`moto_gpt_trainer_aloha.py`)：将qpos传递给模型

## 向后兼容性

- 如果不设置`use_qpos_input: true`，模型行为和之前完全一样
- 旧的checkpoint可以继续使用
- 不会影响现有的训练流程

## 数据要求

确保你的HDF5数据集包含qpos数据：
- 路径：`observations/qpos`（这是ALOHA数据集的默认路径）
- 维度：14（ALOHA双臂机器人）

## 工作原理

模型的输入序列变成：
```
[语言指令] → [图像patch] → [观察特征] → [qpos] → [动作预测]
```

qpos被编码成一个token，通过attention机制影响后续的动作预测。

## 预期效果

- ✅ 模型能感知机器人当前位置
- ✅ 对不同起始位置的泛化能力更强
- ✅ 动作预测更加context-aware

## 测试一下

### 1. 不使用qpos（baseline）
```yaml
use_qpos_input: false
```

### 2. 使用qpos
```yaml
use_qpos_input: true
qpos_dim: 14
```

对比两者的性能差异！

## 问题排查

### 如果遇到维度错误
检查配置中的`qpos_dim`是否与HDF5数据中的qpos维度一致。

### 如果遇到key错误
确认HDF5文件中有`observations/qpos`这个key。

## 详细文档

完整的实现细节和技术说明请查看：
`/home/hlwang/Moto/qpos_input_implementation_summary.md`

---

**提示**：这是一个可选功能，你可以随时通过设置`use_qpos_input: false`来关闭它。
