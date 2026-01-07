# LatentMotionTokenizer_Trainer 改进总结

## 🎯 问题描述
当前训练器在中断后恢复训练时，虽然模型权重被加载，但loss会突然变大，无法恢复到中断前的状态。

## 🔧 需要的改进

### 1. 添加导入语句
在文件顶部添加：
```python
import numpy as np
import random
```

### 2. 修改 __init__ 方法

#### 添加训练状态变量初始化：
```python
# 在 __init__ 方法开始处添加
self.start_epoch = 0
self.start_step = 0
self.best_loss = float('inf')
```

#### 将checkpoint加载移到最后：
```python
# 在 __init__ 方法最后，所有组件初始化完成后添加
if resume_ckpt_path is not None:
    self.load_checkpoint(resume_ckpt_path)
```

#### 移除原有的模型加载代码：
```python
# 删除这段代码：
if resume_ckpt_path is not None:
    print(f"resuming Latent Motion Tokenizer from {resume_ckpt_path} ...")
    missing_keys, unexpected_keys = latent_motion_tokenizer.load_state_dict(torch.load(os.path.join(resume_ckpt_path, 'pytorch_model.bin'), map_location='cpu'), strict=False)
    missing_root_keys = set([k.split(".")[0] for k in missing_keys])
    print('load ', resume_ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)
```

### 3. 替换 save_checkpoint 方法

将原有的：
```python
def save_checkpoint(self, save_dir):
    unwrapped_latent_motion_tokenizer = self.accelerator.unwrap_model(self.latent_motion_tokenizer)
    state_dict = unwrapped_latent_motion_tokenizer.get_state_dict_to_save()
    
    torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
    omegaconf.OmegaConf.save(unwrapped_latent_motion_tokenizer.config, os.path.join(save_dir, "config.yaml"))
    
    self.print(f"A new model checkpoint is saved to {save_dir}!!!")
```

替换为：
```python
def save_checkpoint(self, save_dir, epoch=None, step=None, loss=None):
    """Save complete checkpoint including model, optimizer, scheduler and training states."""
    os.makedirs(save_dir, exist_ok=True)
    
    unwrapped_latent_motion_tokenizer = self.accelerator.unwrap_model(self.latent_motion_tokenizer)
    model_state_dict = unwrapped_latent_motion_tokenizer.get_state_dict_to_save()
    
    # Save model weights and config
    torch.save(model_state_dict, os.path.join(save_dir, "pytorch_model.bin"))
    omegaconf.OmegaConf.save(unwrapped_latent_motion_tokenizer.config, os.path.join(save_dir, "config.yaml"))
    
    # Save training state
    training_state = {
        'epoch': epoch if epoch is not None else getattr(self, 'current_epoch', 0),
        'step': step if step is not None else getattr(self, 'current_step', 0),
        'best_loss': getattr(self, 'best_loss', float('inf')),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict(),
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }
    
    # Add CUDA RNG states if available
    if torch.cuda.is_available():
        training_state['cuda_rng_state'] = torch.cuda.get_rng_state()
        training_state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    
    torch.save(training_state, os.path.join(save_dir, "training_state.bin"))
    
    # Save accelerator state
    try:
        self.accelerator.save_state(save_dir)
    except Exception as e:
        self.print(f"Warning: Could not save accelerator state: {e}")
    
    # Update best loss
    if loss is not None and hasattr(self, 'best_loss'):
        if loss < self.best_loss:
            self.best_loss = loss

    self.print(f"Complete checkpoint saved to {save_dir}!")
```

### 4. 添加 load_checkpoint 方法

```python
def load_checkpoint(self, resume_ckpt_path):
    """Load complete checkpoint including model, optimizer, scheduler and training states."""
    self.print(f"Resuming training from {resume_ckpt_path}...")
    
    if not os.path.exists(resume_ckpt_path):
        self.print(f"Error: Checkpoint directory {resume_ckpt_path} does not exist!")
        return
    
    # Load model weights
    model_path = os.path.join(resume_ckpt_path, 'pytorch_model.bin')
    if os.path.exists(model_path):
        try:
            missing_keys, unexpected_keys = self.latent_motion_tokenizer.load_state_dict(
                torch.load(model_path, map_location='cpu'), strict=False)
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            self.print(f'✓ Loaded model weights')
            if missing_root_keys:
                self.print(f'  Missing key groups: {missing_root_keys}')
        except Exception as e:
            self.print(f"Error loading model weights: {e}")
            return
    
    # Load training state
    training_state_path = os.path.join(resume_ckpt_path, 'training_state.bin')
    if os.path.exists(training_state_path):
        try:
            training_state = torch.load(training_state_path, map_location='cpu')
            
            # Restore training progress
            self.start_epoch = training_state.get('epoch', 0)
            self.start_step = training_state.get('step', 0)
            self.best_loss = training_state.get('best_loss', float('inf'))
            
            # Restore optimizer state
            if 'optimizer_state_dict' in training_state:
                try:
                    self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
                    self.print(f"✓ Restored optimizer state")
                except Exception as e:
                    self.print(f"Warning: Could not restore optimizer state: {e}")
            
            # Restore scheduler state
            if 'scheduler_state_dict' in training_state:
                try:
                    self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
                    self.print(f"✓ Restored scheduler state")
                except Exception as e:
                    self.print(f"Warning: Could not restore scheduler state: {e}")
            
            # Restore random states
            if 'torch_rng_state' in training_state:
                torch.set_rng_state(training_state['torch_rng_state'])
            if 'numpy_rng_state' in training_state:
                np.random.set_state(training_state['numpy_rng_state'])
            if 'python_rng_state' in training_state:
                random.setstate(training_state['python_rng_state'])
            
            # Restore CUDA RNG states
            if torch.cuda.is_available():
                if 'cuda_rng_state' in training_state:
                    torch.cuda.set_rng_state(training_state['cuda_rng_state'])
                if 'cuda_rng_state_all' in training_state:
                    torch.cuda.set_rng_state_all(training_state['cuda_rng_state_all'])
            
            self.print(f"✓ Restored training state: epoch={self.start_epoch}, step={self.start_step}")
            
        except Exception as e:
            self.print(f"Error loading training state: {e}")
    
    # Load accelerator state
    accelerator_state_path = os.path.join(resume_ckpt_path, 'optimizer.bin')
    if os.path.exists(accelerator_state_path):
        try:
            self.accelerator.load_state(resume_ckpt_path)
            self.print(f"✓ Restored accelerator state")
        except Exception as e:
            self.print(f"Warning: Could not restore accelerator state: {e}")
    
    self.print(f"🎉 Checkpoint loading completed!")
```

### 5. 修改 train 方法

将开始的循环：
```python
def train(self):
    eval_loss_steps = len(self.train_prefetcher) // len(self.eval_prefetcher)
    step = 0
    
    for epoch in range(self.num_epochs+1):
```

修改为：
```python
def train(self):
    eval_loss_steps = len(self.train_prefetcher) // len(self.eval_prefetcher)
    
    # Get starting points from checkpoint if resuming
    start_epoch = getattr(self, 'start_epoch', 0)
    start_step = getattr(self, 'start_step', 0)
    step = start_step
    
    self.print(f"Starting training from epoch {start_epoch}, step {start_step}")
    
    for epoch in range(start_epoch, self.num_epochs+1):
```

### 6. 更新所有 save_checkpoint 调用

找到所有的：
```python
self.save_checkpoint(save_dir)
```

替换为：
```python
self.save_checkpoint(save_dir, epoch=epoch, step=step, loss=loss.get('loss', None))
```

### 7. 需要修改的所有类

这些改进需要应用到以下所有trainer类：
- `LatentMotionTokenizer_Trainer`
- `LatentMotionTokenizer_Trainer_Metaworld`
- `LatentMotionTokenizer_Trainer_RLBench`
- `LatentMotionTokenizer_Trainer_Multiview`

## ✅ 预期效果

实施这些改进后：
1. **优化器状态恢复** - 防止loss突然增大
2. **学习率调度器恢复** - 确保学习率正确
3. **训练进度恢复** - 从正确的epoch和step继续
4. **随机状态恢复** - 保证数据加载和训练的一致性
5. **完整状态保存** - 包含所有必要的训练信息

## 🚨 重要提醒

1. **备份原文件** - 在修改前请备份原始文件
2. **测试恢复功能** - 修改后务必测试checkpoint恢复是否正常工作
3. **兼容性检查** - 确保新的checkpoint格式与现有代码兼容
4. **逐步应用** - 建议先在一个trainer类上测试，确认无误后再应用到其他类

## 📝 使用方法

修改完成后，训练时loss不会在恢复后突然增大，可以无缝恢复到中断前的状态。checkpoint目录会包含：
- `pytorch_model.bin` - 模型权重
- `config.yaml` - 模型配置
- `training_state.bin` - 训练状态（新增）
- `optimizer.bin` - 优化器状态（accelerator保存）
- 其他accelerator相关文件