# DeepVQE-AEC 训练性能优化指南

## 🚀 主要性能瓶颈分析

### 1. 数据加载瓶颈 (最严重)
- **问题**: `num_workers=0` 导致单线程数据加载
- **影响**: 数据加载成为训练的主要瓶颈，GPU利用率低
- **解决方案**: 设置 `num_workers=4-8`

### 2. 实时STFT计算
- **问题**: 每个样本都实时计算STFT，CPU密集
- **影响**: 每个epoch重复相同的计算
- **解决方案**: 预计算STFT并缓存

### 3. 内存传输开销
- **问题**: CPU-GPU数据传输频繁
- **影响**: 增加训练时间
- **解决方案**: 启用 `pin_memory=True`

## 🛠️ 具体优化建议

### 立即可实施的优化 (简单)

#### 1. 修改数据加载器参数
```python
# 在 train_aec.py 中修改 DataLoader 配置
dl = DataLoader(
    ds, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=4,  # 改为4-8
    collate_fn=collate_fn, 
    drop_last=True,
    pin_memory=True,  # 添加这行
    persistent_workers=True  # 添加这行
)
```

#### 2. 启用PyTorch优化
```python
# 在训练开始前添加
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
```

#### 3. 减少验证频率
```python
# 每5个epoch验证一次，而不是每个epoch
if epoch % 5 == 0:
    validate()
```

### 中等难度优化

#### 1. 预计算STFT缓存
- 第一次运行时预计算所有STFT
- 后续训练直接加载缓存
- 可节省50-70%的数据预处理时间

#### 2. 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 在训练循环中
with autocast():
    output = model(X_mic, X_far)
    loss = criterion(output, X_clean)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 3. 梯度累积优化
```python
# 减少实际batch_size，增加accumulate_grad_batches
# 例如: batch_size=4, accumulate_grad_batches=4
# 等效于 batch_size=16，但内存使用更少
```

### 高级优化 (复杂)

#### 1. 模型编译 (PyTorch 2.0+)
```python
model = torch.compile(model)
```

#### 2. 数据预处理优化
- 使用GPU进行STFT计算
- 实现自定义CUDA kernel
- 使用TensorRT优化推理

#### 3. 分布式训练
```python
# 使用多GPU训练
python -m torch.distributed.launch --nproc_per_node=2 train_aec.py
```

## 📊 预期性能提升

| 优化项目 | 预期提升 | 实施难度 |
|---------|---------|---------|
| num_workers=4 | 2-4x | 简单 |
| pin_memory | 10-20% | 简单 |
| STFT缓存 | 50-70% | 中等 |
| 混合精度 | 20-30% | 中等 |
| 模型编译 | 10-15% | 简单 |
| 分布式训练 | 1.8x (2GPU) | 复杂 |

## 🔧 快速修复方案

### 方案1: 最小修改 (推荐)
只需修改 `train_aec.py` 中的几行代码：

```python
# 第1步: 修改DataLoader
num_workers = 4  # 或者 min(8, os.cpu_count())
dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, 
                num_workers=num_workers, collate_fn=collate_fn, 
                drop_last=True, pin_memory=True, persistent_workers=True)

# 第2步: 启用cudnn优化
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# 第3步: 减少验证频率
if epoch % 5 == 0 and args.use_val:  # 每5个epoch验证一次
    validate()
```

### 方案2: 使用优化版本脚本
使用我创建的 `train_aec_optimized.py`，包含所有优化。

## 🎯 建议的实施顺序

1. **立即实施**: 修改 `num_workers` 和 `pin_memory`
2. **短期**: 启用cudnn优化，减少验证频率
3. **中期**: 实施STFT缓存
4. **长期**: 混合精度训练，模型编译

## 💡 监控性能

使用以下命令监控训练性能：
```bash
# 监控GPU使用率
nvidia-smi -l 1

# 监控CPU使用率
htop

# 在训练脚本中添加时间统计
import time
start_time = time.time()
# ... 训练代码 ...
print(f"Epoch time: {time.time() - start_time:.2f}s")
```

通过这些优化，你的训练速度应该能提升 **3-5倍**！