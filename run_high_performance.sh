#!/bin/bash

# DeepVQE-AEC 高性能训练脚本
# 针对速度优化的配置

echo "============================================================"
echo "DeepVQE-AEC 高性能训练配置"
echo "============================================================"

# 设置环境变量优化
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 基础参数
MANIFEST_CSV="train.csv"
CKPT_DIR="./checkpoints"
EPOCHS=100
BATCH_SIZE=16  # 增大批次大小
LR=0.001

# 高性能配置
echo "启动高性能训练..."
python train_aec_optimized.py \
    --manifest_csv $MANIFEST_CSV \
    --ckpt_dir $CKPT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --use_val \
    --amp \
    --precompute_stft \
    --compile_model \
    --num_workers 12 \
    --accumulate_grad_batches 2 \
    --optimizer adamw \
    --scheduler cosine \
    --loss_type sisnr \
    --log_interval 50 \
    --segment_frames 128 \
    --n_fft 512 \
    --hop_length 128 \
    --win_length 512

echo "高性能训练完成!"