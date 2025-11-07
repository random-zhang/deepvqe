#!/bin/bash

# DeepVQE-AEC 优化版训练脚本使用示例
# 这个脚本展示了如何使用优化版本的训练脚本进行训练

# 基本配置
MANIFEST_CSV="path/to/your/manifest.csv"  # 请替换为实际的数据清单路径
CKPT_DIR="./checkpoints_optimized"
EPOCHS=100
BATCH_SIZE=8
LR=1e-3

echo "开始使用优化版本训练 DeepVQE-AEC 模型..."

# 基础训练命令（推荐配置）
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
    --early_stopping \
    --patience 10 \
    --optimizer adamw \
    --scheduler cosine \
    --loss_type sisnr \
    --num_workers 8 \
    --log_interval 50 \
    --test_after_training

echo "训练完成！"

# 如果只想运行测试，可以使用以下命令：
# python train_aec_optimized.py \
#     --test \
#     --manifest_csv $MANIFEST_CSV \
#     --checkpoint $CKPT_DIR/deepvqe_aec_best.pt \
#     --batch_size $BATCH_SIZE

# 如果要从检查点恢复训练，可以使用：
# python train_aec_optimized.py \
#     --manifest_csv $MANIFEST_CSV \
#     --ckpt_dir $CKPT_DIR \
#     --resume_checkpoint $CKPT_DIR/deepvqe_aec_epoch50.pt \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --use_val \
#     --amp \
#     --precompute_stft \
#     --compile_model