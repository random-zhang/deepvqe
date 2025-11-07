#!/bin/bash

# DeepVQE-AEC 训练配置脚本
# 使用方法: ./train_configs.sh [config_name]
# 可选配置: basic, high_performance, stable

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo -e "${BLUE}DeepVQE-AEC 训练配置脚本${NC}"
    echo ""
    echo "使用方法: $0 [配置名称]"
    echo ""
    echo -e "可用配置:"
    echo -e "  ${GREEN}basic${NC}           - 基础训练配置（推荐新手使用）"
    echo -e "  ${GREEN}high_performance${NC} - 高性能配置（需要充足GPU内存）"
    echo -e "  ${GREEN}stable${NC}          - 稳定训练配置（训练不稳定时使用）"
    echo -e "  ${GREEN}sophia${NC}          - Sophia优化器配置（二阶优化算法，适合大模型）"
    echo -e "  ${GREEN}custom${NC}          - 自定义配置（交互式设置）"
    echo ""
    echo "示例:"
    echo "  $0 basic"
    echo "  $0 high_performance"
    echo "  $0 stable"
    echo "  $0 sophia"
    echo "  $0 custom"
    echo ""
}

# 基础配置
run_basic_config() {
    echo -e "${GREEN}启动基础训练配置...${NC}"
    echo "配置详情:"
    echo "  - 优化器: AdamW"
    echo "  - 调度器: Cosine Annealing"
    echo "  - 学习率: 5e-4"
    echo "  - 权重衰减: 1e-4"
    echo "  - Warmup步数: 1000"
    echo "  - 早停: 启用 (patience=7)"
    echo "  - 混合精度: 启用"
    echo ""
    
    python train_aec.py \
        --optimizer adamw \
        --scheduler cosine \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --warmup_steps 1000 \
        --early_stopping \
        --patience 7 \
        --amp
}

# 高性能配置
run_high_performance_config() {
    echo -e "${GREEN}启动高性能训练配置...${NC}"
    echo "配置详情:"
    echo "  - 优化器: AdamW"
    echo "  - 调度器: Cosine Annealing"
    echo "  - 学习率: 1e-3"
    echo "  - 权重衰减: 1e-4"
    echo "  - 批次大小: 16"
    echo "  - 梯度累积: 2批次"
    echo "  - Warmup步数: 2000"
    echo "  - 早停: 启用 (patience=10)"
    echo "  - 混合精度: 启用"
    echo ""
    
    python train_aec.py \
        --optimizer adamw \
        --scheduler cosine \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --batch_size 16 \
        --accumulate_grad_batches 2 \
        --warmup_steps 2000 \
        --early_stopping \
        --patience 10 \
        --amp
}

# 稳定训练配置
run_stable_config() {
    echo -e "${GREEN}启动稳定训练配置...${NC}"
    echo "配置详情:"
    echo "  - 优化器: RAdam"
    echo "  - 调度器: ReduceLROnPlateau"
    echo "  - 学习率: 3e-4"
    echo "  - 权重衰减: 5e-5"
    echo "  - Warmup步数: 1500"
    echo "  - 早停: 启用 (patience=15)"
    echo "  - 梯度裁剪: 1.0"
    echo "  - 混合精度: 启用"
    echo ""
    
    python train_aec.py \
        --optimizer radam \
        --scheduler plateau \
        --lr 1e-3 \
        --weight_decay 5e-5 \
        --warmup_steps 1500 \
        --early_stopping \
        --patience 15 \
        --clip_grad 1.0 \
        --amp
}

# 自定义配置
run_custom_config() {
    echo -e "${BLUE}自定义训练配置${NC}"
    echo ""
    
    # 优化器选择
    echo "选择优化器:"
    echo "1) AdamW (推荐)"
    echo "2) Adam"
    echo "3) RAdam"
    echo "4) SGD"
    read -p "请输入选择 (1-4, 默认1): " opt_choice
    case $opt_choice in
        2) optimizer="adam" ;;
        3) optimizer="radam" ;;
        4) optimizer="sgd" ;;
        *) optimizer="adamw" ;;
    esac
    
    # 调度器选择
    echo ""
    echo "选择学习率调度器:"
    echo "1) Cosine Annealing (推荐)"
    echo "2) ReduceLROnPlateau"
    echo "3) StepLR"
    echo "4) ExponentialLR"
    echo "5) None"
    read -p "请输入选择 (1-5, 默认1): " sched_choice
    case $sched_choice in
        2) scheduler="plateau" ;;
        3) scheduler="step" ;;
        4) scheduler="exponential" ;;
        5) scheduler="none" ;;
        *) scheduler="cosine" ;;
    esac
    
    # 学习率
    read -p "学习率 (默认5e-4): " lr
    lr=${lr:-5e-4}
    
    # 批次大小
    read -p "批次大小 (默认8): " batch_size
    batch_size=${batch_size:-8}
    
    # 训练轮数
    read -p "训练轮数 (默认10): " epochs
    epochs=${epochs:-10}
    
    # 早停
    read -p "启用早停? (y/n, 默认y): " early_stop
    if [[ $early_stop == "n" || $early_stop == "N" ]]; then
        early_stopping_flag=""
        patience_flag=""
    else
        early_stopping_flag="--early_stopping"
        read -p "早停patience (默认7): " patience
        patience=${patience:-7}
        patience_flag="--patience $patience"
    fi
    
    echo ""
    echo -e "${GREEN}启动自定义训练配置...${NC}"
    echo "配置详情:"
    echo "  - 优化器: $optimizer"
    echo "  - 调度器: $scheduler"
    echo "  - 学习率: $lr"
    echo "  - 批次大小: $batch_size"
    echo "  - 训练轮数: $epochs"
    echo "  - 早停: $([ -n "$early_stopping_flag" ] && echo "启用 (patience=$patience)" || echo "禁用")"
    echo ""
    
    python train_aec.py \
        --optimizer $optimizer \
        --scheduler $scheduler \
        --lr $lr \
        --batch_size $batch_size \
        --epochs $epochs \
        $early_stopping_flag \
        $patience_flag \
        --amp
}

# 检查Python环境
check_environment() {
    if ! command -v python &> /dev/null; then
        echo -e "${RED}错误: 未找到Python环境${NC}"
        exit 1
    fi
    
    if [ ! -f "train_aec.py" ]; then
        echo -e "${RED}错误: 未找到train_aec.py文件${NC}"
        echo "请确保在正确的目录下运行此脚本"
        exit 1
    fi
}

# 主函数
main() {
    # 检查环境
    check_environment
    
    # 如果没有参数，显示帮助
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # 根据参数选择配置
    case $1 in
        "basic")
            run_basic_config
            ;;
        "high_performance")
            run_high_performance_config
            ;;
        "stable")
            run_stable_config
            ;;
        "sophia")
            echo -e "${BLUE}启动Sophia优化器训练配置...${NC}"
            echo "配置详情:"
            echo "  - 优化器: Sophia-G"
            echo "  - 调度器: Cosine Annealing"
            echo "  - 学习率: 2e-4"
            echo "  - 权重衰减: 1e-1"
            echo "  - Sophia rho: 0.04"
            echo "  - Sophia beta: (0.965, 0.99)"
            echo "  - Hessian更新间隔: 10"
            echo "  - 早停: 启用 (patience=10)"
            echo "  - 混合精度: 启用"
            echo ""
            
            python train_aec.py \
                --manifest_csv train.csv \
                --epochs 100 \
                --batch_size 8 \
                --lr 2e-4 \
                --optimizer sophia \
                --weight_decay 1e-1 \
                --sophia_beta1 0.965 \
                --sophia_beta2 0.99 \
                --sophia_rho 0.04 \
                --sophia_k 10 \
                --scheduler cosine \
                --warmup_steps 2000 \
                --early_stopping true \
                --patience 10 \
                --amp true \
                --log_interval 50
            ;;
        "custom")
            run_custom_config
            ;;
        "help" | "-h" | "--help")
            show_help
            ;;
        *)
            echo -e "${RED}错误: 未知配置 '$1'${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"