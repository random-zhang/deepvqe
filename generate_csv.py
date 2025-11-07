#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import random
from pathlib import Path

def generate_csv(data_dir, output_csv):
    """
    生成包含音频文件路径的CSV文件
    
    参数:
        data_dir: 数据目录路径
        output_csv: 输出CSV文件路径
    """
    # 确保数据目录存在
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"错误: 数据目录 {data_dir} 不存在")
        return
    
    # 查找所有_mic.wav文件作为基础
    mic_files = list(data_path.glob("**/*_mic.wav"))
    
    if not mic_files:
        print(f"警告: 在 {data_dir} 中未找到任何 *_mic.wav 文件")
        return
    # 准备CSV数据
    csv_data = []
    
    for mic_file in mic_files:
        # 获取文件名（不含扩展名）
        base_name = mic_file.stem  # 去掉_mic部分
        
        # 构建其他文件路径
        parent_dir = mic_file.parent
        
        # 根据实际文件名模式推断其他文件路径
        # 从现有文件看，模式是 f00000_mic.wav, f00000_far.wav 等
        echo_file = parent_dir / f"{base_name.replace('_mic', '')}_echo.wav"  # f00000_echo.wav
        farnerd_file = parent_dir / f"{base_name.replace('_mic', '')}_farend.wav"  # f00000_far.wav
        mix_file = parent_dir / f"{base_name.replace('_mic', '')}_mic.wav"  # f00000_mix.wav
        nearend_file = parent_dir / f"{base_name.replace('_mic', '')}_nearend.wav"  # f00000_nearend.wav
        target_file = parent_dir / f"{base_name.replace('_mic', '')}_target.wav"  # f00000_target.wav
        
        # 检查文件是否存在
        files_exist = all([
            echo_file.exists(),
            farnerd_file.exists(),
            mix_file.exists(),
            nearend_file.exists(),
            target_file.exists()
        ])
        
        if not files_exist:
            print(f"警告: 文件 {base_name} 的某些关联文件不存在")
            continue
        
        # 添加到CSV数据
        csv_data.append({
            "filename": base_name.replace('_mic', ''),
            "echo_filepath": str(echo_file),
            "farnerd_filepath": str(farnerd_file),
            "mix_filepath": str(mix_file),
            "nearend_filepath": str(nearend_file),
            "target_filepath": str(target_file)
        })
    
    # 随机打乱数据
    random.shuffle(csv_data)
    
    # 按照8:1:1的比例分割数据集
    total_count = len(csv_data)
    train_count = int(total_count * 0.8)
    val_count = int(total_count * 0.1)
    test_count = total_count - train_count - val_count  # 确保总数一致
    
    # 为每个数据样本添加分割标记
    for i, row in enumerate(csv_data):
        if i < train_count:
            row['split'] = 'train'
        elif i < train_count + val_count:
            row['split'] = 'val'
        else:
            row['split'] = 'test'
    
    # 写入CSV文件
    if csv_data:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'echo_filepath', 'farnerd_filepath', 'mix_filepath', 'nearend_filepath', 'target_filepath', 'split']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 写入数据行
            for row in csv_data:
                writer.writerow(row)
        
        print(f"成功生成CSV文件: {output_csv}")
        print(f"共包含 {len(csv_data)} 条记录")
        print(f"训练集: {train_count} 条记录")
        print(f"验证集: {val_count} 条记录")
        print(f"测试集: {test_count} 条记录")
    else:
        print("错误: 没有找到有效的数据记录")

if __name__ == "__main__":
    # 设置默认路径
    data_directory = "/media/m1998cc/n/pythonProject/deepvqe/data"
    output_csv_file = "/media/m1998cc/n/pythonProject/deepvqe/train.csv"
    
    # 生成CSV文件
    generate_csv(data_directory, output_csv_file)