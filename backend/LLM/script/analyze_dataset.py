#!/usr/bin/env python3
import json
import os

# 统计训练集和验证集中的标签分布
train_file = './llm_dataset_processed/train_final.json'
dev_file = './llm_dataset_processed/dev_final.json'

def analyze_dataset(file_path):
    print(f"分析文件: {file_path}")
    label_counts = {}
    total_samples = 0
    unknown_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    messages = item.get('messages', [])
                    for msg in messages:
                        if msg.get('role') == 'assistant':
                            content = msg.get('content', '')
                            labels = [label.strip() for label in content.split(',') if label.strip()]
                            total_samples += 1
                            
                            if not labels:
                                unknown_count += 1
                                continue
                            
                            for label in labels:
                                if label == '未知':
                                    unknown_count += 1
                                label_counts[label] = label_counts.get(label, 0) + 1
                except Exception as e:
                    pass
    
    print(f"总样本数: {total_samples}")
    print(f"未知标签样本数: {unknown_count}")
    print(f"未知标签比例: {unknown_count/total_samples*100:.2f}%")
    print("标签分布:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count} ({count/total_samples*100:.2f}%)")
    print()

if __name__ == "__main__":
    analyze_dataset(train_file)
    analyze_dataset(dev_file)
