#!/usr/bin/env python3
"""
运行所有改进步骤
"""
import subprocess
import os

def run_script(script_name):
    """
    运行指定的脚本
    
    Args:
        script_name: 脚本名称
    """
    print(f"\n运行 {script_name}...")
    result = subprocess.run(['python', script_name], 
                          cwd=os.path.dirname(os.path.abspath(__file__)),
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("错误输出:")
        print(result.stderr)
    print(f"{script_name} 运行完成\n")

if __name__ == "__main__":
    print("开始运行所有改进步骤...")
    
    # 1. 改进 Prompt 设计
    run_script('improve_prompt.py')
    
    # 2. 增强标签监督
    run_script('enhance_label_supervision.py')
    
    # 3. 处理增强后的标签
    run_script('process_enhanced_labels.py')
    
    print("所有改进步骤运行完成！")
    print("改进后的数据集文件:")
    print("- 训练集: ../llm_dataset_processed/train_final_processed.json")
    print("- 验证集: ../llm_dataset_processed/dev_final_processed.json")
