#!/root/miniconda3/bin/python
"""
修改训练数据集的 prompt，强化逗号分隔要求

1. 将标签列表中的中文顿号改为英文逗号
2. 强化输出格式说明，添加明确示例
3. 确保模型学会用逗号分隔多个标签
"""

import json
import sys
import os


def modify_prompt(prompt):
    """修改 prompt，强化逗号分隔要求
    
    Args:
        prompt: 原始 prompt 文本
        
    Returns:
        str: 修改后的 prompt 文本
    """
    # 将标签列表中的顿号改为逗号
    prompt = prompt.replace('、', ',')
    
    # 强化输出格式说明
    if '输出格式:标签1,标签2,...' in prompt:
        prompt = prompt.replace(
            '输出格式:标签1,标签2,...',
            '输出格式要求：请使用英文逗号(,)分隔多个标签，例如：市场,中立,煤炭'
        )
    elif '输出格式:标签1,标签2' in prompt:
        prompt = prompt.replace(
            '输出格式:标签1,标签2',
            '输出格式要求：请使用英文逗号(,)分隔多个标签，例如：市场,中立,煤炭'
        )
    
    return prompt


def process_file(input_path, output_path):
    """处理数据文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    count = 0
    modified = 0
    
    print(f'处理文件: {input_path}')
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        messages = item.get('messages', [])
                        
                        # 查找并修改 user 消息中的 prompt
                        for msg in messages:
                            if msg.get('role') == 'user':
                                original_content = msg['content']
                                new_content = modify_prompt(original_content)
                                
                                if new_content != original_content:
                                    modified += 1
                                    if count < 3:
                                        print(f'\n?????? {line_num} ?????????:')
                                        print(f'  {original_content[:200]}...')
                                        print(f'?????????:')
                                        print(f'  {new_content[:200]}...')
                                        print()
                                    msg['content'] = new_content
                                break
                        
                        # ????????????????????????
                        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                        count += 1
                        
                    except Exception as e:
                        print(f'??? {line_num} ??????: {e}', file=sys.stderr)
    
    print(f'\n????????????:')
    print(f'  ????????????: {count}')
    print(f'  ???????????????: {modified}')
    print(f'  ????????????: {output_path}')
    return count, modified


def main():
    """?????????"""
    base_dir = '/root/autodl-tmp/LLM/llm_dataset_processed'
    
    print('=' * 60)
    print('??????????????????????????? prompt')
    print('=' * 60)
    print()
    
    # ???????????????
    print('??????????????? (train.json)...\n')
    train_count, train_modified = process_file(
        os.path.join(base_dir, 'train.json'),
        os.path.join(base_dir, 'train_fixed.json')
    )
    
    print()
    print('-' * 60)
    print()
    
    # ???????????????
    print('??????????????? (dev.json)...\n')
    dev_count, dev_modified = process_file(
        os.path.join(base_dir, 'dev.json'),
        os.path.join(base_dir, 'dev_fixed.json')
    )
    
    print()
    print('=' * 60)
    print('???????????????????????????')
    print('=' * 60)
    print()
    print('????????????:')
    print(f'  ?????????: {train_count} ??????, {train_modified} ??????')
    print(f'  ?????????: {dev_count} ??????, {dev_modified} ??????')
    print()
    print('???????????????????????????????????????:')
    print(f'  {os.path.join(base_dir, "train_fixed.json")}')
    print(f'  {os.path.join(base_dir, "dev_fixed.json")}')
    print()
    print('????????? train_lora.py ???????????????????????????:')
    print(f'  --train_data {os.path.join(base_dir, "train_fixed.json")}')
    print(f'  --dev_data {os.path.join(base_dir, "dev_fixed.json")}')


if __name__ == '__main__':
    main()
