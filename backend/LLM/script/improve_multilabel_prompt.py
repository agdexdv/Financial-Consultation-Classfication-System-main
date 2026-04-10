#!/usr/bin/env python3
"""
改进多标签分类的 Prompt 设计，明确鼓励模型生成多个标签
"""
import json
import os

def improve_multilabel_prompt(input_file, output_file):
    """
    改进多标签分类的 Prompt 设计
    
    Args:
        input_file: 输入数据文件路径
        output_file: 输出数据文件路径
    """
    improved_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            try:
                item = json.loads(line)
                messages = item.get('messages', [])
                
                # 找到用户消息和助手消息
                user_msg = None
                assistant_msg = None
                
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_msg = msg
                    elif msg.get('role') == 'assistant':
                        assistant_msg = msg
                
                if user_msg and assistant_msg:
                    # 提取文本内容
                    user_content = user_msg.get('content', '')
                    if "文本内容:" in user_content:
                        text_content = user_msg.get('content', '').split("文本内容:")[-1].split("\n")[0].strip()
                    else:
                        text_content = user_content
                    
                    # 构建改进的多标签 Prompt
                    improved_prompt = f"""请仔细分析以下金融文本内容，理解其核心含义和上下文，然后判断其所属的所有相关标签类别。

标签说明：
- 国际：涉及国际金融、全球经济、跨国公司等内容
- 经济活动：关于经济运行、GDP、就业、消费等宏观经济内容
- 市场：关于金融市场、价格走势、交易等内容
- 金属：关于金属行业、价格、生产等内容
- 政策：关于政府政策、法规、调控措施等内容
- 中立：客观报道、数据发布等中性内容
- 积极：包含正面、增长、利好等积极信息
- 煤炭：关于煤炭行业、价格、生产等内容
- 农业：关于农业、农产品、农村经济等内容
- 能源：关于能源行业、价格、政策等内容
- 畜牧：关于畜牧业、养殖、肉类产品等内容
- 政治：关于政治事件、政府行为等内容
- 消极：包含负面、下降、利空等消极信息

重要提示：
1. 一个文本可能属于多个标签类别，请输出所有相关的标签
2. 基于文本的整体含义和上下文进行判断，而不仅仅是关键词匹配
3. 确保输出的标签都是相关的，不要输出无关的标签
4. 输出格式要求：请使用英文逗号(,)分隔多个标签，例如：市场,中立,煤炭

文本内容:{text_content}

标签:"""
                    
                    # 更新用户消息
                    user_msg['content'] = improved_prompt
                    improved_count += 1
                
                # 写入改进后的内容
                json.dump(item, f_out, ensure_ascii=False)
                f_out.write('\n')
                
            except Exception as e:
                print(f"处理行时出错: {e}")
                # 出错时写入原始内容
                f_out.write(line)
                f_out.write('\n')
    
    print(f"处理完成: 共 {total_count} 条，改进 {improved_count} 条")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    # 改进训练集
    train_input = '../llm_dataset_processed/train_final_processed.json'
    train_output = '../llm_dataset_processed/train_multilabel_prompt.json'
    print("改进训练集多标签 Prompt...")
    improve_multilabel_prompt(train_input, train_output)
    
    # 改进验证集
    dev_input = '../llm_dataset_processed/dev_final_processed.json'
    dev_output = '../llm_dataset_processed/dev_multilabel_prompt.json'
    print("\n改进验证集多标签 Prompt...")
    improve_multilabel_prompt(dev_input, dev_output)
