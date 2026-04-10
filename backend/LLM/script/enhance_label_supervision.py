#!/usr/bin/env python3
"""
改进标签监督方法，为每个标签添加理由，帮助模型更好地理解标签的适用性
"""
import json
import os

# 标签到关键词的映射，用于生成标签理由
LABEL_KEYWORDS = {
    '国际': ['国际', '全球', '美国', '美联储', '央行', '外币', '汇率', '进出口', '跨国', '全球经济'],
    '经济活动': ['经济', 'GDP', '就业', '通胀', '消费', '投资', '增长', '衰退', '经济数据', '宏观经济'],
    '市场': ['市场', '价格', '指数', '股票', '债券', '交易', '行情', '走势', '市场分析', '市场预测'],
    '金属': ['金属', '铜', '铝', '锌', '铅', '锡', '镍', '贵金属', '钢材', '金属价格'],
    '政策': ['政策', '法规', '条例', '措施', '调控', '改革', '指导', '意见', '政策发布', '政策影响'],
    '中立': ['报道', '数据', '信息', '公告', '发布', '统计', '报告', '新闻', '客观', '事实'],
    '积极': ['增长', '上升', '提高', '增加', '利好', '上涨', '创新高', '改善', '积极', '乐观'],
    '煤炭': ['煤炭', '焦炭', '动力煤', '焦煤', '煤矿', '煤价', '煤化工', '煤炭行业', '煤炭生产'],
    '农业': ['农业', '粮食', '作物', '种植', '收割', '农产品', '农业政策', '农村经济', '农业生产'],
    '能源': ['能源', '石油', '天然气', '电力', '新能源', '可再生能源', '能源价格', '能源政策'],
    '畜牧': ['畜牧', '养殖', '生猪', '家禽', '肉类', '饲料', '畜牧业', '养殖行业'],
    '政治': ['政治', '政府', '选举', '领导人', '政策', '外交', '国际关系', '政治事件'],
    '消极': ['下降', '减少', '下跌', '亏损', '利空', '下滑', '创新低', '恶化', '消极', '悲观'],
    '未知': []
}

def generate_label_reason(text, label):
    """
    为给定的标签生成理由
    
    Args:
        text: 文本内容
        label: 标签
    
    Returns:
        str: 标签理由
    """
    keywords = LABEL_KEYWORDS.get(label, [])
    
    # 检查文本中是否包含标签相关的关键词
    matched_keywords = [kw for kw in keywords if kw in text]
    
    if matched_keywords:
        # 如果有匹配的关键词，基于关键词生成理由
        reason = f"文本中包含'{matched_keywords[0]}'等与{label}相关的内容"
    else:
        # 如果没有匹配的关键词，基于标签的一般含义生成理由
        label_descriptions = {
            '国际': '文本涉及国际金融或全球经济相关内容',
            '经济活动': '文本描述宏观经济运行或经济数据',
            '市场': '文本讨论金融市场或价格走势',
            '金属': '文本与金属行业或金属价格相关',
            '政策': '文本涉及政府政策或法规措施',
            '中立': '文本是客观的信息报道或数据发布',
            '积极': '文本包含正面或利好的信息',
            '煤炭': '文本与煤炭行业或煤炭价格相关',
            '农业': '文本涉及农业或农产品相关内容',
            '能源': '文本与能源行业或能源价格相关',
            '畜牧': '文本涉及畜牧业或养殖相关内容',
            '政治': '文本与政治事件或政府行为相关',
            '消极': '文本包含负面或利空的信息',
            '未知': '文本内容无法明确归类到其他标签'
        }
        reason = label_descriptions.get(label, '文本与该标签相关')
    
    return reason

def enhance_label_supervision(input_file, output_file):
    """
    增强标签监督，为每个标签添加理由
    
    Args:
        input_file: 输入数据文件路径
        output_file: 输出数据文件路径
    """
    enhanced_count = 0
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
                        text_content = user_content.split("文本内容:")[-1].split("\n")[0].strip()
                    else:
                        text_content = user_content
                    
                    # 提取标签
                    assistant_content = assistant_msg.get('content', '')
                    labels = [label.strip() for label in assistant_content.split(',') if label.strip()]
                    
                    # 为每个标签生成理由
                    if labels:
                        enhanced_content = []
                        for label in labels:
                            reason = generate_label_reason(text_content, label)
                            enhanced_content.append(f"{label}（{reason}）")
                        
                        # 更新助手消息
                        assistant_msg['content'] = ','.join(enhanced_content)
                        enhanced_count += 1
                
                # 写入增强后的内容
                json.dump(item, f_out, ensure_ascii=False)
                f_out.write('\n')
                
            except Exception as e:
                print(f"处理行时出错: {e}")
                # 出错时写入原始内容
                f_out.write(line)
                f_out.write('\n')
    
    print(f"处理完成: 共 {total_count} 条，增强 {enhanced_count} 条")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    # 增强训练集
    train_input = '../llm_dataset_processed/train_improved_prompt.json'
    train_output = '../llm_dataset_processed/train_enhanced_labels.json'
    print("增强训练集标签监督...")
    enhance_label_supervision(train_input, train_output)
    
    # 增强验证集
    dev_input = '../llm_dataset_processed/dev_improved_prompt.json'
    dev_output = '../llm_dataset_processed/dev_enhanced_labels.json'
    print("\n增强验证集标签监督...")
    enhance_label_supervision(dev_input, dev_output)
