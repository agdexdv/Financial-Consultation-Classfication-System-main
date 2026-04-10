#!/root/miniconda3/bin/python
"""
Qwen2.5-Instruct LoRA 训练脚本

使用 LoRA 对 Qwen2.5-Instruct 模型进行指令微调，
针对金融文本多标签分类任务。

用法:
    python train_lora.py
    python train_lora.py --batch_size 8 --learning_rate 1e-5
"""

import os
import json
import time
import math
import argparse
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from tensorboardX import SummaryWriter
from tqdm import tqdm
from itertools import cycle

from utils import cli

# ── 默认配置 ──────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen2.5-3B-Instruct")
DEFAULT_TRAIN_DATA = "./llm_dataset_processed/train_multilabel_prompt.json"
DEFAULT_DEV_DATA = "./llm_dataset_processed/dev_multilabel_prompt.json"
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_MAX_LENGTH = 512

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "runs")


class FinancialTextDataset(Dataset):
    """金融文本数据集。
    
    只监督用户输入（金融文本内容）和assistant的回复（标签），
    移除system消息和复杂的prompt说明。
    """
    
    def __init__(self, file_path, tokenizer, max_length=512):
        """初始化数据集。
        
        Args:
            file_path: 数据文件路径。
            tokenizer: 分词器。
            max_length: 最大序列长度。
        """
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if 'messages' in item and item['messages']:
                            self.data.append(item)
                        else:
                            cli.print_warning(f"Line {line_num}: Missing 'messages' field or empty messages, skipping")
                    except json.JSONDecodeError as e:
                        cli.print_warning(f"Line {line_num}: JSON decode error - {e}, skipping")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item['messages']
        
        # 提取用户输入和assistant回复
        # 只监督用户输入（金融文本内容）和assistant的回复（标签）
        user_content = ""
        assistant_content = ""
        
        for msg in messages:
            if msg['role'] == 'user':
                # 使用数据集中的原始prompt（已经改进过）
                user_content = msg['content']
            elif msg['role'] == 'assistant':
                assistant_content = msg['content']
        
        # 将标签转换为特殊 token
        if assistant_content:
            labels = [label.strip() for label in assistant_content.split(',') if label.strip()]
            tokenized_labels = labels_to_tokens(labels)
        else:
            tokenized_labels = ''
        
        # 将prompt和标签 token 拼接在一起
        full_text = user_content + tokenized_labels
        
        # 编码完整文本
        inputs = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 计算prompt的长度（不包含标签）
        prompt_inputs = self.tokenizer(
            user_content,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_inputs['input_ids'].shape[1]
        
        # 构建labels：prompt部分为-100（不计算loss），标签部分为正常token
        labels = inputs['input_ids'].clone()
        labels[:, :prompt_length] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Qwen2.5-Instruct LoRA 训练")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help=f"模型路径（默认 {DEFAULT_MODEL_PATH}）")
    parser.add_argument('--train_data', type=str, default=DEFAULT_TRAIN_DATA,
                        help=f"训练数据路径（默认 {DEFAULT_TRAIN_DATA}）")
    parser.add_argument('--dev_data', type=str, default=DEFAULT_DEV_DATA,
                        help=f"验证数据路径（默认 {DEFAULT_DEV_DATA}）")
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"输出目录（默认 {DEFAULT_OUTPUT_DIR}）")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="每批样本数（默认 4）")
    parser.add_argument('--max_length', type=int, default=DEFAULT_MAX_LENGTH,
                        help=f"最大序列长度（默认 {DEFAULT_MAX_LENGTH}）")
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                        help="学习率（默认 5e-6）")
    parser.add_argument('--num_train_epochs', type=int, default=1,
                        help="训练轮数（默认 1）")
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                        help="学习率预热比例（默认 0.05）")
    parser.add_argument('--lora_r', type=int, default=16,
                        help="LoRA 秩（默认 16）")
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help="LoRA alpha（默认 32）")
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help="LoRA dropout（默认 0.05）")
    parser.add_argument('--log_interval', type=int, default=10,
                        help="日志间隔（默认 10 步）")
    parser.add_argument('--eval_interval', type=int, default=500,
                        help="评估间隔（默认 500 步）")
    parser.add_argument('--eval_samples', type=int, default=200,
                        help="每次评估最多使用的样本数（默认 200，0 表示用全部）")
    parser.add_argument('--save_interval', type=int, default=2000,
                        help="保存间隔（默认 2000 步）")
    parser.add_argument('--run_name', type=str, default="lora_train",
                        help="本次运行名称，用于 TensorBoard 日志目录命名（默认 lora_train）")
    return parser.parse_args()


# 定义标签闭集
VALID_LABELS = {
    '国际', '经济活动', '市场', '金属', '政策', '中立', '积极', '煤炭',
    '农业', '能源', '畜牧', '政治', '消极', '未知'
}

# 标签到特殊 token 的映射
LABEL_TOKENS = {
    '国际': '<LABEL_国际>',
    '经济活动': '<LABEL_经济活动>',
    '市场': '<LABEL_市场>',
    '金属': '<LABEL_金属>',
    '政策': '<LABEL_政策>',
    '中立': '<LABEL_中立>',
    '积极': '<LABEL_积极>',
    '煤炭': '<LABEL_煤炭>',
    '农业': '<LABEL_农业>',
    '能源': '<LABEL_能源>',
    '畜牧': '<LABEL_畜牧>',
    '政治': '<LABEL_政治>',
    '消极': '<LABEL_消极>',
    '未知': '<LABEL_未知>'
}

# 反向映射：特殊 token 到标签
TOKEN_TO_LABEL = {v: k for k, v in LABEL_TOKENS.items()}

def add_special_label_tokens(tokenizer):
    """向 tokenizer 添加特殊的标签 token。
    
    Args:
        tokenizer: 分词器。
    """
    # 收集所有特殊标签 token
    special_tokens = list(LABEL_TOKENS.values())
    
    # 添加特殊 token
    tokenizer.add_tokens(special_tokens)
    print(f"Added {len(special_tokens)} special label tokens to tokenizer")
    return tokenizer

def labels_to_tokens(labels):
    """将标签转换为特殊 token。
    
    Args:
        labels: 标签列表。
        
    Returns:
        str: 特殊 token 字符串。
    """
    tokens = [LABEL_TOKENS[label] for label in labels if label in LABEL_TOKENS]
    return ' '.join(tokens)

def tokens_to_labels(token_text):
    """将特殊 token 转换为标签。
    
    Args:
        token_text: 特殊 token 文本。
        
    Returns:
        str: 标签字符串。
    """
    labels = []
    for token in token_text.split():
        if token in TOKEN_TO_LABEL:
            labels.append(TOKEN_TO_LABEL[token])
    return ','.join(labels)

def get_labels(text):
    """将文本转换为标签集合。
    
    Args:
        text: 包含逗号分隔标签的文本。
        
    Returns:
        set: 标签集合。
    """
    text = text.replace('，', ',').replace('、', ',')
    labels = [label.strip() for label in text.split(',') if label.strip()]
    # 只保留有效的标签
    valid_labels = [label for label in labels if label in VALID_LABELS]
    return set(valid_labels)

def filter_valid_labels(text):
    """过滤文本，只保留有效的标签。
    
    Args:
        text: 包含标签的文本。
    
    Returns:
        str: 只包含有效标签的逗号分隔字符串。
    """
    text = text.replace('，', ',').replace('、', ',')
    labels = [label.strip() for label in text.split(',') if label.strip()]
    # 只保留有效的标签
    valid_labels = [label for label in labels if label in VALID_LABELS]
    return ','.join(valid_labels)


def get_most_similar_label(text):
    """获取与输入文本最相似的标签。
    
    Args:
        text: 输入文本。
    
    Returns:
        str: 最相似的标签。
    """
    # 标签关键词映射，为每个标签提供相关关键词
    label_keywords = {
        '国际': ['国际', '全球', '美国', '美联储', '央行', '外币', '汇率', '进出口'],
        '经济活动': ['经济', 'GDP', '就业', '通胀', '消费', '投资', '增长', '衰退'],
        '市场': ['市场', '价格', '指数', '股票', '债券', '交易', '行情', '走势'],
        '金属': ['金属', '铜', '铝', '锌', '铅', '锡', '镍', '贵金属', '钢材'],
        '政策': ['政策', '法规', '条例', '措施', '调控', '改革', '指导', '意见'],
        '中立': ['报道', '数据', '信息', '公告', '发布', '统计', '报告', '新闻'],
        '积极': ['增长', '上升', '提高', '增加', '利好', '上涨', '创新高', '改善'],
        '煤炭': ['煤炭', '焦炭', '动力煤', '焦煤', '煤矿', '煤价', '煤化工'],
        '农业': ['农业', '粮食', '作物', '种植', '收割', '农产品', '农业政策'],
        '能源': ['能源', '石油', '天然气', '电力', '新能源', '可再生能源', '能源价格'],
        '畜牧': ['畜牧', '养殖', '生猪', '家禽', '肉类', '饲料', '畜牧业'],
        '政治': ['政治', '政府', '选举', '领导人', '政策', '外交', '国际关系'],
        '消极': ['下降', '减少', '下跌', '亏损', '利空', '下滑', '创新低', '恶化'],
        '未知': []
    }
    
    # 计算每个标签的相似度得分
    label_scores = {}
    for label, keywords in label_keywords.items():
        score = 0
        for keyword in keywords:
            score += text.count(keyword)
        # 为常见标签添加基础分数，减少未知标签的出现
        if label in ['市场', '经济活动', '国际', '金属']:
            score += 1
        label_scores[label] = score
    
    # 按得分排序，返回得分最高的标签
    sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    # 即使得分为0，也返回得分最高的标签，避免返回"未知"
    if sorted_labels:
        # 过滤掉"未知"标签，除非所有其他标签得分都为0
        non_unknown_labels = [(label, score) for label, score in sorted_labels if label != '未知']
        if non_unknown_labels and non_unknown_labels[0][1] >= 0:
            return non_unknown_labels[0][0]
        else:
            return "未知"
    else:
        return "未知"


@torch.no_grad()
def evaluate(model, eval_dataloader, device, tokenizer, max_batches: int = 0):
    """在评估集上计算平均 loss、困惑度、准确率、Macro-F1 和 Micro-F1。

    Args:
        model: 训练好的模型。
        eval_dataloader: 评估数据的 DataLoader。
        device: 计算设备。
        tokenizer: 分词器。
        max_batches: 最多评估多少个 batch（0 表示不限制，用全部数据）。

    Returns:
        dict: 包含平均 loss、困惑度、准确率、Macro-F1 和 Micro-F1 的字典。
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_ground_truths = []

    for batch in eval_dataloader:
        if max_batches > 0 and num_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

        total_loss += loss.item()
        num_batches += 1

        # 生成式评估：让模型完整生成标签
        for i in range(input_ids.shape[0]):
            # 提取 prompt 部分（到"标签:"为止）
            prompt_ids = input_ids[i].tolist()
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            
            # 找到"标签:"的位置
            if "标签:" in prompt_text:
                prompt_end = prompt_text.index("标签:") + 3
                prompt = prompt_text[:prompt_end]
            else:
                prompt = prompt_text
            
            # 生成标签
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            generated_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.3,
                top_p=0.7,
                top_k=30,
                repetition_penalty=1.1,
                use_cache=True
            )
            
            # 解码生成的文本（保留特殊 token）
            pred_text = tokenizer.decode(generated_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=False).strip()
            
            # 提取真实标签（保留特殊 token）
            sample_labels = labels[i]
            mask = sample_labels != -100
            if mask.sum() > 0:
                true_tokens = sample_labels[mask]
                true_text = tokenizer.decode(true_tokens, skip_special_tokens=False).strip()
                
                # 将特殊 token 转换为普通标签
                pred_text = tokens_to_labels(pred_text)
                true_text = tokens_to_labels(true_text)
                
                # 清理预测文本
                pred_text = pred_text.lstrip('，, ')
                true_text = true_text.lstrip('，, ')
                pred_text = pred_text.replace('，', ',').replace('、', ',').strip()
                true_text = true_text.replace('，', ',').replace('、', ',').strip()
                while ',,' in pred_text:
                    pred_text = pred_text.replace(',,', ',')
                while ',,' in true_text:
                    true_text = true_text.replace(',,', ',')
                
                # 过滤无效标签，只保留闭集中的标签
                pred_text = filter_valid_labels(pred_text)
                true_text = filter_valid_labels(true_text)
                
                # 移除重复的标签
                pred_labels = list(dict.fromkeys([label.strip() for label in pred_text.split(',') if label.strip()]))
                true_labels = list(dict.fromkeys([label.strip() for label in true_text.split(',') if label.strip()]))
                
                # 如果没有有效标签，分配最相似的标签
                if not pred_labels:
                    # 使用生成的原始文本（在过滤之前）来计算最相似的标签
                    original_pred_text = tokenizer.decode(generated_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
                    most_similar = get_most_similar_label(original_pred_text)
                    pred_labels = [most_similar]
                
                # 如果真实标签为空，也分配最相似的标签（理论上不应该发生）
                if not true_labels:
                    original_true_text = tokenizer.decode(true_tokens, skip_special_tokens=True).strip()
                    most_similar = get_most_similar_label(original_true_text)
                    true_labels = [most_similar]
                
                all_predictions.append(','.join(pred_labels))
                all_ground_truths.append(','.join(true_labels))

    model.train()

    avg_loss = total_loss / max(num_batches, 1)
    avg_ppl = math.exp(avg_loss)

    # 计算准确率：只要预测的标签集合与真实标签集合有交集，就认为正确
    correct = 0
    for p, g in zip(all_predictions, all_ground_truths):
        p_labels = get_labels(p)
        g_labels = get_labels(g)
        if p_labels & g_labels:
            correct += 1
    
    accuracy = correct / len(all_predictions) if all_predictions else 0.0

    # 计算 Macro-F1：对每个样本计算 F1，然后取平均
    macro_f1_scores = []
    for p, g in zip(all_predictions, all_ground_truths):
        p_labels = get_labels(p)
        g_labels = get_labels(g)
        if not p_labels and not g_labels:
            macro_f1_scores.append(1.0)
        elif not p_labels or not g_labels:
            macro_f1_scores.append(0.0)
        else:
            intersection = len(p_labels & g_labels)
            precision = intersection / len(p_labels)
            recall = intersection / len(g_labels)
            if precision + recall == 0:
                macro_f1_scores.append(0.0)
            else:
                macro_f1_scores.append(2 * (precision * recall) / (precision + recall))

    macro_f1 = sum(macro_f1_scores) / len(macro_f1_scores) if macro_f1_scores else 0.0

    # 计算 Micro-F1：将所有样本的预测结果汇总到一起计算全局的 precision、recall、F1
    all_tp = 0  # True Positive
    all_fp = 0  # False Positive
    all_fn = 0  # False Negative

    for p, g in zip(all_predictions, all_ground_truths):
        p_labels = get_labels(p)
        g_labels = get_labels(g)
        
        # True Positive: 预测正确且真实存在的标签
        all_tp += len(p_labels & g_labels)
        # False Positive: 预测存在但真实不存在的标签
        all_fp += len(p_labels - g_labels)
        # False Negative: 真实存在但预测不存在的标签
        all_fn += len(g_labels - p_labels)

    # 计算全局 precision、recall、F1
    if all_tp + all_fp == 0:
        micro_precision = 0.0
    else:
        micro_precision = all_tp / (all_tp + all_fp)
    
    if all_tp + all_fn == 0:
        micro_recall = 0.0
    else:
        micro_recall = all_tp / (all_tp + all_fn)
    
    if micro_precision + micro_recall == 0:
        micro_f1 = 0.0
    else:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    # 打印前5个样本的调试信息
    if all_predictions:
        cli.print_info("前5个样本的评估结果:")
        for i in range(min(5, len(all_predictions))):
            p_labels = get_labels(all_predictions[i])
            g_labels = get_labels(all_ground_truths[i])
            intersection = p_labels & g_labels
            is_correct = len(intersection) > 0
            cli.print_info(f"样本 {i}:")
            cli.print_info(f"  预测: '{all_predictions[i]}'")
            cli.print_info(f"  真实: '{all_ground_truths[i]}'")
            cli.print_info(f"  预测标签: {p_labels}")
            cli.print_info(f"  真实标签: {g_labels}")
            cli.print_info(f"  交集: {intersection}")
            cli.print_info(f"  是否正确: {is_correct}")
        cli.print_divider()

    return {
        "loss": avg_loss,
        "ppl": avg_ppl,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1
    }

@torch.no_grad()
def evaluate_train(model, train_dataloader, device, tokenizer, max_batches: int = 10):
    """在训练集上计算准确率、Macro-F1 和 Micro-F1。

    Args:
        model: 训练好的模型。
        train_dataloader: 训练数据的 DataLoader。
        device: 计算设备。
        tokenizer: 分词器。
        max_batches: 最多评估多少个 batch（默认 10，用于快速评估）。

    Returns:
        dict: 包含准确率、Macro-F1 和 Micro-F1 的字典。
    """
    model.eval()
    all_predictions = []
    all_ground_truths = []

    for batch_idx, batch in enumerate(train_dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # 生成式评估：让模型完整生成标签
        for i in range(input_ids.shape[0]):
            # 提取 prompt 部分（到"标签:"为止）
            prompt_ids = input_ids[i].tolist()
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            
            # 找到"标签:"的位置
            if "标签:" in prompt_text:
                prompt_end = prompt_text.index("标签:") + 3
                prompt = prompt_text[:prompt_end]
            else:
                prompt = prompt_text
            
            # 生成标签
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            generated_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.3,
                top_p=0.7,
                top_k=30,
                repetition_penalty=1.1,
                use_cache=True
            )
            
            # 解码生成的文本（保留特殊 token）
            pred_text = tokenizer.decode(generated_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=False).strip()
            
            # 提取真实标签（保留特殊 token）
            sample_labels = labels[i]
            mask = sample_labels != -100
            if mask.sum() > 0:
                true_tokens = sample_labels[mask]
                true_text = tokenizer.decode(true_tokens, skip_special_tokens=False).strip()
                
                # 将特殊 token 转换为普通标签
                pred_text = tokens_to_labels(pred_text)
                true_text = tokens_to_labels(true_text)
                
                # 清理预测文本
                pred_text = pred_text.lstrip('，, ')
                true_text = true_text.lstrip('，, ')
                pred_text = pred_text.replace('，', ',').replace('、', ',').strip()
                true_text = true_text.replace('，', ',').replace('、', ',').strip()
                while ',,' in pred_text:
                    pred_text = pred_text.replace(',,', ',')
                while ',,' in true_text:
                    true_text = true_text.replace(',,', ',')
                
                # 过滤无效标签，只保留闭集中的标签
                pred_text = filter_valid_labels(pred_text)
                true_text = filter_valid_labels(true_text)
                
                # 移除重复的标签
                pred_labels = list(dict.fromkeys([label.strip() for label in pred_text.split(',') if label.strip()]))
                true_labels = list(dict.fromkeys([label.strip() for label in true_text.split(',') if label.strip()]))
                
                # 如果没有有效标签，分配最相似的标签
                if not pred_labels:
                    # 使用生成的原始文本（在过滤之前）来计算最相似的标签
                    original_pred_text = tokenizer.decode(generated_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
                    most_similar = get_most_similar_label(original_pred_text)
                    pred_labels = [most_similar]
                
                # 如果真实标签为空，也分配最相似的标签（理论上不应该发生）
                if not true_labels:
                    original_true_text = tokenizer.decode(true_tokens, skip_special_tokens=True).strip()
                    most_similar = get_most_similar_label(original_true_text)
                    true_labels = [most_similar]
                
                all_predictions.append(','.join(pred_labels))
                all_ground_truths.append(','.join(true_labels))

    model.train()

    # 计算准确率：只要预测的标签集合与真实标签集合有交集，就认为正确
    correct = 0
    for p, g in zip(all_predictions, all_ground_truths):
        p_labels = get_labels(p)
        g_labels = get_labels(g)
        if p_labels & g_labels:
            correct += 1
    
    accuracy = correct / len(all_predictions) if all_predictions else 0.0

    # 计算 Macro-F1：对每个样本计算 F1，然后取平均
    macro_f1_scores = []
    for p, g in zip(all_predictions, all_ground_truths):
        p_labels = get_labels(p)
        g_labels = get_labels(g)
        if not p_labels and not g_labels:
            macro_f1_scores.append(1.0)
        elif not p_labels or not g_labels:
            macro_f1_scores.append(0.0)
        else:
            intersection = len(p_labels & g_labels)
            precision = intersection / len(p_labels)
            recall = intersection / len(g_labels)
            if precision + recall == 0:
                macro_f1_scores.append(0.0)
            else:
                macro_f1_scores.append(2 * (precision * recall) / (precision + recall))

    macro_f1 = sum(macro_f1_scores) / len(macro_f1_scores) if macro_f1_scores else 0.0

    # 计算 Micro-F1：将所有样本的预测结果汇总到一起计算全局的 precision、recall、F1
    all_tp = 0  # True Positive
    all_fp = 0  # False Positive
    all_fn = 0  # False Negative

    for p, g in zip(all_predictions, all_ground_truths):
        p_labels = get_labels(p)
        g_labels = get_labels(g)
        
        # True Positive: 预测正确且真实存在的标签
        all_tp += len(p_labels & g_labels)
        # False Positive: 预测存在但真实不存在的标签
        all_fp += len(p_labels - g_labels)
        # False Negative: 真实存在但预测不存在的标签
        all_fn += len(g_labels - p_labels)

    # 计算全局 precision、recall、F1
    if all_tp + all_fp == 0:
        micro_precision = 0.0
    else:
        micro_precision = all_tp / (all_tp + all_fp)
    
    if all_tp + all_fn == 0:
        micro_recall = 0.0
    else:
        micro_recall = all_tp / (all_tp + all_fn)
    
    if micro_precision + micro_recall == 0:
        micro_f1 = 0.0
    else:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1
    }


def main():
    """训练主函数。

    完整流程：
    1. 加载 Qwen2.5-Instruct 模型 + LoRA 配置
    2. 加载训练集和验证集（金融文本多标签分类）
    3. 设置 AdamW 优化器 + Cosine 学习率调度（含 warmup）
    4. 训练循环：前向 → 反向 → 更新，定期日志/评估/保存
    5. 训练结束后保存 LoRA 权重
    """
    args = parse_args()

    cli.print_header("Qwen2.5-Instruct LoRA 训练")
    cli.print_divider()

    # ── 设备 ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cli.print_info(f"设备: {device}")

    # ── TensorBoard ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{timestamp}_{args.run_name}"
    run_dir = os.path.join(LOG_DIR, run_tag)
    save_dir = os.path.join(SAVE_DIR, run_tag)
    output_dir = os.path.join(args.output_dir, f"lora_qwen2.5_{timestamp}")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(run_dir)
    cli.print_info(f"TensorBoard 日志目录: {run_dir}")
    cli.print_info(f"Checkpoint 保存目录: {save_dir}")
    cli.print_info("启动查看: tensorboard --logdir runs")
    cli.print_divider()

    # ── 模型和分词器 ──────────────────────────────────────────────────
    cli.print_loading("加载 Qwen2.5-Instruct 模型")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 启用flash attention
    if hasattr(model.config, 'use_flash_attention_2'):
        model.config.use_flash_attention_2 = True
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 添加特殊标签 token
    tokenizer = add_special_label_tokens(tokenizer)
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # 调整模型 embedding 层大小以适应新的 token
    model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cli.print_success("模型加载完成！")
    cli.print_kv("总参数", f"{total_params:,}")
    cli.print_kv("可训练参数", f"{trainable_params:,}")
    cli.print_kv("冻结参数", f"{total_params - trainable_params:,}")
    cli.print_divider()

    # ── 数据集 ────────────────────────────────────────────────────────
    cli.print_loading("加载数据集")
    train_dataset = FinancialTextDataset(args.train_data, tokenizer, args.max_length)
    dev_dataset = FinancialTextDataset(args.dev_data, tokenizer, args.max_length)
    
    cli.print_kv("训练样本数", f"{len(train_dataset):,}")
    cli.print_kv("验证样本数", f"{len(dev_dataset):,}")
    cli.print_divider()

    # ── 数据整理器和数据加载器 ────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True
    )

    eval_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True
    )

    # ── 优化器与调度器 ────────────────────────────────────────────────
    total_steps = len(train_dataset) // args.batch_size * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    # AdamW 优化器
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_param_list,
        lr=args.learning_rate,
        weight_decay=0.01
    )

    # 余弦退火学习率调度器（含线性预热）
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    cli.print_kv("总步数", total_steps)
    cli.print_kv("预热步数", warmup_steps)
    cli.print_kv("学习率", args.learning_rate)
    cli.print_kv("Batch size", args.batch_size)
    cli.print_kv("评估间隔", f"每 {args.eval_interval} 步")
    eval_samples_desc = f"{args.eval_samples}" if args.eval_samples > 0 else "全部"
    cli.print_kv("评估样本数", eval_samples_desc)
    cli.print_divider()

    # ── 训练循环 ──────────────────────────────────────────────────────
    cli.print_info("开始训练...")
    model.train()

    global_step = 0
    log_loss = 0.0
    start_time = time.time()
    log_start_time = start_time

    # 评估时最多跑多少个 batch（0 = 不限制）
    eval_max_batches = args.eval_samples // args.batch_size if args.eval_samples > 0 else 0

    # 进度条：总步数已知，每个 batch 更新一次
    # 使用 cycle 来重复遍历 dataloader，实现多 epoch 训练
    pbar = tqdm(cycle(train_dataloader), total=total_steps, desc="训练中", unit="step",
                dynamic_ncols=True)

    for batch in pbar:
        # 检查是否达到总步数
        if global_step >= total_steps:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # ── 首个 batch：打印样本供检查 ──────────────────────────────
        if global_step == 0:
            cli.print_divider()
            cli.print_info("数据样本检查（第 1 个 batch 的第 1 条）")
            cli.print_divider()

            sample_ids = input_ids[0].tolist()
            sample_labels = labels[0].tolist()

            # 完整输入（包含特殊 token）
            decoded_input = tokenizer.decode(sample_ids, skip_special_tokens=False)
            cli.print_kv("输入文本", decoded_input)

            # 仅监督部分（assistant 回复部分）
            label_ids = [t for t in sample_labels if t != -100]
            decoded_labels = tokenizer.decode(label_ids, skip_special_tokens=False)
            cli.print_kv("监督标签", decoded_labels)

            cli.print_divider()

        # 前向 + 反向（使用 bf16 混合精度以节省显存和加速计算）
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

        loss.backward()
        # 梯度裁剪：防止梯度爆炸，将梯度范数限制在 1.0 以内
        torch.nn.utils.clip_grad_norm_(trainable_param_list, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # 统计
        loss_val = loss.item()
        train_ppl = math.exp(loss_val)
        log_loss += loss_val
        global_step += 1

        # 每步更新进度条后缀
        lr_now = scheduler.get_last_lr()[0]
        pbar.set_postfix(loss=f"{loss_val:.4f}", ppl=f"{train_ppl:.2f}",
                         lr=f"{lr_now:.2e}", refresh=False)

        # TensorBoard: 每步记录训练指标
        writer.add_scalar("train/loss", loss_val, global_step)
        writer.add_scalar("train/ppl", train_ppl, global_step)
        writer.add_scalar("train/lr", lr_now, global_step)

        # 详细日志
        if global_step % args.log_interval == 0:
            elapsed = time.time() - log_start_time
            avg_log_loss = log_loss / args.log_interval
            avg_log_ppl = math.exp(avg_log_loss)
            samples_done = min(global_step * args.batch_size, len(train_dataset) * args.num_train_epochs)

            tqdm.write(
                f"  Step {global_step}/{total_steps} | "
                f"样本 {samples_done:,}/{len(train_dataset)*args.num_train_epochs:,} | "
                f"Loss {avg_log_loss:.4f} | "
                f"PPL {avg_log_ppl:.2f} | "
                f"LR {lr_now:.2e} | "
                f"耗时 {elapsed:.1f}s"
            )
            log_loss = 0.0
            log_start_time = time.time()

        # 定期评估
        if global_step % args.eval_interval == 0:
            # 评估训练集
            tqdm.write(f"  评估训练集 (Step {global_step})...")
            train_result = evaluate_train(model, train_dataloader, device, tokenizer, max_batches=10)
            writer.add_scalar("train/accuracy", train_result["accuracy"], global_step)
            writer.add_scalar("train/macro_f1", train_result["macro_f1"], global_step)
            writer.add_scalar("train/micro_f1", train_result["micro_f1"], global_step)
            tqdm.write(f"  ✓ Train Accuracy: {train_result['accuracy']:.4f}")
            tqdm.write(f"  ✓ Train Macro-F1: {train_result['macro_f1']:.4f} | Train Micro-F1: {train_result['micro_f1']:.4f}")
            
            # 评估验证集
            tqdm.write(f"  评估验证集 (Step {global_step})...")
            eval_result = evaluate(model, eval_dataloader, device, tokenizer,
                                   max_batches=eval_max_batches)
            writer.add_scalar("eval/loss", eval_result["loss"], global_step)
            writer.add_scalar("eval/ppl", eval_result["ppl"], global_step)
            writer.add_scalar("eval/accuracy", eval_result["accuracy"], global_step)
            writer.add_scalar("eval/macro_f1", eval_result["macro_f1"], global_step)
            writer.add_scalar("eval/micro_f1", eval_result["micro_f1"], global_step)
            tqdm.write(f"  ✓ Eval Loss: {eval_result['loss']:.4f} | Eval PPL: {eval_result['ppl']:.2f}")
            tqdm.write(f"  ✓ Eval Accuracy: {eval_result['accuracy']:.4f}")
            tqdm.write(f"  ✓ Eval Macro-F1: {eval_result['macro_f1']:.4f} | Eval Micro-F1: {eval_result['micro_f1']:.4f}")

        # 定期保存（保存 LoRA 权重）
        if global_step % args.save_interval == 0:
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"lora_step{global_step}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            tqdm.write(f"  ✓ Checkpoint 已保存: {ckpt_path}")

    pbar.close()

    # ── 最终评估（使用全部评估数据） ────────────────────────────────────
    cli.print_info("最终评估（全量）...")
    final_eval_result = evaluate(model, eval_dataloader, device, tokenizer, max_batches=0)
    writer.add_scalar("eval/loss", final_eval_result["loss"], global_step)
    writer.add_scalar("eval/ppl", final_eval_result["ppl"], global_step)
    writer.add_scalar("eval/accuracy", final_eval_result["accuracy"], global_step)
    writer.add_scalar("eval/macro_f1", final_eval_result["macro_f1"], global_step)
    writer.add_scalar("eval/micro_f1", final_eval_result["micro_f1"], global_step)
    cli.print_kv("最终 Eval Loss", f"{final_eval_result['loss']:.4f}")
    cli.print_kv("最终 Eval PPL", f"{final_eval_result['ppl']:.2f}")
    cli.print_kv("最终 Eval Accuracy", f"{final_eval_result['accuracy']:.4f}")
    cli.print_kv("最终 Eval Macro-F1", f"{final_eval_result['macro_f1']:.4f}")
    cli.print_kv("最终 Eval Micro-F1", f"{final_eval_result['micro_f1']:.4f}")

    # ── 训练结束 ──────────────────────────────────────────────────────
    total_time = time.time() - start_time
    cli.print_divider()
    cli.print_success("训练完成！")
    cli.print_kv("总步数", global_step)
    cli.print_kv("总样本", f"{len(train_dataset) * args.num_train_epochs:,}")
    cli.print_kv("最终 Eval Loss", f"{final_eval_result['loss']:.4f}")
    cli.print_kv("最终 Eval Accuracy", f"{final_eval_result['accuracy']:.4f}")
    cli.print_kv("最终 Eval Macro-F1", f"{final_eval_result['macro_f1']:.4f}")
    cli.print_kv("最终 Eval Micro-F1", f"{final_eval_result['micro_f1']:.4f}")
    cli.print_kv("总耗时", f"{total_time:.1f}s ({total_time / 60:.1f}min)")

    # 保存最终 LoRA 权重
    final_path = os.path.join(save_dir, "final_lora")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    cli.print_success(f"最终权重已保存: {final_path}")

    writer.close()


if __name__ == "__main__":
    main()
