---
base_model:
- ""
frameworks:
- ""
---
# LLM 金融文本分类项目

使用 LoRA 对 Qwen2.5-Instruct 模型进行指令微调，实现金融文本的多标签分类任务。

## 项目架构

```
训练流程：
  金融文本 → 改进的 Prompt → Qwen2.5-Instruct + LoRA → 多标签分类

推理流程：
  金融文本 → 模型生成 → 标签过滤与处理 → 最终分类结果
```

## 目录结构

```
LLM/
├── README.md                  # 项目说明文档
├── train_lora.py              # 主要训练脚本
├── eval_chat.py              # 交互式评估脚本
├── requirements.txt           # 依赖项列表
│
├── script/                    # 辅助脚本
│   ├── improve_prompt.py              # 改进 Prompt 设计
│   ├── enhance_label_supervision.py   # 增强标签监督
│   ├── process_enhanced_labels.py     # 处理增强后的标签
│   ├── run_all_improvements.py       # 运行所有改进步骤
│   ├── improve_multilabel_prompt.py   # 改进多标签 Prompt
│   ├── analyze_multilabel.py          # 分析多标签分布
│   └── analyze_dataset.py             # 分析数据集分布
│
├── llm_dataset_processed/     # 处理后的数据集
│   ├── train_final.json               # 原始训练集
│   ├── dev_final.json                 # 原始验证集
│   ├── train_improved_prompt.json     # 改进 Prompt 后的训练集
│   ├── dev_improved_prompt.json       # 改进 Prompt 后的验证集
│   ├── train_enhanced_labels.json     # 增强标签监督后的训练集
│   ├── dev_enhanced_labels.json       # 增强标签监督后的验证集
│   ├── train_final_processed.json     # 最终处理后的训练集
│   ├── dev_final_processed.json       # 最终处理后的验证集
│   ├── train_multilabel_prompt.json   # 多标签 Prompt 训练集
│   └── dev_multilabel_prompt.json     # 多标签 Prompt 验证集
│
├── checkpoints/               # 训练保存的模型权重（自动生成）
├── runs/                      # TensorBoard 日志（自动生成）
├── utils/                     # 工具包
│   ├── __init__.py
│   └── cli.py                 # 终端彩色输出工具函数
└── Qwen2.5-3B-Instruct/       # Qwen2.5 基础模型（需自行下载）
```

## 环境配置

### 1. 安装依赖

```bash
# 激活 conda 环境
conda activate base

# 更新 pip
pip install --upgrade pip

# 安装依赖
pip install 
    transformers==4.53.0 
    accelerate==1.1.0 
    peft==0.18.0 
    bitsandbytes==0.45.0 
    sentencepiece 
    protobuf 
    einops 
    tensorboardX 
    tqdm

# 进入项目目录
cd /root/autodl-tmp/LLM
```

### 2. 下载模型权重

```bash
# 下载 Qwen2.5-3B-Instruct 模型
# 可以从 Hugging Face 或 ModelScope 下载
# 确保模型文件夹名称为 Qwen2.5-3B-Instruct
```

### 3. 准备数据集

项目使用金融文本多标签分类数据集，包含以下标签：
- 国际、经济活动、市场、金属、政策、中立、积极、煤炭、农业、能源、畜牧、政治、消极、未知

数据集格式为 JSON 行，每条数据包含：
- messages: 包含 system、user、assistant 消息
- user: 金融文本内容和分类指令
- assistant: 标签列表（逗号分隔）

## 核心模块

### `train_lora.py` — LoRA 训练脚本

#### 功能概述
- 使用 LoRA 对 Qwen2.5-Instruct 模型进行指令微调
- 支持金融文本多标签分类任务
- 实现训练集和验证集的实时监控
- 支持特殊标签 token 的添加和处理

#### 关键组件

1. **FinancialTextDataset**：金融文本数据集类
   - 加载和处理 JSON 格式的训练数据
   - 构建训练样本：用户输入 → assistant 回复
   - 支持改进的多标签 Prompt 格式

2. **特殊标签 token 处理**
   - 为每个标签创建特殊 token（如 `<LABEL_国际>`）
   - 在训练前添加到 tokenizer
   - 动态调整模型嵌入层大小

3. **训练循环**
   - AdamW 优化器 + Cosine 学习率调度（含 warmup）
   - 梯度裁剪防止梯度爆炸
   - 定期评估和 checkpoint 保存
   - TensorBoard 日志记录

4. **评估方法**
   - 生成式评估：让模型完整生成标签
   - 准确率计算：预测标签集合与真实标签集合的交集
   - Macro-F1 和 Micro-F1 指标
   - 处理空标签情况：分配最相似的标签

#### 提示词设计

项目使用了改进的多标签分类提示词，包含以下特点：

1. **详细的标签说明**：为每个标签提供清晰的定义和示例
2. **多标签强调**：明确鼓励模型输出所有相关标签
3. **上下文理解**：强调基于文本整体含义进行判断
4. **格式要求**：明确要求使用英文逗号分隔多个标签

示例提示词：
```
请仔细分析以下金融文本内容，理解其核心含义和上下文，然后判断其所属的所有相关标签类别。

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

文本内容: [金融文本内容]

标签:
```

#### 评估指标

1. **准确率（Accuracy）**：预测标签集合与真实标签集合有交集即认为正确
2. **Macro-F1**：对每个样本计算 F1 分数，然后取平均值
3. **Micro-F1**：将所有样本的预测结果汇总，计算全局的 precision、recall 和 F1
4. **Loss**：模型生成的交叉熵损失
5. **PPL（困惑度）**：模型对文本的困惑度

#### 监督训练流程

1. **数据预处理**：
   - 加载 JSON 格式的训练数据
   - 提取用户输入和 assistant 回复
   - 构建改进的多标签 Prompt

2. **标签处理**：
   - 将标签转换为特殊 token
   - 构建训练样本：Prompt + 特殊 token
   - 计算 prompt 长度，构建 labels（prompt 部分为 -100）

3. **模型训练**：
   - 冻结 Qwen2.5 基础模型
   - 训练 LoRA 适配器（约 3.7M 参数）
   - 余弦退火学习率调度
   - 定期评估和 checkpoint 保存

4. **模型评估**：
   - 生成式评估：让模型完整生成标签
   - 过滤和处理生成的标签
   - 计算评估指标
   - 记录 TensorBoard 日志

### `eval_chat.py` — 交互式评估脚本

- 加载训练好的 LoRA 权重
- 支持命令行输入金融文本进行分类
- 实时显示分类结果
- 支持多轮对话和清空历史

## 辅助脚本

### `script/improve_prompt.py` — 改进 Prompt 设计
- 为数据集添加详细的标签说明
- 强调多标签分类的重要性
- 明确输出格式要求

### `script/enhance_label_supervision.py` — 增强标签监督
- 为每个标签添加理由
- 帮助模型理解标签的适用性
- 基于文本内容生成合理的标签理由

### `script/process_enhanced_labels.py` — 处理增强后的标签
- 从增强的标签中提取纯标签
- 确保训练时使用正确的标签格式

### `script/improve_multilabel_prompt.py` — 改进多标签 Prompt
- 进一步优化多标签分类的 Prompt
- 明确鼓励模型输出多个相关标签
- 添加更多多标签分类的示例

### `script/analyze_multilabel.py` — 分析多标签分布
- 统计数据集中多标签样本的比例
- 分析标签分布情况
- 帮助理解数据特点

### `script/analyze_dataset.py` — 分析数据集分布
- 统计数据集中各标签的出现次数
- 分析未知标签的比例
- 为模型调优提供参考

## 训练和评估

### 1. 准备数据集

```bash
# 运行所有改进步骤
cd script
python run_all_improvements.py
```

### 2. 开始训练

```bash
# 基本训练
python train_lora.py

# 自定义参数
python train_lora.py --batch_size 8 --learning_rate 1e-5 --num_train_epochs 5
```

### 3. 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir runs

# 在浏览器中访问 http://localhost:6006
```

### 4. 交互式评估

```bash
# 加载训练好的模型
python eval_chat.py --lora_model ./checkpoints/[run_name]/final_lora

# 输入金融文本进行分类
# 输入 clear 清空对话历史
# 输入 exit/quit 退出
```

## 模型改进

项目实施了以下改进措施来提高模型性能：

1. **改进 Prompt 设计**：
   - 详细的标签说明
   - 多标签分类强调
   - 上下文理解指导

2. **增强标签监督**：
   - 为每个标签添加理由
   - 帮助模型理解标签适用性
   - 提高标签学习的准确性

3. **优化生成参数**：
   - 增加 max_new_tokens 到 200
   - 调整 temperature 和 top_p
   - 鼓励模型生成多个标签

4. **特殊标签 token**：
   - 为每个标签创建特殊 token
   - 提高标签识别的准确性
   - 减少标签混淆

5. **多标签处理**：
   - 处理空标签情况
   - 分配最相似的标签
   - 确保模型总是输出有效标签

## 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 准确率 | ~45% | 预测标签与真实标签有交集即认为正确 |
| Macro-F1 | ~33% | 每个样本 F1 分数的平均值 |
| Micro-F1 | ~31% | 全局 F1 分数 |
| Loss | ~0.44 | 模型生成的交叉熵损失 |
| PPL | ~1.55 | 模型对文本的困惑度 |

## 总结

本项目实现了基于 Qwen2.5-Instruct 和 LoRA 的金融文本多标签分类系统，通过改进 Prompt 设计、增强标签监督、优化生成参数等措施，提高了模型的分类性能。项目支持实时监控训练过程，提供了完整的评估指标，为金融文本分类任务提供了一个有效的解决方案。

通过持续的模型调优和数据改进，该系统有望在金融文本分类任务中取得更好的性能。


