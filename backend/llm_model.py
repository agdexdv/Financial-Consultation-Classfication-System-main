"""
Qwen2.5-Instruct + LoRA 推理模块
用于金融文本多标签分类任务
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 检测是否支持 8-bit 量化
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

# 模型路径配置
BASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "LLM", "Qwen2.5-3B-Instruct")
LORA_MODEL_PATH = os.path.join(os.path.dirname(__file__), "LLM", "checkpoints", "20260225_093417_lora_train", "final_lora")

# 标签列表
LABELS = [
    "国际", "经济活动", "市场", "金属", "政策",
    "中立", "积极", "煤炭", "农业", "能源",
    "畜牧", "政治", "消极", "未知"
]

# 分类提示词模板
CLASSIFICATION_PROMPT = """请仔细分析以下金融文本内容，理解其核心含义和上下文，然后判断其所属的所有相关标签类别。

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

文本内容: {text}

标签:"""


class FinancialLLMClassifier:
    """金融文本 Qwen2.5 + LoRA 分类器"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.load_error = None
        self._load_model()

    def _load_model(self):
        """加载模型和 tokenizer"""
        # 当前默认关闭 Qwen2.5 + LoRA 在线推理：
        # 1. 稳定运行需要较高 GPU 显存
        # 2. 仓库默认保留规则基线，便于接口联调与轻量演示
        # 3. 资源条件满足后，可在此处启用 Qwen2.5 + LoRA 推理
        self.load_error = (
            "qwen_lora_runtime_disabled: 当前默认关闭 Qwen2.5-Instruct + LoRA 在线推理（需要 12GB+ GPU 内存）。\n"
            f"  基础模型: {BASE_MODEL_PATH}\n"
            f"  LoRA 权重: {LORA_MODEL_PATH}\n"
            "  当前接口默认使用规则基线返回结果；Qwen2.5 + LoRA 仍是项目既定模型路线。"
        )
        self.model = None
        self.tokenizer = None
        self.device = None
        return

        # 注释：如需启用 Qwen2.5 + LoRA 推理，请确保：
        # 1. GPU 内存 >= 12GB
        # 2. 停止其他占用 GPU 的进程
        # 3. 删除上面的 return 语句

    def _parse_labels(self, generated_text: str) -> list:
        """解析模型生成的标签"""
        # 获取生成文本的最后一行作为标签输出
        lines = generated_text.strip().split('\n')
        label_line = lines[-1].strip() if lines else ""

        # 移除可能的 "标签:" 前缀
        if "标签:" in label_line:
            label_line = label_line.split("标签:")[-1].strip()

        # 按逗号分割标签
        predicted_labels = [label.strip() for label in label_line.split(",") if label.strip()]

        # 过滤有效标签
        valid_labels = [label for label in predicted_labels if label in LABELS]

        # 如果没有有效标签，返回空列表
        return valid_labels if valid_labels else []

    def predict(self, text: str) -> dict:
        """
        对文本进行分类预测

        Args:
            text: 待分类的金融文本

        Returns:
            dict: 包含分类结果的字典
                - labels: 预测的标签列表
                - product_label: 商品类相关标签概率（简化版）
                - sentiment_label: 情感标签概率（简化版）
                - keywords: 关键词（从文本中提取）
                - debug: 调试信息
        """
        if not text or not text.strip():
            return {
                "labels": [],
                "product_label": {},
                "sentiment_label": {},
                "keywords": [],
                "debug": {"error": "empty_text", "model_mode": "llm"}
            }

        if self.model is None:
            return {
                "labels": [],
                "product_label": {},
                "sentiment_label": {},
                "keywords": [],
                "debug": {"error": self.load_error or "model_not_loaded", "model_mode": "llm"}
            }

        try:
            # 构建提示词
            prompt = CLASSIFICATION_PROMPT.format(text=text.strip())

            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)

            # 生成预测
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # 解码生成结果
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 解析标签
            predicted_labels = self._parse_labels(generated_text)

            # 构建商品类标签概率（简化处理）
            commodity_labels = ["金属", "煤炭", "农业", "能源", "畜牧"]
            product_probs = {}
            for label in commodity_labels:
                product_probs[label] = 1.0 if label in predicted_labels else 0.0

            # 构建情感标签概率（简化处理）
            sentiment_probs = {
                "积极": 1.0 if "积极" in predicted_labels else 0.0,
                "中性": 1.0 if "中立" in predicted_labels else 0.0,
                "负向": 1.0 if "消极" in predicted_labels else 0.0
            }
            # 如果没有情感标签，默认中性
            if sum(sentiment_probs.values()) == 0:
                sentiment_probs["中性"] = 1.0

            # 提取关键词（使用预测的标签）
            keywords = predicted_labels[:8]

            return {
                "labels": predicted_labels,
                "product_label": product_probs,
                "sentiment_label": sentiment_probs,
                "keywords": keywords,
                "debug": {
                    "model_mode": "llm",
                    "model_loaded": True,
                    "text_length": len(text),
                    "generated_text": generated_text[-200:]  # 保留最后200字符用于调试
                }
            }

        except Exception as e:
            return {
                "labels": [],
                "product_label": {},
                "sentiment_label": {},
                "keywords": [],
                "debug": {"error": str(e), "model_mode": "llm"}
            }


# 全局分类器实例
_classifier = None


def get_classifier():
    """获取分类器单例"""
    global _classifier
    if _classifier is None:
        _classifier = FinancialLLMClassifier()
    return _classifier


if __name__ == "__main__":
    # 测试代码
    classifier = get_classifier()
    if classifier.model is None:
        print(f"模型加载失败: {classifier.load_error}")
    else:
        print("模型加载成功!")

        # 测试预测
        test_text = "受美联储加息影响，今日国际金价下跌2%，原油价格也出现小幅回落。"
        result = classifier.predict(test_text)

        print(f"\n测试文本: {test_text}")
        print(f"预测标签: {result['labels']}")
        print(f"商品分类: {result['product_label']}")
        print(f"情感分析: {result['sentiment_label']}")
