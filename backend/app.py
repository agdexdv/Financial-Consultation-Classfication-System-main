import importlib.util
import io
import os
import re
from typing import Dict, List, Optional, Tuple
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# Word 文档解析依赖（可选）
try:
    import docx  # python-docx
except Exception:  # pragma: no cover
    docx = None
# PDF 解析依赖（可选）
try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None
# Excel 解析依赖（可选）
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None
# 历史兼容的 HuggingFace 分类模型依赖（可选，当前非主路线）
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    import torch  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    torch = None

# LLM 模型依赖（可选）
try:
    from llm_model import get_classifier
    LLM_AVAILABLE = True
except Exception:  # pragma: no cover
    LLM_AVAILABLE = False
# -----------------------------
# 应用初始化
# -----------------------------
app = FastAPI(title="Finance Classifier API")
# 允许本地开发：前端(5173) -> 后端(8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------
# 规则基线关键词
# -----------------------------
COMMODITY_KEYWORDS = [
    "大宗商品",
    "黄金",
    "原油",
    "铜",
    "铁矿",
    "煤炭",
    "玉米",
    "大豆",
    "棉花",
    "螺纹钢",
    "现货",
    "期货",
    "库存",
    "价格",
    "报价",
    "品种",
    "供需",
    "仓单",
    "升贴水",
    "进口",
    "出口",
    "炼厂",
    "指数",
    "commodity",
    "commodities",
    "gold",
    "silver",
    "oil",
    "crude oil",
    "petroleum",
    "natural gas",
    "gas",
    "copper",
    "aluminum",
    "aluminium",
    "zinc",
    "nickel",
    "iron ore",
    "steel",
    "rebar",
    "coal",
    "coke",
    "corn",
    "soybean",
    "soybeans",
    "wheat",
    "cotton",
    "hog",
    "pork",
    "spot",
    "futures",
    "inventory",
    "stockpile",
    "price",
    "prices",
    "quotation",
    "supply",
    "demand",
    "refinery",
]
FINANCE_KEYWORDS = [
    "金融",
    "咨询",
    "研报",
    "宏观",
    "通胀",
    "利率",
    "美元",
    "汇率",
    "信用",
    "风险",
    "债券",
    "股票",
    "基金",
    "期权",
    "期货",
    "指数",
    "供给",
    "需求",
    "库存",
    "价格",
    "成本",
    "finance",
    "financial",
    "research",
    "report",
    "macro",
    "inflation",
    "interest rate",
    "rates",
    "dollar",
    "usd",
    "exchange rate",
    "fx",
    "credit",
    "risk",
    "bond",
    "bonds",
    "stock",
    "stocks",
    "equity",
    "fund",
    "funds",
    "option",
    "options",
    "futures",
    "index",
    "cost",
    "market",
    "markets",
    "policy",
]
POSITIVE_WORDS = [
    "上涨",
    "走强",
    "利好",
    "向好",
    "正面",
    "积极",
    "良好",
    "更好",
    "最好",
    "强劲",
    "利多",
    "增产",
    "盈利",
    "改善",
    "回升",
    "突破",
    "需求上升",
    "好消息",
    "rise",
    "rises",
    "rising",
    "gain",
    "gains",
    "good",
    "positive",
    "better",
    "best",
    "strong",
    "higher",
    "stronger",
    "bullish",
    "improve",
    "improves",
    "improved",
    "increase",
    "increases",
    "rebound",
    "rebounds",
    "recovery",
    "good news",
]
NEGATIVE_WORDS = [
    "下跌",
    "走弱",
    "利空",
    "不利",
    "不好",
    "糟糕",
    "更差",
    "最差",
    "负面",
    "疲弱",
    "担忧",
    "担心",
    "忧虑",
    "减产",
    "亏损",
    "恶化",
    "回落",
    "跌破",
    "需求下降",
    "坏消息",
    "fall",
    "falls",
    "falling",
    "drop",
    "drops",
    "decline",
    "declines",
    "bad",
    "worse",
    "worst",
    "negative",
    "weak",
    "concern",
    "concerns",
    "lower",
    "weaker",
    "bearish",
    "decrease",
    "decreases",
    "loss",
    "losses",
    "uncertainty",
    "slump",
    "bad news",
]
MULTILABEL_KEYWORDS = {
    "国际": [
        "国际",
        "全球",
        "海外",
        "美国",
        "欧洲",
        "欧盟",
        "日本",
        "美联储",
        "美元",
        "global",
        "overseas",
        "federal reserve",
        "fed",
        "europe",
        "european union",
        "japan",
        "dollar",
    ],
    "经济活动": [
        "经济",
        "GDP",
        "就业",
        "消费",
        "制造业",
        "工业",
        "投资",
        "通胀",
        "PMI",
        "economy",
        "gdp",
        "employment",
        "consumption",
        "manufacturing",
        "industrial",
        "investment",
        "inflation",
        "pmi",
    ],
    "市场": [
        "市场",
        "价格",
        "报价",
        "期货",
        "现货",
        "指数",
        "交易",
        "收盘",
        "行情",
        "market",
        "markets",
        "price",
        "prices",
        "quotation",
        "futures",
        "spot",
        "index",
        "trading",
    ],
    "金属": [
        "金属",
        "黄金",
        "白银",
        "铜",
        "铝",
        "锌",
        "镍",
        "钢",
        "螺纹钢",
        "铁矿",
        "metal",
        "metals",
        "gold",
        "silver",
        "copper",
        "aluminum",
        "aluminium",
        "zinc",
        "nickel",
        "steel",
        "iron ore",
    ],
    "政策": [
        "政策",
        "监管",
        "央行",
        "财政",
        "会议",
        "公告",
        "法案",
        "关税",
        "降准",
        "加息",
        "policy",
        "regulation",
        "central bank",
        "fiscal",
        "tariff",
        "rate hike",
        "rate cut",
    ],
    "煤炭": ["煤炭", "动力煤", "焦煤", "焦炭", "喷吹煤", "coal", "coking coal", "coke"],
    "农业": [
        "农业",
        "农产品",
        "玉米",
        "大豆",
        "小麦",
        "棉花",
        "豆粕",
        "油脂",
        "agriculture",
        "farm",
        "corn",
        "soybean",
        "soybeans",
        "wheat",
        "cotton",
        "meal",
        "vegetable oil",
    ],
    "能源": [
        "能源",
        "原油",
        "石油",
        "天然气",
        "电力",
        "燃料",
        "新能源",
        "光伏",
        "风电",
        "energy",
        "oil",
        "crude oil",
        "petroleum",
        "natural gas",
        "electricity",
        "fuel",
        "solar",
        "wind power",
    ],
    "畜牧": [
        "畜牧",
        "生猪",
        "猪肉",
        "鸡蛋",
        "饲料",
        "养殖",
        "白羽鸡",
        "仔猪",
        "livestock",
        "hog",
        "hogs",
        "pork",
        "egg",
        "eggs",
        "feed",
        "poultry",
    ],
    "政治": [
        "政治",
        "政府",
        "总统",
        "大选",
        "外交",
        "冲突",
        "战争",
        "议会",
        "制裁",
        "politics",
        "government",
        "president",
        "election",
        "diplomacy",
        "conflict",
        "war",
        "parliament",
        "sanction",
        "sanctions",
    ],
}
SENTIMENT_TO_MULTILABEL = {"正向": "积极", "中性": "中立", "负向": "消极"}
ENGLISH_FALLBACK_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "over",
    "under",
    "after",
    "before",
    "while",
    "amid",
    "across",
    "about",
    "their",
    "there",
    "would",
    "could",
    "should",
    "global",
}
# -----------------------------
# 模型封装（支持扩展）
# -----------------------------
class ModelWrapper:
    """
    模型加载顺序：
    1) 自定义 Python 模型文件（MODEL_PATH 指向 .py）
    2) 历史兼容的 HuggingFace 分类模型目录（MODEL_PATH 指向模型目录）
    3) 规则基线兜底
    适配“模型文件 + 权重文件”的交付方式。
    """
    def __init__(self, model_path: Optional[str], weights_path: Optional[str]):
        self.model_path = model_path
        self.weights_path = weights_path
        self.available = False
        self.load_error: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self._load_model()
    def _load_model(self) -> None:
        if not self.model_path:
            return
        if not os.path.exists(self.model_path):
            self.load_error = f"model_path_not_found: {self.model_path}"
            return
        # 1) 自定义 Python 文件：load_model(weights_path) 或 Model(weights_path)
        ext = os.path.splitext(self.model_path)[1].lower()
        if ext == ".py":
            try:
                spec = importlib.util.spec_from_file_location("custom_model", self.model_path)
                if spec is None or spec.loader is None:
                    self.load_error = "import_failed: invalid spec"
                    return
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "load_model"):
                    self.model = module.load_model(self.weights_path)
                elif hasattr(module, "Model"):
                    self.model = module.Model(self.weights_path)
                else:
                    self.load_error = "custom_model_missing: load_model() or Model"
                    return
                self.available = True
                return
            except Exception as exc:
                self.load_error = f"custom_model_error: {exc}"
                return
        # 2) 历史兼容的 HuggingFace 分类模型目录，需安装 transformers
        #    当前项目主路线为 Qwen2.5 + LoRA；如需兼容旧模型，可在这里加载
        if os.path.isdir(self.model_path) and AutoModelForSequenceClassification is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.available = True
                return
            except Exception as exc:
                self.load_error = f"hf_load_error: {exc}"
                return
        self.load_error = "unsupported_model_file: use .py or HF directory"
    def _predict_hf(self, text: str) -> Dict[str, float]:
        # HuggingFace 预测（此处仅返回 raw 概率，映射逻辑需按任务自定义）
        # 扩展点：在这里把 logits/probs 转成 “商品类/情感” 对应的概率
        assert self.model is not None and self.tokenizer is not None and torch is not None
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1).tolist()
        return {"probs": probs}
    def predict(self, text: str) -> Tuple[Dict[str, float], Dict[str, float], List[str], Dict, List[str]]:
        # 优先级：1) LLM 模型 2) 自定义模型 3) HuggingFace 模型 4) 规则基线

        # 1) 尝试使用 LLM 模型
        if LLM_AVAILABLE:
            try:
                llm_classifier = get_classifier()
                if llm_classifier.model is not None:
                    llm_result = llm_classifier.predict(text)
                    # LLM 模型返回了结果
                    if "error" not in llm_result.get("debug", {}):
                        return (
                            llm_result["product_label"],
                            llm_result["sentiment_label"],
                            llm_result["keywords"],
                            llm_result["debug"],
                            list(llm_result.get("labels", []))[:8],
                        )
            except Exception as exc:
                # LLM 出错，继续尝试其他方法
                pass

        # 2) 尝试使用自定义/HuggingFace 模型
        if self.available and self.model is not None:
            try:
                if self.tokenizer is not None:
                    hf = self._predict_hf(text)
                    product_label, sentiment_label, keywords, debug, labels = rule_based_predict(text)
                    debug["hf_probs"] = hf.get("probs", [])
                    debug["model_mode"] = "hf"
                    return product_label, sentiment_label, keywords, debug, labels
                result = self.model.predict(text)
                product_label = normalize_binary_label(result.get("product_label", {}))
                sentiment_label = normalize_sentiment_label(result.get("sentiment_label", {}))
                keywords = list(result.get("keywords", []))[:8]
                labels = list(result.get("labels", []))[:8] or derive_multilabels(text, sentiment_label)
                debug = {"model_mode": "custom_model"}
                return product_label, sentiment_label, keywords, debug, labels
            except Exception as exc:
                self.load_error = f"infer_error: {exc}"

        # 3) 兜底使用规则基线
        product_label, sentiment_label, keywords, debug, labels = rule_based_predict(text)
        debug["model_mode"] = "rule_based"
        if self.load_error:
            debug["model_error"] = self.load_error
        return product_label, sentiment_label, keywords, debug, labels
# -----------------------------
# 规则基线工具方法
# -----------------------------
def clamp(value: float, min_v: float = 0.0, max_v: float = 1.0) -> float:
    return max(min_v, min(max_v, value))
def normalize_binary_label(label: Dict[str, float]) -> Dict[str, float]:
    product = float(label.get("商品类", 0.0))
    non_product = float(label.get("非商品类", 0.0))
    total = product + non_product
    if total <= 0:
        return {"商品类": 0.0, "非商品类": 0.0}
    return {"商品类": product / total, "非商品类": non_product / total}
def normalize_sentiment_label(label: Dict[str, float]) -> Dict[str, float]:
    pos = float(label.get("正向", 0.0))
    neu = float(label.get("中性", 0.0))
    neg = float(label.get("负向", 0.0))
    total = pos + neu + neg
    if total <= 0:
        return {"正向": 0.0, "中性": 0.0, "负向": 0.0}
    return {"正向": pos / total, "中性": neu / total, "负向": neg / total}
def keyword_occurrences(text: str, keyword: str) -> int:
    if re.search(r"[A-Za-z]", keyword):
        pattern = r"(?<![a-z])" + re.escape(keyword.lower()) + r"(?![a-z])"
        return len(re.findall(pattern, text.lower()))
    return text.count(keyword)
def count_hits(text: str, keywords: List[str]) -> int:
    return sum(keyword_occurrences(text, keyword) for keyword in keywords)
def extract_keywords(text: str, limit: int = 8) -> List[str]:
    hits = {}
    for word in COMMODITY_KEYWORDS + FINANCE_KEYWORDS:
        count = keyword_occurrences(text, word)
        if count > 0:
            hits[word] = count
    if hits:
        return [k for k, _ in sorted(hits.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))][
            :limit
        ]
    candidates = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    freq: Dict[str, int] = {}
    for token in candidates:
        freq[token] = freq.get(token, 0) + 1
    english_candidates = re.findall(r"[A-Za-z][A-Za-z-]{2,}", text.lower())
    for token in english_candidates:
        if token in ENGLISH_FALLBACK_STOPWORDS:
            continue
        freq[token] = freq.get(token, 0) + 1
    if not freq:
        return []
    return [k for k, _ in sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))][
        :limit
    ]
def derive_multilabels(text: str, sentiment_label: Dict[str, float]) -> List[str]:
    labels: List[str] = []
    for label, keywords in MULTILABEL_KEYWORDS.items():
        if count_hits(text, keywords) > 0:
            labels.append(label)

    if not any(label in labels for label in ["国际", "经济活动", "市场", "政策"]):
        if count_hits(text, COMMODITY_KEYWORDS) > 0 or count_hits(text, FINANCE_KEYWORDS) > 0:
            labels.append("市场")

    if not labels and count_hits(text, FINANCE_KEYWORDS) > 0:
        labels.append("经济活动")

    if sentiment_label:
        sentiment_main = sorted(sentiment_label.items(), key=lambda x: (-x[1], x[0]))[0][0]
        sentiment_tag = SENTIMENT_TO_MULTILABEL.get(sentiment_main)
        if sentiment_tag and sentiment_tag not in labels:
            labels.append(sentiment_tag)

    if not labels:
        labels.append("未知")

    return labels[:8]
def rule_based_predict(text: str) -> Tuple[Dict[str, float], Dict[str, float], List[str], Dict, List[str]]:
    product_hits = count_hits(text, COMMODITY_KEYWORDS)
    finance_hits = count_hits(text, FINANCE_KEYWORDS)
    score = clamp(0.2 + 0.25 * product_hits + 0.1 * finance_hits, 0.0, 1.0)
    product_prob = clamp(score, 0.05, 0.95)
    product_label = {"商品类": product_prob, "非商品类": 1.0 - product_prob}
    pos = count_hits(text, POSITIVE_WORDS)
    neg = count_hits(text, NEGATIVE_WORDS)
    if pos == 0 and neg == 0:
        sentiment_label = {"正向": 0.2, "中性": 0.6, "负向": 0.2}
    else:
        diff = pos - neg
        strength = clamp(abs(diff) / max(pos + neg, 1), 0.2, 0.8)
        if diff > 0:
            sentiment_label = {
                "正向": 0.5 + strength / 2,
                "中性": 0.3 - strength / 4,
                "负向": 0.2 - strength / 4,
            }
        elif diff < 0:
            sentiment_label = {
                "正向": 0.2 - strength / 4,
                "中性": 0.3 - strength / 4,
                "负向": 0.5 + strength / 2,
            }
        else:
            sentiment_label = {"正向": 0.3, "中性": 0.4, "负向": 0.3}
        sentiment_label = normalize_sentiment_label(sentiment_label)
    keywords = extract_keywords(text, limit=8)
    debug = {
        "product_hits": product_hits,
        "finance_hits": finance_hits,
        "sentiment_pos": pos,
        "sentiment_neg": neg,
    }
    labels = derive_multilabels(text, sentiment_label)
    debug["labels_source"] = "rule_based"
    return product_label, sentiment_label, keywords, debug, labels
def pick_main_label(probs: Dict[str, float]) -> Tuple[str, float]:
    if not probs:
        return "-", 0.0
    label, value = sorted(probs.items(), key=lambda x: (-x[1], x[0]))[0]
    return label, float(value)
def build_response(
    product_label: Dict[str, float],
    sentiment_label: Dict[str, float],
    keywords: List[str],
    debug: Dict,
    labels: Optional[List[str]] = None,
    preview: Optional[Dict[str, List]] = None,
) -> Dict:
    main_product_label, main_product_conf = pick_main_label(product_label)
    main_sentiment_label, main_sentiment_conf = pick_main_label(sentiment_label)
    return {
        "classification": {"label": main_product_label, "confidence": main_product_conf},
        "sentiment": {"label": main_sentiment_label, "score": round(main_sentiment_conf * 100, 2)},
        "labels": labels or [],
        "keywords": keywords,
        "product_label": product_label,
        "sentiment_label": sentiment_label,
        "preview": preview,
        "debug": debug,
    }
# -----------------------------
# 文件解析
# -----------------------------
def read_file_text(file: UploadFile) -> Tuple[str, Optional[str], Optional[Dict[str, List]]]:
    # 读取 txt/csv/docx/pdf/xlsx/xls 为纯文本，并提供 Excel 预览
    if file is None:
        return "", None, None
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".txt", ".csv"]:
        data = file.file.read()
        try:
            return data.decode("utf-8"), None, None
        except Exception:
            return data.decode("gbk", errors="ignore"), None, None
    if ext == ".docx":
        if docx is None:
            return "", "docx_missing_dependency", None
        try:
            data = file.file.read()
            document = docx.Document(io.BytesIO(data))
            text = "\n".join(p.text for p in document.paragraphs if p.text)
            return text, None, None
        except Exception as exc:
            return "", f"docx_read_error: {exc}", None
    if ext == ".pdf":
        if PdfReader is None:
            return "", "pdf_missing_dependency", None
        try:
            data = file.file.read()
            reader = PdfReader(io.BytesIO(data))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            text = "\n".join(pages).strip()
            return text, None, None
        except Exception as exc:
            return "", f"pdf_read_error: {exc}", None
    if ext in [".xlsx", ".xls"]:
        if pd is None:
            return "", "excel_missing_dependency", None
        try:
            data = file.file.read()
            df = pd.read_excel(io.BytesIO(data))
            df = df.fillna("")
            preview_rows = df.head(5).astype(str).values.tolist()
            preview = {"columns": [str(c) for c in df.columns], "rows": preview_rows}
            text = df.head(200).astype(str).to_csv(index=False)
            return text, None, preview
        except Exception as exc:
            return "", f"excel_read_error: {exc}", None
    return "", "unsupported_file", None
# -----------------------------
# API 接口
# -----------------------------
MODEL = ModelWrapper(os.getenv("MODEL_PATH"), os.getenv("WEIGHTS_PATH"))
class TextAnalyzeRequest(BaseModel):
    text: str
    model_version: Optional[str] = None
@app.post("/api/v1/analyze/text")
async def analyze_text(payload: TextAnalyzeRequest = Body(...)):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty_text")
    product_label, sentiment_label, keywords, debug, labels = MODEL.predict(text)
    debug.update({"text_length": len(text), "model_loaded": MODEL.available})
    return build_response(product_label, sentiment_label, keywords, debug, labels=labels)
@app.post("/api/v1/analyze/file")
async def analyze_file(file: UploadFile = File(...), model_version: Optional[str] = Form(None)):
    _ = model_version
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".xlsx", ".xls", ".docx"]:
        raise HTTPException(status_code=400, detail="unsupported_file")
    file_text, file_error, preview = read_file_text(file)
    if file_error:
        raise HTTPException(status_code=400, detail=file_error)
    if not file_text.strip():
        raise HTTPException(status_code=400, detail="empty_file_text")
    product_label, sentiment_label, keywords, debug, labels = MODEL.predict(file_text)
    debug.update(
        {"text_length": len(file_text), "model_loaded": MODEL.available, "file_name": file.filename}
    )
    return build_response(product_label, sentiment_label, keywords, debug, labels=labels, preview=preview)
@app.post("/api/predict")
async def predict(text: str = Form(""), file: Optional[UploadFile] = File(None)):
    # 兼容旧版接口：单条预测入口
    if not text and file is None:
        return {
            "labels": [],
            "product_label": {"商品类": 0.0, "非商品类": 0.0},
            "sentiment_label": {"正向": 0.0, "中性": 0.0, "负向": 0.0},
            "keywords": [],
            "debug": {"error": "empty_input"},
        }
    file_text, file_error, _preview = ("", None, None)
    if file is not None:
        file_text, file_error, _preview = read_file_text(file)
    merged_text = (text or "").strip()
    if file_text:
        merged_text = (merged_text + "\n" + file_text).strip()
    if not merged_text:
        return {
            "labels": [],
            "product_label": {"商品类": 0.0, "非商品类": 0.0},
            "sentiment_label": {"正向": 0.0, "中性": 0.0, "负向": 0.0},
            "keywords": [],
            "debug": {"error": file_error or "unsupported_file"},
        }
    product_label, sentiment_label, keywords, debug, labels = MODEL.predict(merged_text)
    debug.update({"text_length": len(merged_text), "model_loaded": MODEL.available})
    return {
        "labels": labels,
        "product_label": product_label,
        "sentiment_label": sentiment_label,
        "keywords": keywords,
        "debug": debug,
    }
