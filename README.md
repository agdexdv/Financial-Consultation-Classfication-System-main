# 金融咨询分类辅助系统

本项目采用 **React 静态前端 + FastAPI 后端** 架构，提供金融咨询文本的分类与情感分析能力。

## 页面功能（已对照 PRD 更新）

- **双栏 Dashboard 布局**：左侧输入控制区（40%），右侧结果可视化区（60%），移动端自动折叠单栏
- **文本分析模式**：支持 Markdown 预览、字数统计（限制 1000 字）、一键清空、分析 Loading
- **文件上传模式**：支持拖拽/多选上传 `pdf/xlsx/xls/docx`，显示上传列表、进度与状态
- **Excel 预览**：若上传 Excel，前端展示前 5 行数据预览
- **结果可视化**：商品类置信度仪表盘、情感强度条、关键词 Tag 云、多标签列表
- **JSON 原始数据视图**：折叠面板查看后端原始返回，便于调试
- **交互体验**：骨架屏、Toast 错误提示
- **历史记录**：LocalStorage 保存最近 8 条分析（支持删除/清空）
- **报告导出**：支持 JSON / CSV / PDF 导出
- **批量结果概览**：分组卡片 + JSON 视图，点击卡片查看详情，支持收起回到概览

## 目录结构

- `frontend/`  React (Vite)
  - `frontend/src/App.jsx`  页面与交互逻辑
  - `frontend/src/App.css`  页面样式
  - `frontend/src/index.css`  全局字体与背景
  - `frontend/vite.config.js`  代理 `/api` 到后端
- `backend/`  FastAPI
  - `backend/app.py`  接口与推理逻辑（含模型扩展入口）
  - `backend/requirements.txt`  后端依赖
- `README.md`  项目说明

## 启动方式

### 1) 启动后端

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

> 说明：Excel 解析依赖 `pandas/openpyxl/xlrd`，已加入 `requirements.txt`。

### 2) 启动前端

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 `http://localhost:5173`。
![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)

## 接口说明（PRD 对齐）

### `POST /api/v1/analyze/text`

**Request (JSON)**
```json
{
  "text": "受美联储加息影响，今日国际金价下跌...",
  "model_version": "bert-base-finetuned-v1"
}
```

**Response (JSON)**
```json
{
  "classification": {"label": "黄金", "confidence": 0.98},
  "sentiment": {"label": "negative", "score": 85},
  "keywords": ["美联储", "加息", "金价"],
  "product_label": {"黄金": 0.92, "原油": 0.03, "螺纹钢": 0.02},
  "sentiment_label": {"正向": 0.1, "中性": 0.2, "负向": 0.7},
  "preview": null,
  "debug": {"text_length": 120, "model_loaded": false}
}
```

### `POST /api/v1/analyze/file`

**Content-Type:** `multipart/form-data`
- `file`: 仅支持 `pdf/xlsx/xls/docx`
- `model_version`: 可选

**Response:** 同上，若为 Excel 会包含 `preview`（前 5 行）。

### 兼容旧接口

- `POST /api/predict`：保留旧版接口，仍支持 `text` + `file` 组合。

## 模型接入（重点）

模型微调后可能产出：**一个模型文件 + 一个权重文件**。后端已预留扩展入口，位置在：

- `backend/app.py` -> `ModelWrapper` 类
- `backend/app.py` -> `_load_model()` 和 `_predict_hf()`

### 方案 A：自定义 Python 模型文件（推荐）

设置环境变量：
- `MODEL_PATH`：模型 Python 文件路径（`.py`）
- `WEIGHTS_PATH`：权重路径

模型文件需提供：

```python
def load_model(weights_path):
    return MyModel(weights_path)

class MyModel:
    def predict(self, text):
        return {
            "product_label": {"黄金": 0.92, "原油": 0.03, "螺纹钢": 0.02},
            "sentiment_label": {"正向": 0.6, "中性": 0.3, "负向": 0.1},
            "keywords": ["原油", "价格"],
        }
```

### 方案 B：HuggingFace/BERT 目录（多分类输出）

如果模型侧产出的是 BERT 等 HuggingFace 目录：

- 将 `MODEL_PATH` 指向模型目录
- 安装额外依赖（按需）：`transformers` 和 `torch`

**推荐输出格式（多分类商品）：**
- 模型输出 `logits/probs` 对应商品类别列表（如 `黄金/原油/螺纹钢/...`）
- 后端将其映射为 `product_label: {label: prob}`
- `classification` 使用最高概率作为主类别

你可以在 `backend/app.py` 的 `_predict_hf()` 与 `predict()` 中完成“概率 -> 标签”的映射。

## Pending Items（模型侧）

1. **BERT 多分类微调与输出**
- 需要模型侧提供商品类别的多分类输出（label -> prob）
- 需要提供 label mapping 列表与推理输出格式
- 若仅有“商品类/非商品类”二分类，前端无法展示具体商品名称

2. **情感输出规范化**
- 需要明确情感标签集合与顺序（正/中/负）
- 需要给出 score 的含义范围（0-1 or 0-100）

3. **实体高亮（可选高级功能）**
- 若希望在原文中高亮实体，模型需返回 span 位置信息

### Impact on UI
- **无多分类输出时**：
  - “商品分类”卡片只能展示“商品类/非商品类”
  - 多标签列表无法显示具体商品
- **缺少情感规范时**：
  - 情感强度条可能显示不准确
- **无实体 span 时**：
  - 仅能展示关键词 Tag，无法在原文中高亮

## 导出说明

- JSON/CSV 为标准文本导出
- PDF 由前端 `jsPDF` 生成，当前为英文模板；如需中文字体，需要额外嵌入字体文件

## 常见问题

- **后端访问 `/` 返回 404**：正常，后端只提供 API。
- **前端接口报错**：确认后端已启动，且 `http://127.0.0.1:8000` 可访问。
- **PDF/Excel/Word 无法解析**：确认后端已安装 `pypdf` / `pandas` / `openpyxl` / `xlrd` / `python-docx`。
