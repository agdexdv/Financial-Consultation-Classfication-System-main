# 金融咨询分类辅助系统 PRD（对照实现更新）

## 1. 页面整体架构 (Page Structure)
- React SPA，响应式分栏 Dashboard
- 左侧 40%：输入与控制区（Input & Control Panel）
- 右侧 60%：分析结果可视化区（Analysis Visualization Dashboard）
- 移动端自动折叠为单栏（Input 在上，Result 在下）

## 2. 核心功能模块

### 2.1 多模态输入控制区 (Input Zone)

#### 2.1.1 文本分析模式 (Text Mode)
- 多行文本输入，支持 Markdown 预览
- Placeholder：`请输入金融新闻文本...`
- 字数统计 + 限制（当前实现 1000 字）
- 立即分析按钮：请求 `/api/v1/analyze/text`
- Loading 状态禁用按钮
- 清空按钮

#### 2.1.2 文件上传模式 (File Mode)
- 拖拽上传（Dropzone），支持点击选择或拖入
- 支持多文件上传，展示列表 + 进度条 + 状态
- 支持格式：`pdf/xlsx/xls/docx`
- 上传后调用 `/api/v1/analyze/file`（逐文件上传）
- Excel 预览：前 5 行数据

### 2.2 智能分析结果区 (Result Dashboard)

#### 2.2.0 批量结果概览/详情
- 批量上传后默认显示“分组卡片 + JSON 视图（批量原始数据）”
- 点击卡片进入该文件的详细分析结果
- 提供“返回概览”按钮回到概览
- 概览/详情切换带折叠高度动画（平滑过渡）

#### 2.2.1 大宗商品分类卡片 (Classification Card)
- 主类别标签高亮显示
- 置信度仪表盘（环形图）
- 多标签列表（按权重降序）
- 颜色逻辑：高/中/低置信度区分

#### 2.2.2 情感分析模块 (Sentiment Analysis)
- 情感倾向：Positive/Negative/Neutral（前端显示正向/负向/中性）
- 中国股市习惯：正向红色、负向绿色
- 情感强度条（0-100）

#### 2.2.3 关键实体识别 (NER & Keywords)
- 关键词 Tag 云展示
- 高亮实体（span）为高级功能，当前实现为关键词展示

#### 2.2.4 JSON 数据视图 (Developer Mode)
- 折叠面板查看原始 JSON
- 概览视图提供“批量原始数据”JSON

## 3. 交互体验与状态管理 (UX & State)
- 骨架屏：分析加载时展示
- 错误处理：Toast 提示（右上角）
- 响应式布局适配移动端

## 4. 前后端接口定义 (API Contract)

### 4.1 文本分析接口
Endpoint: `POST /api/v1/analyze/text`

Request:
```json
{
  "text": "受美联储加息影响，今日国际金价下跌...",
  "model_version": "bert-base-finetuned-v1"
}
```

Response:
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

### 4.2 文件上传接口
Endpoint: `POST /api/v1/analyze/file`

Content-Type: `multipart/form-data`
- `file`: `pdf/xlsx/xls/docx`
- `model_version`: 可选

Response: 同上，Excel 会包含 `preview`（前 5 行）

### 4.3 兼容旧接口
- `POST /api/predict` 仍保留

## 5. 模型侧 Pending Items 与 UI 影响

### 5.1 Pending Items（模型侧）
1. **BERT 多分类微调与输出**
- 需要模型侧提供商品类别的多分类输出（label -> prob）
- 需要提供 label mapping 列表与推理输出格式
- 若仅有“商品类/非商品类”二分类，前端无法展示具体商品名称

2. **情感输出规范化**
- 需要明确情感标签集合与顺序（正/中/负）
- 需要给出 score 的含义范围（0-1 or 0-100）

3. **实体高亮（可选高级功能）**
- 若希望在原文中高亮实体，模型需返回 span 位置信息

### 5.2 Impact on UI
- **无多分类输出时**：
  - “商品分类”卡片只能展示“商品类/非商品类”
  - 多标签列表无法显示具体商品
- **缺少情感规范时**：
  - 情感强度条可能显示不准确
- **无实体 span 时**：
  - 仅能展示关键词 Tag，无法在原文中高亮

## 6. 开发路线建议 (Roadmap)

### P0（核心版）
- React 基础框架
- 文本输入 + 结果卡片
- 调通 React -> FastAPI

### P1（完整版）
- 文件上传组件（拖拽）
- 结果可视化（图表/仪表盘/标签云）
- 关键词 Tag 展示

### P2（优化版）
- 历史记录（LocalStorage）
- 历史记录弹窗 + 支持删除/清空
- PDF 导出
