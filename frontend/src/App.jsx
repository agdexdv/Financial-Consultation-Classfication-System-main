// 主页面：金融咨询分类辅助系统前端
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { jsPDF } from "jspdf";
import "./App.css";

const MAX_CHARS = 1000;

const emptyResult = {
  classification: { label: "-", confidence: 0 },
  sentiment: { label: "-", score: 0 },
  product_label: { "商品类": 0, "非商品类": 0 },
  sentiment_label: { "正向": 0, "中性": 0, "负向": 0 },
  keywords: [],
  preview: null,
  debug: {},
};

const SENTIMENT_LABELS = {
  positive: "正向",
  negative: "负向",
  neutral: "中性",
  "正向": "正向",
  "负向": "负向",
  "中性": "中性",
};

function formatPercent(value) {
  if (typeof value !== "number") return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatScore(value) {
  if (typeof value !== "number") return "-";
  const normalized = value > 1 ? value : value * 100;
  return `${normalized.toFixed(0)}`;
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

function pickMainLabel(probs = {}) {
  const entries = Object.entries(probs);
  if (!entries.length) return ["-", 0];
  const [label, value] = entries.sort((a, b) => (b[1] || 0) - (a[1] || 0))[0];
  return [label, typeof value === "number" ? value : 0];
}

function normalizeResponse(data = {}) {
  const result = {
    ...emptyResult,
    product_label: data.product_label || emptyResult.product_label,
    sentiment_label: data.sentiment_label || emptyResult.sentiment_label,
    keywords: data.keywords || [],
    preview: data.preview || null,
    debug: data.debug || {},
  };

  if (data.classification) {
    result.classification = {
      label: data.classification.label ?? "-",
      confidence: Number(data.classification.confidence || 0),
    };
  } else {
    const [label, confidence] = pickMainLabel(result.product_label);
    result.classification = { label, confidence };
  }

  if (data.sentiment) {
    result.sentiment = {
      label: data.sentiment.label ?? "-",
      score: Number(data.sentiment.score || 0),
    };
  } else {
    const [label, confidence] = pickMainLabel(result.sentiment_label);
    result.sentiment = { label, score: Math.round(confidence * 100) };
  }

  return result;
}

function ConfidenceGauge({ value }) {
  const percent = Math.max(0, Math.min(1, Number(value) || 0));
  const degrees = percent * 360;
  let tone = "var(--tone-low)";
  if (percent >= 0.8) tone = "var(--tone-high)";
  else if (percent >= 0.5) tone = "var(--tone-mid)";

  return (
    <div className="gauge">
      <div
        className="gauge-ring"
        style={{
          background: `conic-gradient(${tone} ${degrees}deg, #e2e8f0 0deg)`,
        }}
      />
      <div className="gauge-center">
        <span>{formatPercent(percent)}</span>
        <small>置信度</small>
      </div>
    </div>
  );
}

function SentimentBadge({ label }) {
  const normalized = SENTIMENT_LABELS[label] || label || "-";
  const cls =
    normalized === "正向"
      ? "badge positive"
      : normalized === "负向"
        ? "badge negative"
        : "badge neutral";
  const icon = normalized === "正向" ? "↑" : normalized === "负向" ? "↓" : "•";
  return (
    <span className={cls}>
      <span className="badge-icon">{icon}</span>
      {normalized}
    </span>
  );
}

function SentimentMeter({ label, score }) {
  const normalizedLabel = SENTIMENT_LABELS[label] || label || "-";
  const normalizedScore = Math.max(0, Math.min(100, score > 1 ? score : score * 100));
  const tone =
    normalizedLabel === "正向"
      ? "var(--tone-positive)"
      : normalizedLabel === "负向"
        ? "var(--tone-negative)"
        : "var(--tone-neutral)";
  return (
    <div className="sentiment-meter">
      <div className="sentiment-meta">
        <span>强度</span>
        <span>{formatScore(normalizedScore)} / 100</span>
      </div>
      <div className="sentiment-track">
        <div className="sentiment-fill" style={{ width: `${normalizedScore}%`, background: tone }} />
      </div>
    </div>
  );
}

function DownloadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M12 3v10m0 0l4-4m-4 4l-4-4M5 17v2h14v-2"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M4 7h16M10 11v6M14 11v6M6 7l1 12h10l1-12M9 7l1-2h4l1 2"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M12 16V8m0 0l-3 3m3-3l3 3M5 16v2h14v-2"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function Spinner() {
  return <span className="spinner" aria-hidden="true" />;
}

function App() {
  const [mode, setMode] = useState("text");
  const [text, setText] = useState("");
  const [markdownTab, setMarkdownTab] = useState("edit");
  const [uploads, setUploads] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(emptyResult);
  const [resultsList, setResultsList] = useState([]);
  const [activeResultId, setActiveResultId] = useState(null);
  const [resultView, setResultView] = useState("detail");
  const [rawResponse, setRawResponse] = useState(null);
  const [toast, setToast] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState(() => {
    try {
      const stored = localStorage.getItem("analysis_history");
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });
  const overviewRef = useRef(null);
  const detailRef = useRef(null);
  const [overviewHeight, setOverviewHeight] = useState("0px");
  const [detailHeight, setDetailHeight] = useState("0px");

  const textCount = text.length;
  const isTextTooLong = textCount > MAX_CHARS;

  const markdownHtml = useMemo(() => {
    const rendered = marked.parse(text || "");
    return DOMPurify.sanitize(rendered);
  }, [text]);

  const mainProduct = useMemo(() => {
    if (result.classification?.label) {
      return [result.classification.label, result.classification.confidence || 0];
    }
    return pickMainLabel(result.product_label);
  }, [result]);

  const mainSentiment = useMemo(() => {
    if (result.sentiment?.label) {
      return [result.sentiment.label, result.sentiment.score || 0];
    }
    const [label, value] = pickMainLabel(result.sentiment_label);
    return [label, Math.round(value * 100)];
  }, [result]);

  function showToast(message, tone = "error") {
    setToast({ id: Date.now(), message, tone });
    setTimeout(() => setToast(null), 3200);
  }

  function persistHistory(next) {
    setHistory(next);
    localStorage.setItem("analysis_history", JSON.stringify(next));
  }

  function addHistory(entry) {
    const next = [entry, ...history].slice(0, 8);
    persistHistory(next);
  }

  function removeHistory(id) {
    const next = history.filter((item) => item.id !== id);
    persistHistory(next);
  }

  function clearHistory() {
    persistHistory([]);
  }

  function handleTextChange(event) {
    const value = event.target.value || "";
    if (value.length <= MAX_CHARS) {
      setText(value);
    } else {
      setText(value.slice(0, MAX_CHARS));
    }
  }

  function handleFiles(selectedFiles) {
    const allowed = [".pdf", ".xlsx", ".xls", ".docx"];
    const list = Array.from(selectedFiles || []);
    const valid = [];

    list.forEach((file) => {
      const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
      if (!allowed.includes(ext)) {
        showToast(`文件格式不支持：${file.name}`, "error");
        return;
      }
      valid.push({
        id: `${file.name}-${file.size}-${file.lastModified}`,
        file,
        progress: 0,
        status: "pending",
        error: "",
      });
    });

    if (valid.length) {
      setUploads(valid);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);
    handleFiles(event.dataTransfer.files);
  }

  function updateUpload(id, patch) {
    setUploads((prev) => prev.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  }

  function removeUpload(id) {
    setUploads((prev) => prev.filter((item) => item.id !== id));
  }

  async function analyzeText() {
    if (!text.trim()) {
      showToast("请输入文本后再分析");
      return;
    }
    setLoading(true);
    try {
      const response = await fetch("/api/v1/analyze/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model_version: "qwen2.5-instruct-lora" }),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || "请求失败");
      }

      const data = await response.json();
      setRawResponse(data);
      setResult(normalizeResponse(data));
      setResultsList([]);
      setActiveResultId(null);
      setResultView("detail");
      addHistory({
        id: Date.now(),
        mode: "text",
        title: text.slice(0, 18) || "文本分析",
        result: data,
        createdAt: new Date().toISOString(),
      });
    } catch (err) {
      showToast(err.message || "分析失败");
    } finally {
      setLoading(false);
    }
  }

  function uploadFile(file, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/v1/analyze/file");
      xhr.responseType = "json";
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          onProgress(Math.round((event.loaded / event.total) * 100));
        }
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(xhr.response);
        } else {
          const detail = xhr.response?.detail || xhr.response?.message || xhr.statusText;
          reject(new Error(detail || "上传失败"));
        }
      };
      xhr.onerror = () => reject(new Error("网络错误"));
      const formData = new FormData();
      formData.append("file", file);
      formData.append("model_version", "qwen2.5-instruct-lora");
      xhr.send(formData);
    });
  }

  async function analyzeFiles() {
    if (!uploads.length) {
      showToast("请先选择文件");
      return;
    }
    setLoading(true);
    setResultView("overview");
    try {
      for (const item of uploads) {
        updateUpload(item.id, { status: "uploading", progress: 0, error: "" });
        try {
          const data = await uploadFile(item.file, (progress) =>
            updateUpload(item.id, { progress })
          );
          const normalized = normalizeResponse(data);
          const entry = {
            id: Date.now() + Math.random(),
            fileName: item.file.name,
            result: normalized,
            raw: data,
          };
          updateUpload(item.id, { status: "success", progress: 100 });
          setRawResponse(data);
          setResult(normalized);
          setResultsList((prev) => [...prev, entry]);
          setActiveResultId(entry.id);
          addHistory({
            id: Date.now() + Math.random(),
            mode: "file",
            title: item.file.name,
            result: data,
            createdAt: new Date().toISOString(),
          });
          showToast(`上传成功：${item.file.name}`, "success");
        } catch (err) {
          updateUpload(item.id, { status: "error", progress: 0, error: err.message });
          showToast(err.message || "上传失败");
        }
      }
    } finally {
      setLoading(false);
    }
  }

  function handleAnalyze() {
    if (mode === "text") {
      analyzeText();
    } else {
      analyzeFiles();
    }
  }

  function handleClear() {
    setText("");
    setUploads([]);
    setResult(emptyResult);
    setResultsList([]);
    setActiveResultId(null);
    setResultView("detail");
    setRawResponse(null);
  }

  function loadHistory(item) {
    setResult(normalizeResponse(item.result));
    setRawResponse(item.result);
    setResultsList([]);
    setActiveResultId(null);
    setResultView("detail");
    setShowHistory(false);
  }

  function selectFileResult(entry) {
    setResult(entry.result);
    setRawResponse(entry.raw);
    setActiveResultId(entry.id);
    setResultView("detail");
  }

  function downloadJson() {
    const payload = { single: result };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "analysis_report.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadCsv() {
    const rows = [["type", "text", "classification", "sentiment", "keywords"]];
    const classification = JSON.stringify(result.classification || {});
    const sentiment = JSON.stringify(result.sentiment || {});
    const keywords = (result.keywords || []).join("|");
    rows.push(["single", text.trim(), classification, sentiment, keywords]);

    const csv = rows
      .map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(","))
      .join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "analysis_report.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadPdf() {
    const doc = new jsPDF();
    const lines = [
      "Financial Analysis Report",
      `Time: ${new Date().toLocaleString()}`,
      `Classification: ${mainProduct[0]} (${formatPercent(mainProduct[1])})`,
      `Sentiment: ${SENTIMENT_LABELS[mainSentiment[0]] || mainSentiment[0]} (${formatScore(
        result.sentiment?.score || 0
      )}/100)`,
      `Keywords: ${(result.keywords || []).join(", ") || "N/A"}`,
    ];
    doc.setFont("helvetica", "normal");
    doc.setFontSize(12);
    doc.text(lines, 14, 20, { maxWidth: 180 });
    doc.save("analysis_report.pdf");
  }

  const secondaryLabels = useMemo(() => {
    const entries = Object.entries(result.product_label || {});
    return entries
      .sort((a, b) => (b[1] || 0) - (a[1] || 0))
      .map(([label, value]) => ({ label, value }));
  }, [result]);

  const recentHistory = useMemo(() => history.slice(0, 3), [history]);
  const showOverview = mode === "file" && resultsList.length && resultView === "overview";
  const overviewJson = useMemo(
    () =>
      resultsList.map((entry) => ({
        fileName: entry.fileName,
        result: entry.raw,
      })),
    [resultsList]
  );

  useLayoutEffect(() => {
    const raf = requestAnimationFrame(() => {
      if (overviewRef.current) {
        setOverviewHeight(`${overviewRef.current.scrollHeight}px`);
      }
      if (detailRef.current) {
        setDetailHeight(`${detailRef.current.scrollHeight}px`);
      }
    });
    return () => cancelAnimationFrame(raf);
  }, [resultsList, result, rawResponse, showOverview, mode]);

  useEffect(() => {
    if (typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(() => {
      if (overviewRef.current) {
        setOverviewHeight(`${overviewRef.current.scrollHeight}px`);
      }
      if (detailRef.current) {
        setDetailHeight(`${detailRef.current.scrollHeight}px`);
      }
    });
    if (overviewRef.current) {
      observer.observe(overviewRef.current);
    }
    if (detailRef.current) {
      observer.observe(detailRef.current);
    }
    return () => observer.disconnect();
  }, []);

  function recalcViewHeights() {
    requestAnimationFrame(() => {
      if (overviewRef.current) {
        setOverviewHeight(`${overviewRef.current.scrollHeight}px`);
      }
      if (detailRef.current) {
        setDetailHeight(`${detailRef.current.scrollHeight}px`);
      }
    });
  }

  return (
    <div className="page">
      <header className="hero">
        <div className="hero-content">
          <div className="hero-strip">
            <div className="logo-dot" />
            <span>Financial Intelligence</span>
          </div>
          <div className="hero-title">金融咨询分类辅助系统</div>
          <p className="hero-sub">
            面向金融咨询文本的快速分类：判断是否为商品类，并输出情感倾向与关键实体。
          </p>
          <div className="hero-tags">
            <span className="hero-tag">商品类识别</span>
            <span className="hero-tag">情感分析</span>
            <span className="hero-tag">结构化上传</span>
          </div>
        </div>
      </header>

      <main className="grid">
        <section className="panel">
          <div className="panel-title">输入内容</div>
          <div className="tabs">
            <button
              className={mode === "text" ? "tab active" : "tab"}
              onClick={() => {
                setMode("text");
                setResultView("detail");
              }}
              disabled={loading}
            >
              文本分析
            </button>
            <button
              className={mode === "file" ? "tab active" : "tab"}
              onClick={() => {
                setMode("file");
                if (resultsList.length) {
                  setResultView("overview");
                }
              }}
              disabled={loading}
            >
              文件上传
            </button>
          </div>

          {mode === "text" ? (
            <div className="text-mode">
              <div className="subtabs">
                <button
                  className={markdownTab === "edit" ? "subtab active" : "subtab"}
                  onClick={() => setMarkdownTab("edit")}
                  disabled={loading}
                >
                  编辑
                </button>
                <button
                  className={markdownTab === "preview" ? "subtab active" : "subtab"}
                  onClick={() => setMarkdownTab("preview")}
                  disabled={loading}
                >
                  预览
                </button>
              </div>

              {markdownTab === "edit" ? (
                <label className="field">
                  <span>咨询文本</span>
                  <textarea
                    value={text}
                    onChange={handleTextChange}
                    placeholder="请输入金融新闻文本..."
                    rows={10}
                  />
                </label>
              ) : (
                <div className="markdown-preview">
                  {text.trim() ? (
                    <div
                      className="markdown-body"
                      dangerouslySetInnerHTML={{ __html: markdownHtml }}
                    />
                  ) : (
                    <span className="empty">暂无预览内容</span>
                  )}
                </div>
              )}

              <div className={isTextTooLong ? "word-count warn" : "word-count"}>
                字数：{textCount} / {MAX_CHARS}
              </div>
            </div>
          ) : (
            <div className="file-mode">
              <label
                className={dragActive ? "file-box active" : "file-box"}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  accept=".pdf,.xlsx,.xls,.docx"
                  multiple
                  onChange={(event) => handleFiles(event.target.files)}
                />
                <UploadIcon />
                <div className="file-hint">拖拽或点击上传 PDF / Excel / Word</div>
              </label>

              <div className="upload-list">
                {uploads.length ? (
                  uploads.map((item) => (
                    <div key={item.id} className={`upload-item ${item.status}`}>
                      <div>
                        <div className="upload-name">{item.file.name}</div>
                        <div className="upload-meta">{formatBytes(item.file.size)}</div>
                      </div>
                      <div className="upload-progress">
                        <div className="upload-track">
                          <div
                            className="upload-fill"
                            style={{ width: `${item.progress}%` }}
                          />
                        </div>
                        <span className="upload-state">
                          {item.status === "success"
                            ? "成功"
                            : item.status === "error"
                              ? "失败"
                              : item.status === "uploading"
                                ? "上传中"
                                : "待上传"}
                        </span>
                      </div>
                      <button
                        className="ghost small"
                        onClick={() => removeUpload(item.id)}
                        disabled={loading}
                      >
                        移除
                      </button>
                    </div>
                  ))
                ) : (
                  <div className="empty">暂无上传文件</div>
                )}
              </div>
            </div>
          )}

          <div className="action-row">
            <button
              className="primary"
              onClick={handleAnalyze}
              disabled={loading || (mode === "text" && (!text.trim() || isTextTooLong))}
            >
              {loading ? (
                <>
                  <Spinner />
                  分析中...
                </>
              ) : (
                "立即分析"
              )}
            </button>
            <button className="ghost" onClick={handleClear} disabled={loading}>
              <TrashIcon />
              清空
            </button>
          </div>

          <div className="history">
            <div className="history-title-row">
              <div className="history-title">历史记录</div>
              <button className="ghost small" onClick={() => setShowHistory(true)}>
                查看全部
              </button>
            </div>
            {recentHistory.length ? (
              recentHistory.map((item) => (
                <div key={item.id} className="history-item">
                  <button className="history-main" onClick={() => loadHistory(item)}>
                    <span>{item.title}</span>
                    <small>{new Date(item.createdAt).toLocaleString()}</small>
                  </button>
                  <button
                    className="history-delete"
                    onClick={(event) => {
                      event.stopPropagation();
                      removeHistory(item.id);
                    }}
                  >
                    删除
                  </button>
                </div>
              ))
            ) : (
              <div className="empty">暂无历史记录</div>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-title">分析结果</div>
          {mode === "file" && resultsList.length && resultView === "detail" ? (
            <div className="result-toolbar">
              <button className="toggle-pill" onClick={() => setResultView("overview")}>
                <span className="toggle-icon">←</span>
                返回概览
              </button>
            </div>
          ) : null}

          {!showOverview ? (
            <div className="export-row">
              <button className="ghost" onClick={downloadJson}>
                <DownloadIcon />
                导出 JSON
              </button>
              <button className="ghost" onClick={downloadCsv}>
                <DownloadIcon />
                导出 CSV
              </button>
              <button className="ghost" onClick={downloadPdf}>
                <DownloadIcon />
                导出 PDF
              </button>
            </div>
          ) : null}

          {loading ? (
            <div className="skeleton">
              <div className="skeleton-title">Qwen2.5 + LoRA 模型正在推理中...</div>
              <div className="skeleton-grid">
                <div className="skeleton-card" />
                <div className="skeleton-card" />
              </div>
              <div className="skeleton-line" />
              <div className="skeleton-line" />
            </div>
          ) : (
            <>
              <div
                ref={overviewRef}
                className={`result-view overview ${showOverview ? "open" : "collapsed"}`}
                style={{ maxHeight: showOverview ? overviewHeight : "0px" }}
              >
                <div className="file-results">
                  {resultsList.map((entry) => (
                    <button
                      key={entry.id}
                      className={`file-result-card ${activeResultId === entry.id ? "active" : ""}`}
                      onClick={() => selectFileResult(entry)}
                    >
                      <div className="file-result-header">
                        <div className="file-result-title">{entry.fileName}</div>
                        <span className="file-result-meta">
                          {formatPercent(entry.result.classification?.confidence || 0)}
                        </span>
                      </div>
                      <div className="file-result-row">
                        <span>分类</span>
                        <strong>{entry.result.classification?.label || "-"}</strong>
                      </div>
                      <div className="file-result-row">
                        <span>情感</span>
                        <SentimentBadge label={entry.result.sentiment?.label} />
                      </div>
                      <div className="file-result-tags">
                        {(entry.result.keywords || []).slice(0, 6).map((item) => (
                          <span key={item}>{item}</span>
                        ))}
                        {!entry.result.keywords?.length ? (
                          <span className="empty">无关键词</span>
                        ) : null}
                      </div>
                    </button>
                  ))}
                </div>
                <div className="result-section">
                  <div className="result-section-title">JSON 数据视图</div>
                  <details className="accordion" onToggle={recalcViewHeights}>
                    <summary>查看批量原始数据</summary>
                    <pre>{JSON.stringify(overviewJson, null, 2)}</pre>
                  </details>
                </div>
              </div>

              <div
                ref={detailRef}
                className={`result-view detail ${showOverview ? "collapsed" : "open"}`}
                style={{ maxHeight: showOverview ? "0px" : detailHeight }}
              >
                <div className="result-grid">
                  <div className="result-card">
                    <div className="result-card-title">商品类判断</div>
                    <div className="result-card-value">{mainProduct[0]}</div>
                    <ConfidenceGauge value={mainProduct[1]} />
                    <div className="secondary-labels">
                      {secondaryLabels.map((item) => (
                        <div key={item.label} className="secondary-item">
                          <span>{item.label}</span>
                          <strong>{formatPercent(item.value || 0)}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="result-card">
                    <div className="result-card-title">情感倾向</div>
                    <div className="result-card-value">
                      <SentimentBadge label={mainSentiment[0]} />
                    </div>
                    <div className="result-card-sub">
                      概率：{formatPercent((result.sentiment_label || {})[SENTIMENT_LABELS[mainSentiment[0]] || mainSentiment[0]] || 0)}
                    </div>
                    <SentimentMeter label={mainSentiment[0]} score={result.sentiment?.score || 0} />
                  </div>
                </div>

                <div className="result-section">
                  <div className="result-section-title">关键词</div>
                  <div className="keywords">
                    {result.keywords.length ? (
                      result.keywords.map((item) => <span key={item}>{item}</span>)
                    ) : (
                      <span className="empty">暂无关键词</span>
                    )}
                  </div>
                </div>

                {result.preview ? (
                  <div className="result-section">
                    <div className="result-section-title">Excel 预览（前 5 行）</div>
                    <div className="preview-table">
                      <table>
                        <thead>
                          <tr>
                            {result.preview.columns?.map((col, index) => (
                              <th key={`${col}-${index}`}>{col}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {result.preview.rows?.map((row, rowIndex) => (
                            <tr key={`row-${rowIndex}`}>
                              {row.map((cell, cellIndex) => (
                                <td key={`cell-${rowIndex}-${cellIndex}`}>{cell}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : null}

                <div className="result-section">
                  <div className="result-section-title">JSON 数据视图</div>
                  <details className="accordion" onToggle={recalcViewHeights}>
                    <summary>查看原始数据</summary>
                    <pre>{JSON.stringify(rawResponse || result.debug, null, 2)}</pre>
                  </details>
                </div>
              </div>
            </>
          )}
        </section>
      </main>

      {toast ? (
        <div className={`toast ${toast.tone}`} key={toast.id}>
          {toast.message}
        </div>
      ) : null}

      {showHistory ? (
        <div className="modal-overlay" onClick={() => setShowHistory(false)}>
          <div className="modal-card" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <div>
                <div className="modal-title">历史记录</div>
                <div className="modal-sub">最多保留 8 条最近记录</div>
              </div>
              <button className="ghost small" onClick={() => setShowHistory(false)}>
                关闭
              </button>
            </div>
            <div className="modal-body">
              {history.length ? (
                history.map((item) => (
                  <div key={item.id} className="history-item">
                    <button className="history-main" onClick={() => loadHistory(item)}>
                      <span>{item.title}</span>
                      <small>{new Date(item.createdAt).toLocaleString()}</small>
                    </button>
                    <button
                      className="history-delete"
                      onClick={(event) => {
                        event.stopPropagation();
                        removeHistory(item.id);
                      }}
                    >
                      删除
                    </button>
                  </div>
                ))
              ) : (
                <div className="empty">暂无历史记录</div>
              )}
            </div>
            {history.length ? (
              <div className="modal-footer">
                <button className="ghost small" onClick={clearHistory}>
                  清空历史记录
                </button>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default App;
