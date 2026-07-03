# Diabetes AI System

以 PyTorch 建立的糖尿病風險預測專案，整合資料前處理、模型推論、SHAP 可解釋性與 Streamlit 互動介面。

## 專案亮點

- 輸入 8 項病人特徵，快速取得風險分數
- 輸出低 / 中 / 高風險分層與建議說明
- 產生 SHAP waterfall 圖與文字報告
- 訓練與預測共用同一份 scaler，維持流程一致
- 提供命令列與 Streamlit 兩種使用方式

## 技術架構

- 模型：PyTorch 二元分類神經網路
- 前處理：StandardScaler
- 可解釋性：SHAP
- 前端：Streamlit
- 資料集：`data/diabetes.csv`取自Pima Indians Diabetes Database

## 專案結構

- `app.py`：Streamlit 入口
- `src/train.py`：模型訓練與輸出權重
- `src/predict.py`：命令列預測、風險分層、SHAP 解釋與報告輸出
- `src/explain.py`：獨立 SHAP 分析腳本
- `src/preprocess.py`：資料前處理與 scaler
- `src/model.py`：PyTorch 模型定義
- `models/diabetes_model.pth`：訓練後模型權重
- `models/scaler.pkl`：固定前處理 scaler
- `data/diabetes.csv`：原始資料集

## 安裝方式

1. 建立虛擬環境並啟用。
2. 安裝依賴：

```bash
pip install -r requirements.txt
```

## 使用方式

### 啟動 Streamlit 前端

```bash
streamlit run app.py
```

### 使用終端進行預測

```bash
python src/predict.py
```

### 重新訓練模型

```bash
python src/train.py
```

### 只執行 SHAP 分析

```bash
python src/explain.py
```

## 輸出結果

預測流程會在 `outputs/` 產生：

- `predict_report.txt`：文字報告
- `predict_shap_waterfall.png`：SHAP 解釋圖
- `shap_summary.png`：SHAP summary 圖


## 注意事項

- 這是利用PyTorch模型建立的糖尿病風險初步篩檢工具，不是真實的醫療診斷工具，結果僅供參考。
