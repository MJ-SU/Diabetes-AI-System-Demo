from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

try:
    from src.model import DiabetesModel
    from src.preprocess import FEATURE_COLUMNS, load_scaler, prepare_data, transform_input_features
except ImportError:
    from model import DiabetesModel
    from preprocess import FEATURE_COLUMNS, load_scaler, prepare_data, transform_input_features

# 依優先順序嘗試設定常見的中文字型；若都找不到（例如部署在無中文字型的 Linux 主機），
# 則回傳 False，呼叫端應 fallback 成英文標籤，避免畫出方框字。
_CJK_FONT_CANDIDATES = [
    "Microsoft JhengHei",   # Windows 繁中
    "Microsoft YaHei",      # Windows 簡中
    "PingFang TC",          # macOS 繁中
    "PingFang SC",          # macOS 簡中
    "Noto Sans CJK TC",     # Linux (如有安裝 fonts-noto-cjk)
    "Noto Sans CJK SC",
    "Noto Sans TC",
    "SimHei",
    "Heiti TC",
]

#嘗試設定可顯示中文的字型，回傳 True 表示成功、False 表示找不到，需 fallback 英文
def configure_cjk_font():

    import matplotlib
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True

    # 找不到中文字型，修正負號顯示問題，回傳 False 讓呼叫端改用英文標籤
    matplotlib.rcParams["axes.unicode_minus"] = False
    return False


# 臨床參考設定 hard_min / hard_max: 生理上不可能的範圍，超出則拒絕輸入
# soft_min / soft_max: 常見參考範圍，超出僅顯示警示，仍允許繼續
FEATURE_GUIDE = {
    "Pregnancies": {
        "display_name": "懷孕次數",
        "english_name": "Pregnancies",
        "unit": "次",
        "unit_en": "times",
        "hard_min": 0, "hard_max": 20,
        "soft_min": 0, "soft_max": 17,
        "note": "懷孕次數",
    },
    "Glucose": {
        "display_name": "血糖",
        "english_name": "Glucose",
        "unit": "mg/dL",
        "unit_en": "mg/dL",
        "hard_min": 1, "hard_max": 400,
        "soft_min": 70, "soft_max": 140,
        "note": "空腹/常見血糖參考區間（0 視為缺值，不允許輸入）",
    },
    "BloodPressure": {
        "display_name": "舒張壓",
        "english_name": "Blood Pressure",
        "unit": "mmHg",
        "unit_en": "mmHg",
        "hard_min": 1, "hard_max": 200,
        "soft_min": 60, "soft_max": 90,
        "note": "舒張壓常見參考範圍（0 視為缺值，不允許輸入）",
    },
    "SkinThickness": {
        "display_name": "皮下脂肪厚度",
        "english_name": "Skin Thickness",
        "unit": "mm",
        "unit_en": "mm",
        "hard_min": 0, "hard_max": 100,
        "soft_min": 10, "soft_max": 40,
        "note": "皮下脂肪厚度",
    },
    "Insulin": {
        "display_name": "胰島素",
        "english_name": "Insulin",
        "unit": "uU/mL",
        "unit_en": "uU/mL",
        "hard_min": 0, "hard_max": 900,
        "soft_min": 16, "soft_max": 166,
        "note": "胰島素參考區間",
    },
    "BMI": {
        "display_name": "身體質量指數",
        "english_name": "BMI",
        "unit": "kg/m^2",
        "unit_en": "kg/m^2",
        "hard_min": 5, "hard_max": 80,
        "soft_min": 18.5, "soft_max": 24.9,
        "note": "身體質量指數",
    },
    "DiabetesPedigreeFunction": {
        "display_name": "家族糖尿病風險指數",
        "english_name": "Diabetes Pedigree",
        "unit": "score",
        "unit_en": "score",
        "hard_min": 0.0, "hard_max": 3.0,
        "soft_min": 0.0, "soft_max": 1.0,
        "note": "糖尿病家族風險指標",
    },
    "Age": {
        "display_name": "年齡",
        "english_name": "Age",
        "unit": "歲",
        "unit_en": "years",
        "hard_min": 1, "hard_max": 120,
        "soft_min": 18, "soft_max": 80,
        "note": "年齡",
    },
}


def get_default_patient_values():
    try:
        from src.preprocess import clean_data, fill_missing_values, load_data
    except ImportError:
        from preprocess import clean_data, fill_missing_values, load_data

    df = fill_missing_values(clean_data(load_data()))
    return {feature: float(df[feature].median()) for feature in FEATURE_COLUMNS}

DISCLAIMER = (
    "本報告由機器學習模型自動產生，僅供糖尿病風險「初步篩檢參考」之用，"
    "不能取代醫師之專業診斷與檢驗。若風險分層為中度以上，建議盡快諮詢醫療專業人員。"
)



# 模型 / 路徑工具


def load_model(model_path):
    state_dict = torch.load(model_path, map_location="cpu")
    hidden1 = state_dict["Layer1.weight"].shape[0]
    hidden2 = state_dict["Layer2.weight"].shape[0]
    model = DiabetesModel(hidden1=hidden1, hidden2=hidden2)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_model_path():
    return Path(__file__).resolve().parent.parent / "models" / "diabetes_model.pth"


def get_scaler_path():
    return Path(__file__).resolve().parent.parent / "models" / "scaler.pkl"


def get_output_dir():
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# 輸入階段：分級驗證

def print_reference_ranges():
    print("\n輸入參考範圍（僅供輸入檢查，不等於醫師診斷）：", flush=True)
    for feature in FEATURE_COLUMNS:
        guide = FEATURE_GUIDE[feature]
        print(
            f"- {guide['display_name']} ({feature}) [{guide['unit']}]: "
            f"常見範圍 {guide['soft_min']}~{guide['soft_max']} | {guide['note']}",
            flush=True,
        )


def validate_value(feature, value):
    """回傳 (是否通過硬性檢查, 是否超出常見範圍的警示文字或 None)"""
    guide = FEATURE_GUIDE[feature]
    if not (guide["hard_min"] <= value <= guide["hard_max"]):
        return False, (
            f"{guide['display_name']}={value} 超出生理可能範圍 "
            f"({guide['hard_min']}~{guide['hard_max']} {guide['unit']})，請重新輸入"
        )
    if not (guide["soft_min"] <= value <= guide["soft_max"]):
        warning = (
            f" {guide['display_name']}={value} {guide['unit']} 超出常見參考範圍 "
            f"({guide['soft_min']}~{guide['soft_max']})，請確認輸入是否正確"
        )
        return True, warning
    return True, None


def read_patient_input():
    values = {}
    warnings = []
    print("請輸入病人資料：", flush=True)
    print_reference_ranges()
    for feature in FEATURE_COLUMNS:
        guide = FEATURE_GUIDE[feature]
        prompt = f"{guide['display_name']} ({feature}) [{guide['soft_min']}~{guide['soft_max']} {guide['unit']}]"
        while True:
            raw_value = input(f"{prompt}: ").strip()
            try:
                value = float(raw_value)
            except ValueError:
                print("請輸入有效數字", flush=True)
                continue

            ok, message = validate_value(feature, value)
            if not ok:
                print(message, flush=True)
                continue
            if message:
                print(message, flush=True)
                confirm = input("數值超出常見範圍，是否仍要使用此數值？(y=確認使用 / n=重新輸入): ").strip().lower()
                if confirm not in ("y", "yes"):
                    print("請重新輸入。", flush=True)
                    continue
                warnings.append(message)
            values[feature] = value
            break

    return pd.DataFrame([values], columns=FEATURE_COLUMNS), warnings



# 預測 / 風險分層

def predict_risk(model, scaler, patient_df):
    scaled = transform_input_features(patient_df, scaler)
    tensor = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        probability = float(model(tensor).item())
    return probability, scaled


def risk_tier(probability):
    if probability < 0.3:
        return "低風險", "建議維持健康生活習慣，定期健檢追蹤即可。"
    elif probability < 0.6:
        return "中度風險", "建議於 1-3 個月內安排進一步檢查（如口服葡萄糖耐受試驗）並諮詢醫師。"
    else:
        return "高風險", "建議盡快至醫療院所進行完整糖尿病相關檢查與評估。"



# SHAP 解釋：使用原始單位 + 中文標籤

def plot_custom_waterfall(base_value, prediction_value, ranked, patient_df, output_dir, cjk_ok):
    """手動繪製 waterfall 圖，完全自行控制顏色、正負號與字型，避免 shap.plots.waterfall
    內部符號（特殊負號字元等）在中文字型下渲染成方塊/缺字的問題。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(ranked)
    fig, ax = plt.subplots(figsize=(10, max(4.0, n * 0.65 + 1.8)))

    color_up = "#e54848"    # 紅：推高風險
    color_down = "#3b82f6"  # 藍：降低風險

    y_positions = list(range(n, 0, -1))
    cumulative = base_value
    labels = []

    for (feature, effect), y in zip(ranked, y_positions):
        start = cumulative
        end = cumulative + effect
        left = min(start, end)
        width = abs(effect)
        color = color_up if effect > 0 else color_down

        ax.barh(y, width, left=left, height=0.6, color=color, edgecolor="white", linewidth=0.5, zorder=3)

        sign = "+" if effect >= 0 else "-"  # 使用 ASCII 減號，避免 unicode 負號缺字問題
        label_x = max(start, end) + (abs(base_value - prediction_value) * 0.02 + 0.004)
        ax.text(
            label_x, y, f"{sign}{abs(effect):.3f}",
            va="center", ha="left", fontsize=10, color=color, fontweight="bold", zorder=4,
        )

        guide = FEATURE_GUIDE[feature]
        value = patient_df.iloc[0][feature]
        name = guide["display_name"] if cjk_ok else guide["english_name"]
        unit = guide["unit"] if cjk_ok else guide["unit_en"]
        labels.append(f"{value:g} = {name}\n({unit})")

        cumulative = end

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_ylim(0.3, n + 1.2)

    x_min = min(0, base_value, prediction_value) - 0.04
    x_max = max(base_value, prediction_value) + 0.08
    ax.set_xlim(x_min, x_max)

    ax.axvline(base_value, color="#999999", linestyle="--", linewidth=1, zorder=2)
    ax.axvline(prediction_value, color="#222222", linestyle="-", linewidth=1.3, zorder=2)

    base_label = f"E[f(x)] = {base_value:.3f}"
    pred_label = f"f(x) = {prediction_value:.3f}"
    ax.text(base_value, n + 0.6, base_label, ha="center", va="bottom", fontsize=10, color="#555555")
    ax.text(prediction_value, 0.35, pred_label, ha="center", va="top", fontsize=11, fontweight="bold", color="#222222")

    xlabel = "預測風險機率" if cjk_ok else "Predicted Risk Probability"
    title = "各項指標對糖尿病風險預測的影響 (SHAP)" if cjk_ok else "Feature Contribution to Diabetes Risk Prediction (SHAP)"
    legend_up = "推高風險" if cjk_ok else "Increases risk"
    legend_down = "降低風險" if cjk_ok else "Decreases risk"

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=14, pad=14)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", zorder=0)
    ax.legend(
        handles=[Patch(color=color_up, label=legend_up), Patch(color=color_down, label=legend_down)],
        loc="lower right", fontsize=9, frameon=False,
    )

    plt.tight_layout()
    image_path = output_dir / "predict_shap_waterfall.png"
    plt.savefig(image_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return image_path


def explain_prediction(model, background_scaled, sample_scaled, patient_df, feature_names, output_dir):
    import shap

    cjk_ok = configure_cjk_font()

    background_df = pd.DataFrame(background_scaled, columns=feature_names)
    sample_df = pd.DataFrame(sample_scaled, columns=feature_names)

    def predict_fn(data):
        if hasattr(data, "to_numpy"):
            data = data.to_numpy()
        tensor = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            return model(tensor).cpu().numpy().reshape(-1)

    explainer = shap.Explainer(predict_fn, background_df)
    shap_values = explainer(sample_df)

    feature_effects = shap_values.values[0]
    base_value = float(shap_values.base_values[0])
    prediction_value = float(feature_effects.sum() + base_value)

    ranked = sorted(zip(feature_names, feature_effects), key=lambda item: abs(item[1]), reverse=True)

    image_path = plot_custom_waterfall(base_value, prediction_value, ranked, patient_df, output_dir, cjk_ok)

    return {
        "base_value": base_value,
        "prediction_value": prediction_value,
        "ranked": ranked,
        "image_path": image_path,
    }


def summarize_shap(ranked, top_n=3):
    pushing_up = [(f, v) for f, v in ranked if v > 0][:top_n]
    pushing_down = [(f, v) for f, v in ranked if v < 0][:top_n]

    up_str = "、".join(f"{FEATURE_GUIDE[f]['display_name']}（影響值 +{v:.3f}）" for f, v in pushing_up) or "無明顯因子"
    down_str = "、".join(f"{FEATURE_GUIDE[f]['display_name']}（影響值 {v:.3f}）" for f, v in pushing_down) or "無明顯因子"

    return up_str, down_str

# 文字報告

def generate_text_report(patient_df, warnings, probability, tier, tier_advice, shap_result, image_path, output_dir):
    up_str, down_str = summarize_shap(shap_result["ranked"])
    lines = []
    lines.append("=== 糖尿病風險預測報告 ===")
    lines.append(f"產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("【病人輸入資料】")
    for feature in FEATURE_COLUMNS:
        guide = FEATURE_GUIDE[feature]
        value = patient_df.iloc[0][feature]
        lines.append(f"- {guide['display_name']}: {value} {guide['unit']}")

    if warnings:
        lines.append("")
        lines.append("【輸入警示】")
        for w in warnings:
            lines.append(f"- {w}")

    lines.append("")
    lines.append("【風險預測結果】")
    lines.append(f"風險分數 (機率): {probability:.4f}")
    lines.append(f"風險分層: {tier}")
    lines.append(f"建議: {tier_advice}")

    lines.append("")
    lines.append("【SHAP 解釋摘要】")
    lines.append(f"基準值 E[f(x)]: {shap_result['base_value']:.4f}")
    lines.append(f"模型輸出 f(x): {shap_result['prediction_value']:.4f}")
    lines.append(f"主要推高風險的因子: {up_str}")
    lines.append(f"主要降低風險的因子: {down_str}")
    lines.append("")
    lines.append("各項指標詳細影響值（正值=推高風險，負值=降低風險）:")
    for feature, value in shap_result["ranked"]:
        guide = FEATURE_GUIDE[feature]
        lines.append(f"- {guide['display_name']}: {value:+.4f}")

    lines.append("")
    lines.append(f"對應圖檔: {image_path.name}")
    lines.append("")
    lines.append("【免責聲明】")
    lines.append(DISCLAIMER)

    report_path = output_dir / "predict_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

def main():
    print("=== Diabetes Risk Prediction ===", flush=True)

    model_path = get_model_path()
    scaler_path = get_scaler_path()

    if not model_path.exists():
        print(f"找不到模型檔案: {model_path}", flush=True)
        return

    if not scaler_path.exists():
        print(f"找不到 scaler 檔案: {scaler_path}", flush=True)
        return

    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    patient_df, warnings = read_patient_input()
    probability, scaled_sample = predict_risk(model, scaler, patient_df)
    tier, tier_advice = risk_tier(probability)

    print(f"\n預測結果: {tier}", flush=True)
    print(f"風險分數: {probability:.4f}", flush=True)
    print(f"建議: {tier_advice}", flush=True)

    X_train_tensor, _, _, _, _ = prepare_data(use_saved_scaler=True)
    background_scaled = X_train_tensor[:50].cpu().numpy()

    output_dir = get_output_dir()
    shap_result = explain_prediction(
        model, background_scaled, scaled_sample, patient_df, FEATURE_COLUMNS, output_dir
    )

    up_str, down_str = summarize_shap(shap_result["ranked"])
    print("\nSHAP 解釋：", flush=True)
    print(f"E[f(x)] 基準值: {shap_result['base_value']:.4f}", flush=True)
    print(f"f(x) 模型輸出: {shap_result['prediction_value']:.4f}", flush=True)
    print(f"主要推高風險的因子: {up_str}", flush=True)
    print(f"主要降低風險的因子: {down_str}", flush=True)

    report_path = generate_text_report(
        patient_df, warnings, probability, tier, tier_advice, shap_result, shap_result["image_path"], output_dir
    )

    print(f"\n圖檔已輸出至: {shap_result['image_path']}", flush=True)
    print(f"文字報告已輸出至: {report_path}", flush=True)
    print(f"\n{DISCLAIMER}", flush=True)


if __name__ == "__main__":
    main()
