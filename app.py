from pathlib import Path

import streamlit as st

from src.predict import (
	DISCLAIMER,
	FEATURE_COLUMNS,
	FEATURE_GUIDE,
	explain_prediction,
	generate_text_report,
	get_model_path,
	get_output_dir,
	get_scaler_path,
	load_model,
	load_scaler,
	get_default_patient_values,
	predict_risk,
	prepare_data,
	risk_tier,
)


st.set_page_config(
	page_title="Diabetes AI System",
	page_icon="🩺",
	layout="wide",
)


def get_reference_rows():
	rows = []
	for feature in FEATURE_COLUMNS:
		guide = FEATURE_GUIDE[feature]
		rows.append(
			{
				"欄位": guide["display_name"],
				"英文欄位": feature,
				"單位": guide["unit"],
				"常見參考範圍": f'{guide["soft_min"]} ~ {guide["soft_max"]}',
				"說明": guide["note"],
			}
		)
	return rows


@st.cache_resource(show_spinner=False)
def load_artifacts():
	model_path = get_model_path()
	scaler_path = get_scaler_path()

	if not model_path.exists():
		raise FileNotFoundError(f"找不到模型檔案: {model_path}")
	if not scaler_path.exists():
		raise FileNotFoundError(f"找不到 scaler 檔案: {scaler_path}")

	model = load_model(model_path)
	scaler = load_scaler(scaler_path)
	return model, scaler


def build_patient_dataframe(values):
	import pandas as pd

	return pd.DataFrame([values], columns=FEATURE_COLUMNS)


@st.cache_data(show_spinner=False)
def get_default_inputs():
	return get_default_patient_values()


def main():
	st.title("Diabetes AI System")
	st.caption("糖尿病風險初步篩檢與 SHAP 解釋介面")

	col_left, col_right = st.columns([1.05, 0.95], gap="large")

	with col_left:
		st.subheader("病人資料輸入")
		st.markdown(
			"請輸入病人的 8 項特徵。超出常見範圍時，系統會提示，但仍可送出。"
		)
		default_inputs = get_default_inputs()

		with st.form("patient_form"):
			inputs = {}
			for feature in FEATURE_COLUMNS:
				guide = FEATURE_GUIDE[feature]
				inputs[feature] = st.number_input(
					f'{guide["display_name"]} ({feature})',
					value=float(default_inputs[feature]),
					min_value=float(guide["hard_min"]),
					max_value=float(guide["hard_max"]),
					step=0.1 if feature == "DiabetesPedigreeFunction" else 1.0,
					help=f'常見參考範圍: {guide["soft_min"]} ~ {guide["soft_max"]} {guide["unit"]}',
				)

			submitted = st.form_submit_button("開始預測")

		st.markdown("---")
		st.subheader("參考範圍")
		st.dataframe(get_reference_rows(), use_container_width=True, hide_index=True)

	with col_right:
		st.subheader("預測結果")
		st.info(DISCLAIMER)

		if submitted:
			try:
				model, scaler = load_artifacts()
				patient_df = build_patient_dataframe(inputs)
				probability, scaled_sample = predict_risk(model, scaler, patient_df)
				tier, tier_advice = risk_tier(probability)

				st.metric("風險分數", f"{probability:.4f}")
				st.metric("風險分層", tier)
				st.write(tier_advice)

				X_train_tensor, _, _, _, _ = prepare_data(use_saved_scaler=True)
				background_scaled = X_train_tensor[:50].cpu().numpy()

				output_dir = get_output_dir()
				shap_result = explain_prediction(
					model,
					background_scaled,
					scaled_sample,
					patient_df,
					FEATURE_COLUMNS,
					output_dir,
				)

				st.subheader("SHAP 解釋圖")
				st.image(str(shap_result["image_path"]), use_container_width=True)

				st.subheader("重要影響因子")
				for feature, value in shap_result["ranked"][:5]:
					st.write(f'- {FEATURE_GUIDE[feature]["display_name"]}: {value:+.4f}')

				report_path = generate_text_report(
					patient_df,
					[],
					probability,
					tier,
					tier_advice,
					shap_result,
					shap_result["image_path"],
					output_dir,
				)
				report_text = report_path.read_text(encoding="utf-8")
				st.success(f"報告已輸出: {report_path.name}")

				st.subheader("文字報告")
				st.caption("下方內容可直接全選複製，或下載成 txt 檔。")
				st.text_area(
					"報告內容",
					value=report_text,
					height=520,
				)
				st.download_button(
					label="下載文字報告",
					data=report_text,
					file_name=report_path.name,
					mime="text/plain",
				)

			except Exception as exc:
				st.error(f"執行失敗: {exc}")

		else:
			st.write("按下「開始預測」後，這裡會顯示風險分數、風險分層、SHAP 圖與文字報告。")


if __name__ == "__main__":
	main()
