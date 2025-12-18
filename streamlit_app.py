import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="YouTube Comments Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Sentiment Dashboard – YouTube Comments")
st.write(
    "Dashboard ini memanggil endpoint FastAPI `/analyze` untuk mengambil hasil inferensi "
    "terbaru (berdasarkan file `data/raw/comments_*.csv`)."
)

api_url_default = "http://localhost:8000"
api_url = st.text_input("Base URL FastAPI", api_url_default).rstrip("/")
preview_rows = st.slider(
    "Jumlah komentar yang ditampilkan",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
)

st.divider()

fetch_button = st.button("Ambil & Tampilkan Hasil Inferensi", type="primary")

@st.cache_data(show_spinner=False)
def _fetch_inference(base_url: str) -> Dict[str, Any]:
    endpoint = f"{base_url}/analyze"
    response = requests.post(endpoint, timeout=120)
    response.raise_for_status()
    return response.json()


def _format_percentage(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return "n/a"

if fetch_button:
    if not api_url:
        st.error("Base URL tidak boleh kosong.")
    else:
        with st.spinner("Memanggil API /analyze ..."):
            try:
                payload = _fetch_inference(api_url)
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Request gagal: {exc}\n{detail}")
            except Exception as exc:
                st.error(f"Gagal memanggil API: {exc}")
            else:
                st.success(
                    f"Berhasil memuat {payload.get('total_comments', 0)} komentar dari "
                    f"{payload.get('source_file', 'n/a')}"
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Video ID", payload.get("video_id") or "(tidak tersedia)")
                with col2:
                    st.metric(
                        "Total Komentar",
                        payload.get("total_comments", 0),
                    )
                with col3:
                    accuracy = payload.get("accuracy")
                    st.metric(
                        "Model Accuracy",
                        f"{accuracy:.2%}" if isinstance(accuracy, (int, float)) else "n/a",
                    )

                model_metrics = payload.get("metrics") or {}
                if not model_metrics and isinstance(accuracy, (int, float)):
                    model_metrics = {"accuracy": accuracy}

                if model_metrics:
                    st.subheader("Model Evaluation Metrics")

                    metric_items = [
                        (key.replace("_", " ").title(), value)
                        for key, value in sorted(model_metrics.items())
                        if isinstance(value, (int, float))
                    ]

                    if metric_items:
                        for start_idx in range(0, len(metric_items), 3):
                            cols = st.columns(min(3, len(metric_items) - start_idx))
                            for (metric_name, metric_value), col in zip(
                                metric_items[start_idx : start_idx + len(cols)], cols
                            ):
                                col.metric(metric_name, _format_percentage(metric_value))

                        metrics_table = pd.DataFrame(
                            [
                                {
                                    "Metric": metric_name,
                                    "Value": metric_value,
                                    "Value (%)": _format_percentage(metric_value),
                                }
                                for metric_name, metric_value in metric_items
                            ]
                        ).set_index("Metric")

                        st.dataframe(metrics_table, use_container_width=True)

                st.subheader("Ringkasan Sentimen")
                breakdown_df = pd.DataFrame(payload.get("breakdown", []))
                if not breakdown_df.empty:
                    breakdown_df["ratio_pct"] = breakdown_df["ratio"].apply(lambda x: f"{x:.1%}")
                    st.dataframe(
                        breakdown_df[["label_text", "count", "ratio_pct"]]
                        .rename(
                            columns={
                                "label_text": "Label",
                                "count": "Jumlah",
                                "ratio_pct": "Rasio",
                            }
                        )
                        .set_index("Label"),
                        use_container_width=True,
                    )
                    chart_data = breakdown_df.set_index("label_text")["count"]
                    st.bar_chart(chart_data, use_container_width=True)
                else:
                    st.info("Tidak ada ringkasan yang dapat ditampilkan.")

                st.subheader("Detil Prediksi Komentar")
                predictions = payload.get("predictions", [])
                if predictions:
                    df_pred = pd.DataFrame(predictions)
                    df_pred = df_pred.rename(
                        columns={
                            "label": "Label",
                            "label_text": "Sentiment",
                            "confidence": "Confidence",
                            "text": "Comment",
                        }
                    )
                    df_pred["Confidence"] = df_pred["Confidence"].apply(
                        lambda x: float(x) if pd.notna(x) else None
                    )
                    st.dataframe(
                        df_pred[["Sentiment", "Confidence", "Comment"]].head(preview_rows),
                        use_container_width=True,
                    )
                    with st.expander("Lihat JSON mentah"):
                        st.code(json.dumps(predictions[:preview_rows], indent=2, ensure_ascii=False))
                else:
                    st.info("Tidak ada data prediksi.")

st.caption(
    "Pastikan server FastAPI berjalan dan sudah memiliki file `comments_*.csv` terbaru "
    "di folder `data/raw/`."
)
