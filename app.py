import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# Configuration & Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analisis Opini Pemda Sumsel",
    page_icon="📊",
    layout="wide"
)

# Custom CSS untuk Kartu Indikator & Kartu Opini
st.markdown("""
    <style>
    .metric-card-green {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .metric-card-yellow {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 14px;
        font-weight: 600;
        color: #333333;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: bold;
        color: #111111;
    }
    .metric-delta-pos {
        font-size: 13px;
        font-weight: 600;
        color: #155724;
    }
    .metric-delta-neg {
        font-size: 13px;
        font-weight: 600;
        color: #856404;
    }

    /* Kartu Opini BPK (Rata Tengah + Dynamic Background) */
    .opini-card-wtp {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-bottom: 10px;
    }
    .opini-card-wtp-psh {
        background-color: #eef5db;
        border: 2px solid #8a9a86;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-bottom: 10px;
    }
    .opini-card-wdp {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-bottom: 10px;
    }
    .opini-card-default {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-bottom: 10px;
    }
    .opini-title {
        font-size: 15px;
        font-weight: 600;
        color: #495057;
        margin-bottom: 6px;
    }
    .opini-value {
        font-size: 26px;
        font-weight: bold;
        color: #111111;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard Analisis Kinerja & Opini Pemda (Sumatera Selatan)")

# Pemetaan Indeks Opini
OPINI_MAP = {
    1: "WDP",
    2: "WTP PSH",
    3: "WTP"
}

# ---------------------------------------------------------
# Sidebar: Upload File & Filter Radio Buttons
# ---------------------------------------------------------
st.sidebar.header("📁 Filter Data")

uploaded_file = st.sidebar.file_uploader("Unggah File Excel/CSV Data Pemda", type=["xlsx", "xls", "csv"])

LIST_PEMDA = [
    "Banyuasin", "Empatlawang", "Lahat", "Lubuklinggau", "Ma_Enim", 
    "Muba", "Mura", "Muratara", "OI", "OKI", "OKU", "OKUS", 
    "OKUT", "Pagaralam", "Palembang", "Pali", "Prabumulih", "Sumsel"
]
LIST_TAHUN = [2021, 2022, 2023, 2024, 2025]

@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    records = []
    
    for thn in LIST_TAHUN:
        for pmd in LIST_PEMDA:
            opini_code = np.random.choice([1, 2, 3], p=[0.15, 0.25, 0.60])
            records.append({
                "Tahun": thn,
                "Pemda": pmd,
                "IKF": np.round(np.random.uniform(0.2, 1.8), 3),
                "DCC": np.round(np.random.uniform(30, 180), 2),
                "solvabilitas": np.round(np.random.uniform(0.8, 3.5), 2),
                "likuiditas": np.round(np.random.uniform(1.0, 5.0), 2),
                "b_peg": np.round(np.random.uniform(25, 45), 2),
                "b_brg": np.round(np.random.uniform(20, 40), 2),
                "b_mdl": np.round(np.random.uniform(10, 30), 2),
                "TLRHP Y-1_Sesuai": np.round(np.random.uniform(50, 98), 2),
                "%lunas_rugi(t-1)": np.round(np.random.uniform(20, 95), 2),
                "Opini Y_Kode": opini_code,
                "Opini Y": OPINI_MAP[opini_code]
            })
    return pd.DataFrame(records)

# Load Data
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("File Excel/CSV berhasil diunggah!")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file data: {e}")
        df = generate_sample_data()
else:
    st.sidebar.info("Menggunakan sampel data Pemda Sumsel (2021-2025).")
    df = generate_sample_data()

# Standardisasi Kolom Opini Y (Aktual)
if "Opini Y" not in df.columns:
    if "Opini Y-1" in df.columns:
        df["Opini Y"] = df["Opini Y-1"]
    elif "Opini Y_Kode" in df.columns:
        df["Opini Y"] = df["Opini Y_Kode"].map(OPINI_MAP)
    else:
        df["Opini Y_Kode"] = np.random.choice([1, 2, 3], size=len(df), p=[0.15, 0.25, 0.60])
        df["Opini Y"] = df["Opini Y_Kode"].map(OPINI_MAP)

if "Opini Y_Kode" not in df.columns:
    if set(df["Opini Y"].dropna().unique()).issubset({1, 2, 3}):
        df["Opini Y_Kode"] = df["Opini Y"].astype(int)
        df["Opini Y"] = df["Opini Y_Kode"].map(OPINI_MAP)
    else:
        reverse_map = {"WDP": 1, "WTP PSH": 2, "WTP": 3}
        df["Opini Y_Kode"] = df["Opini Y"].map(reverse_map).fillna(3).astype(int)

# ---------------------------------------------------------
# ALGORITMA LOGISTIC REGRESSION UNTUK PREDIKSI OPINI BPK
# ---------------------------------------------------------
feature_cols = [
    "IKF", "DCC", "solvabilitas", "likuiditas", 
    "b_peg", "b_mdl", "TLRHP Y-1_Sesuai", "%lunas_rugi(t-1)"
]

available_features = [col for col in feature_cols if col in df.columns]

model_accuracy = None
class_report = None

if len(available_features) == len(feature_cols) and "Opini Y_Kode" in df.columns:
    df_clean = df.dropna(subset=available_features + ["Opini Y_Kode"]).copy()
    
    if len(df_clean) > 5:
        X = df_clean[available_features]
        y = df_clean["Opini Y_Kode"].astype(int)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        preds_code = model.predict(df[available_features].fillna(0))
        df["Prediksi_Kode"] = preds_code
        df["Prediksi Opini BPK (ML)"] = df["Prediksi_Kode"].map(OPINI_MAP)
        
        y_pred = model.predict(X)
        model_accuracy = accuracy_score(y, y_pred)
        class_report = classification_report(y, y_pred, target_names=["WDP (1)", "WTP PSH (2)", "WTP (3)"], output_dict=True)
    else:
        df["Prediksi Opini BPK (ML)"] = df["Opini Y"]
else:
    df["Prediksi Opini BPK (ML)"] = df["Opini Y"]

df_filtered = df.copy()

# ---------------------------------------------------------
# SIDEBAR RADIO BUTTON FILTERS (GLOBAL)
# ---------------------------------------------------------
selected_year = "Semua Tahun"
if "Tahun" in df.columns:
    available_years = ["Semua Tahun"] + sorted([str(y) for y in df["Tahun"].dropna().unique()])
    selected_year = st.sidebar.radio("🗓️ Pilih Tahun:", options=available_years, index=0)
    
    if selected_year != "Semua Tahun":
        try:
            df_filtered = df_filtered[df_filtered["Tahun"] == int(selected_year)]
        except ValueError:
            df_filtered = df_filtered[df_filtered["Tahun"] == selected_year]

selected_pemda = "Semua Pemda"
if "Pemda" in df.columns:
    available_pemda = ["Semua Pemda"] + sorted(list(df["Pemda"].dropna().unique()))
    selected_pemda = st.sidebar.radio("🏛️ Pilih Pemda:", options=available_pemda, index=0)
    
    if selected_pemda != "Semua Pemda":
        df_filtered = df_filtered[df_filtered["Pemda"] == selected_pemda]

# ---------------------------------------------------------
# SETUP TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📌 Ringkasan & Korelasi Belanja", 
    "📈 Tren Indikator Keuangan", 
    "🎯 Klaster Pemda (K-Means)"
])

# ---------------------------------------------------------
# TAB 1: RINGKASAN INDIKATOR & KORELASI BELANJA
# ---------------------------------------------------------
with tab1:
    st.subheader("📌 Ringkasan Indikator Keuangan & Kepatuhan")

    def compute_metric_with_delta(col_name, is_dcc=False):
        if col_name not in df.columns or df_filtered.empty:
            return None, None, True
        
        if selected_pemda != "Semua Pemda":
            df_curr_pemda = df[df["Pemda"] == selected_pemda]
        else:
            df_curr_pemda = df
            
        if selected_year != "Semua Tahun":
            try:
                curr_y = int(selected_year)
            except ValueError:
                curr_y = selected_year
            prev_y = curr_y - 1 if isinstance(curr_y, int) else None
        else:
            years = sorted(df_curr_pemda["Tahun"].dropna().unique())
            curr_y = years[-1] if years else None
            prev_y = years[-2] if len(years) > 1 else None

        df_curr = df_curr_pemda[df_curr_pemda["Tahun"] == curr_y] if curr_y else df_filtered
        val_curr = df_curr[col_name].mean() if not df_curr.empty else None

        val_prev = None
        if prev_y is not None:
            df_prev = df_curr_pemda[df_curr_pemda["Tahun"] == prev_y]
            if not df_prev.empty:
                val_prev = df_prev[col_name].mean()

        delta = (val_curr - val_prev) if (val_curr is not None and val_prev is not None) else None
        
        if is_dcc:
            is_good = (val_curr >= 30.0) if val_curr is not None else True
        else:
            is_good = (delta >= 0) if delta is not None else True

        return val_curr, delta, is_good

    def render_custom_card(title, value_str, delta, is_good, unit=""):
        card_class = "metric-card-green" if is_good else "metric-card-yellow"
        
        if delta is not None:
            icon = "⬆️" if delta >= 0 else "⬇️"
            delta_class = "metric-delta-pos" if delta >= 0 else "metric-delta-neg"
            delta_str = f"<div class='{delta_class}'>{icon} {abs(delta):.2f}{unit} dibanding thn sebelumnya</div>"
        else:
            delta_str = "<div style='font-size:12px; color:#666;'>Data thn sebelumnya (-)</div>"
            
        status_tag = "🟢 Baik" if is_good else "⚠️ Memerlukan Perhatian"

        html_code = f"""
        <div class="{card_class}">
            <div class="metric-title">{title} <span style="float:right; font-size:11px;">{status_tag}</span></div>
            <div class="metric-value">{value_str}</div>
            {delta_str}
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)

    def render_opini_card(title, opini_val):
        opini_clean = str(opini_val).strip().upper()
        if "WTP PSH" in opini_clean:
            card_class = "opini-card-wtp-psh"
        elif "WTP" in opini_clean:
            card_class = "opini-card-wtp"
        elif "WDP" in opini_clean:
            card_class = "opini-card-wdp"
        else:
            card_class = "opini-card-default"

        html_code = f"""
        <div class="{card_class}">
            <div class="opini-title">{title}</div>
            <div class="opini-value">{opini_val}</div>
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)

    # BARIS 1: Solvabilitas, Likuiditas, Days Cash Coverage
    b1_c1, b1_c2, b1_c3 = st.columns(3)
    with b1_c1:
        v, d, g = compute_metric_with_delta("solvabilitas")
        v_str = f"{v:.2f}" if v is not None else "-"
        render_custom_card("Solvabilitas (Aktual)", v_str, d, g)
        
    with b1_c2:
        v, d,
