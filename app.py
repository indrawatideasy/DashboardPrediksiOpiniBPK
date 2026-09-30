import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# Configuration & Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analisis Opini Pemda Sumsel",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Analisis Kinerja & Opini Pemda (Sumatera Selatan)")

# ---------------------------------------------------------
# Sidebar: Upload File & Filter Radio Buttons
# ---------------------------------------------------------
st.sidebar.header("📁 Filter Data")

uploaded_file = st.sidebar.file_uploader("Unggah File Excel/CSV", type=["xlsx", "xls", "csv"])

# Pemetaan Indeks Opini
OPINI_MAP = {
    1: "WDP",
    2: "WTP PSH",
    3: "WTP"
}

# Daftar Pemda dan Tahun (2021-2025)
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
            # Generate opini aktual Opini Y (1: WDP, 2: WTP PSH, 3: WTP)
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
                "Opini Y": OPINI_MAP[opini_code],
                "Kluster": np.random.choice(["Kluster 1 (Kinerja Tinggi)", "Kluster 2 (Sedang)", "Kluster 3 (Perlu Perhatian)"])
            })
    return pd.DataFrame(records)

# Load Data
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("File Excel berhasil diunggah!")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file: {e}")
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

# Mapping Kode Angka untuk Opini Y
if "Opini Y_Kode" not in df.columns:
    if set(df["Opini Y"].dropna().unique()).issubset({1, 2, 3}):
        df["Opini Y_Kode"] = df["Opini Y"].astype(int)
        df["Opini Y"] = df["Opini Y_Kode"].map(OPINI_MAP)
    else:
        reverse_map = {"WDP": 1, "WTP PSH": 2, "WTP": 3}
        df["Opini Y_Kode"] = df["Opini Y"].map(reverse_map).fillna(3).astype(int)

if "Kluster" not in df.columns:
    df["Kluster"] = np.random.choice(["Kluster 1", "Kluster 2", "Kluster 3"], size=len(df))

# ---------------------------------------------------------
# ALGORITMA LOGISTIC REGRESSION UNTUK PREDIKSI OPINI BPK
# Variabel yang Digunakan: IKF, DCC, solvabilitas, likuiditas, b_peg, b_mdl, TLRHP Y-1_Sesuai, %lunas_rugi(t-1)
# ---------------------------------------------------------
feature_cols = [
    "IKF", 
    "DCC", 
    "solvabilitas", 
    "likuiditas", 
    "b_peg", 
    "b_mdl", 
    "TLRHP Y-1_Sesuai", 
    "%lunas_rugi(t-1)"
]

available_features = [col for col in feature_cols if col in df.columns]

model_accuracy = None
class_report = None

if len(available_features) == len(feature_cols) and "Opini Y_Kode" in df.columns:
    df_clean = df.dropna(subset=available_features + ["Opini Y_Kode"]).copy()
    
    if len(df_clean) > 5:
        X = df_clean[available_features]
        y = df_clean["Opini Y_Kode"].astype(int)
        
        # Pelatihan Logistic Regression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        # Prediksi Seluruh Dataset menggunakan Fitur Tertentu
        preds_code = model.predict(df[available_features].fillna(0))
        df["Prediksi_Kode"] = preds_code
        df["Prediksi Opini BPK (ML)"] = df["Prediksi_Kode"].map(OPINI_MAP)
        
        # Evaluasi Model
        y_pred = model.predict(X)
        model_accuracy = accuracy_score(y, y_pred)
        class_report = classification_report(y, y_pred, target_names=["WDP (1)", "WTP PSH (2)", "WTP (3)"], output_dict=True)
    else:
        df["Prediksi Opini BPK (ML)"] = df["Opini Y"]
else:
    st.warning(f"Variabel untuk Logistic Regression tidak lengkap. Diperlukan: {', '.join(feature_cols)}")
    df["Prediksi Opini BPK (ML)"] = df["Opini Y"]

df_filtered = df.copy()

# ---------------------------------------------------------
# SIDEBAR RADIO BUTTON FILTERS
# ---------------------------------------------------------
if "Tahun" in df.columns:
    available_years = ["Semua Tahun"] + sorted([str(y) for y in df["Tahun"].dropna().unique()])
    selected_year = st.sidebar.radio("🗓️ Pilih Tahun:", options=available_years, index=0)
    
    if selected_year != "Semua Tahun":
        try:
            df_filtered = df_filtered[df_filtered["Tahun"] == int(selected_year)]
        except ValueError:
            df_filtered = df_filtered[df_filtered["Tahun"] == selected_year]

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
    "🎯 Kluster Pemda"
])

# ---------------------------------------------------------
# TAB 1: RINGKASAN INDIKATOR & KORELASI BELANJA
# ---------------------------------------------------------
with tab1:
    st.subheader("📌 Ringkasan Indikator Keuangan & Kepatuhan")

    # Fungsi penolong mengambil nilai aktual
    def get_actual_val(dataframe, col_name):
        if col_name not in dataframe.columns or dataframe.empty:
            return "-"
        vals = dataframe[col_name].dropna()
        if vals.empty:
            return "-"
        if len(vals) == 1:
            return f"{vals.iloc[0]:.2f}"
        else:
            return f"{vals.iloc[-1]:.2f}"

    # Baris 1 KPI: Indikator Keuangan & Fiskal
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        val_solv = get_actual_val(df_filtered, "solvabilitas")
        st.metric("Solvabilitas (Aktual)", val_solv)
    with m2:
        val_lik = get_actual_val(df_filtered, "likuiditas")
        st.metric("Likuiditas (Aktual)", val_lik)
    with m3:
        val_dcc = df_filtered["DCC"].mean() if "DCC" in df_filtered.columns and not df_filtered.empty else None
        st.metric("Days Cash Coverage (DCC)", f"{val_dcc:.1f} Hari" if pd.notnull(val_dcc) else "-")
    with m4:
        val_ikf = df_filtered["IKF"].mean() if "IKF" in df_filtered.columns and not df_filtered.empty else None
        st.metric("Indeks Kemampuan Fiskal (IKF)", f"{val_ikf:.3f}" if pd.notnull(val_ikf) else "-")

    st.markdown("<br>", unsafe_allow_html=True)

    # Baris 2 KPI: Kepatuhan & Status Opini BPK (Aktual vs Prediksi ML)
    m5, m6, m7, m8 = st.columns(4)
    with m5:
        val_tlrhp = df_filtered["TLRHP Y-1_Sesuai"].mean() if "TLRHP Y-1_Sesuai" in df_filtered.columns and not df_filtered.empty else None
        st.metric("Penyelesaian TLRHP (%)", f"{val_tlrhp:.2f}%" if pd.notnull(val_tlrhp) else "-")
    with m6:
        val_rugi = df_filtered["%lunas_rugi(t-1)"].mean() if "%lunas_rugi(t-1)" in df_filtered.columns and not df_filtered.empty else None
        st.metric("Penyelesaian Ganti Rugi (%)", f"{val_rugi:.2f}%" if pd.notnull(val_rugi) else "-")
    with m7:
        # Opini BPK Aktual (Opini Y)
        if not df_filtered.empty and "Opini Y" in df_filtered.columns:
            opini_aktual = df_filtered["Opini Y"].iloc[0] if len(df_filtered) == 1 else df_filtered["Opini Y"].mode()[0]
        else:
            opini_aktual = "-"
        st.metric("Opini BPK Aktual (Opini Y)", opini_aktual)
    with m8:
        # Prediksi Opini BPK dari Model Logistic Regression
        if not df_filtered.empty and "Prediksi Opini BPK (ML)" in df_filtered.columns:
            opini_pred = df_filtered["Prediksi Opini BPK (ML)"].iloc[0] if len(df_filtered) == 1 else df_filtered["Prediksi Opini BPK (ML)"].mode()[0]
        else:
            opini_pred = "-"
        st.metric("Prediksi Opini BPK (ML)", opini_pred)

    # Komparasi Detail & Performa Model Logistic Regression
    if model_accuracy is not None:
        with st.expander("ℹ️️ Perbandingan Detail: Opini Aktual vs Prediksi Logistic Regression"):
            st.write(f"**Akurasi Logistic Regression (Model Training):** {model_accuracy * 100:.2f}%")
            
            col_exp1, col_exp2 = st.columns([1, 2])
            with col_exp1:
                st.markdown("**Laporan Klasifikasi:**")
                if class_report:
                    st.dataframe(pd.DataFrame(class_report).transpose().style.format("{:.2f}"))
            
            with col_exp2:
                st.markdown("**Tabel Perbandingan Opini Aktual (Y) vs Prediksi (ML):**")
                show_cols = ["Pemda", "Tahun", "Opini Y", "Prediksi Opini BPK (ML)"]
                existing_show_cols = [c for c in show_cols if c in df_filtered.columns]
                
                df_compare = df_filtered[existing_show_cols].copy()
                df_compare["Status Evaluasi"] = np.where(
                    df_compare["Opini Y"] == df_compare["Prediksi Opini BPK (ML)"], 
                    "✅ Sesuai", 
                    "❌ Beda"
                )
                st.dataframe(df_compare, use_container_width=True)

    st.markdown("---")

    # SECTION SCATTER PLOT KORELASI BELANJA
    st.subheader("🔍 Analisis Scatterplot Korelasi Belanja vs Indikator Keuangan")

    if not df_filtered.empty:
        col_sc1, col_sc2 = st.columns(2)

        indikator_options = ["IKF", "DCC", "solvabilitas", "likuiditas"]

        # 1. Scatterplot Korelasi Belanja Pegawai (b_peg)
        with col_sc1:
            var_peg = st.radio(
                "Pilih Variabel Korelasi Belanja Pegawai:",
                options=indikator_options,
                key="radio_peg",
                horizontal=True
            )
            
            fig_sc_peg = px.scatter(
                df_filtered,
                x="b_peg",
                y=var_peg,
                color="Tahun" if "Tahun" in df_filtered.columns else None,
                hover_data=["Pemda", "Tahun", "Opini Y", "Prediksi Opini BPK (ML)"],
                title=f"Scatterplot: Belanja Pegawai (%) vs {var_peg.capitalize()}",
                labels={"b_peg": "Belanja Pegawai (%)", var_peg: var_peg.capitalize()},
                trendline="ols"
            )
            fig_sc_peg.update_traces(marker=dict(size=10, opacity=0.8))
            st.plotly_chart(fig_sc_peg, use_container_width=True)

        # 2. Scatterplot Korelasi Belanja Modal (b_mdl)
        with col_sc2:
            var_mdl = st.radio(
                "Pilih Variabel Korelasi Belanja Modal:",
                options=indikator_options,
                key="radio_mdl",
                horizontal=True
            )
            
            fig_sc_mdl = px.scatter(
                df_filtered,
                x="b_mdl",
                y=var_mdl,
                color="Tahun" if "Tahun" in df_filtered.columns else None,
                hover_data=["Pemda", "Tahun", "Opini Y", "Prediksi Opini BPK (ML)"],
                title=f"Scatterplot: Belanja Modal (%) vs {var_mdl.capitalize()}",
                labels={"b_mdl": "Belanja Modal (%)", var_mdl: var_mdl.capitalize()},
                trendline="ols"
            )
            fig_sc_mdl.update_traces(marker=dict(size=10, opacity=0.8))
            st.plotly_chart(fig_sc_mdl, use_container_width=True)
    else:
        st.info("Data tidak cukup untuk menampilkan scatterplot korelasi.")

# ---------------------------------------------------------
# TAB 2: TREN INDIKATOR KEUANGAN PER TAHUN
# ---------------------------------------------------------
with tab2:
    st.subheader("📊 Tren Indikator Keuangan per Tahun (2021 - 2025)")
    
    if "Tahun" in df_filtered.columns and not df_filtered.empty:
        df_trend_fin = df_filtered.groupby("Tahun")[["likuiditas", "solvabilitas", "DCC", "IKF"]].mean().reset_index()
        
        col_t1, col_t2 = st.columns(2)
        col_t3, col_t4 = st.columns(2)
        
        with col_t1:
            fig_lik = px.line(
                df_trend_fin, x="Tahun", y="likuiditas", markers=True,
                title="Grafik Likuiditas per Tahun", labels={"likuiditas": "Rasio Likuiditas"}
            )
            fig_lik.update_traces(line_color="#2ecc71", line_width=3)
            st.plotly_chart(fig_lik, use_container_width=True)
            
        with col_t2:
            fig_solv = px.line(
                df_trend_fin, x="Tahun", y="solvabilitas", markers=True,
                title="Grafik Solvabilitas per Tahun", labels={"solvabilitas": "Rasio Solvabilitas"}
            )
            fig_solv.update_traces(line_color="#e74c3c", line_width=3)
            st.plotly_chart(fig_solv, use_container_width=True)
            
        with col_t3:
            fig_dcc = px.line(
                df_trend_fin, x="Tahun", y="DCC", markers=True,
                title="Grafik Days Cash Coverage (DCC) per Tahun", labels={"DCC": "DCC (Hari)"}
            )
            fig_dcc.update_traces(line_color="#9b59b6", line_width=3)
            st.plotly_chart(fig_dcc, use_container_width=True)
            
        with col_t4:
            fig_ikf = px.line(
                df_trend_fin, x="Tahun", y="IKF", markers=True,
                title="Grafik IKF per Tahun", labels={"IKF": "Indeks Kemampuan Fiskal"}
            )
            fig_ikf.update_traces(line_color="#f39c12", line_width=3)
            st.plotly_chart(fig_ikf, use_container_width=True)
    else:
        st.info("Data tidak cukup untuk menampilkan tren indikator keuangan.")

# ---------------------------------------------------------
# TAB 3: GRAFIK KLUSTER PEMDA
# ---------------------------------------------------------
with tab3:
    st.subheader("🎯 Visualisasi Kluster Pemda")
    st.markdown("Pemetaan kelompok Pemda berdasarkan indikator keuangan dan kepatuhan.")
    
    if not df_filtered.empty:
        col_c1, col_c2 = st.columns([3, 1])
        
        with col_c2:
            st.markdown("**Pengaturan Sumbu Grafik:**")
            x_axis = st.selectbox("Sumbu X:", ["IKF", "DCC", "solvabilitas", "likuiditas", "TLRHP Y-1_Sesuai"], index=0)
            y_axis = st.selectbox("Sumbu Y:", ["likuiditas", "solvabilitas", "DCC", "IKF", "%lunas_rugi(t-1)"], index=0)
        
        with col_c1:
            fig_cluster = px.scatter(
                df_filtered,
                x=x_axis,
                y=y_axis,
                color="Kluster" if "Kluster" in df_filtered.columns else "Opini Y",
                symbol="Opini Y",
                hover_data=["Pemda", "Tahun", "Opini Y", "Prediksi Opini BPK (ML)"],
                title=f"Pemetaan Kluster Pemda ({x_axis} vs {y_axis})",
                size="DCC" if "DCC" in df_filtered.columns else None
            )
            fig_cluster.update_traces(marker=dict(size=12, opacity=0.8))
            st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("Data tidak cukup untuk menampilkan pemetaan kluster.")
