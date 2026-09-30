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
    "🎯 Klaster Pemda (K-Means)"
])

# ---------------------------------------------------------
# TAB 1: RINGKASAN INDIKATOR & KORELASI BELANJA
# ---------------------------------------------------------
with tab1:
    st.subheader("📌 Ringkasan Indikator Keuangan & Kepatuhan")

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

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Solvabilitas (Aktual)", get_actual_val(df_filtered, "solvabilitas"))
    with m2:
        st.metric("Likuiditas (Aktual)", get_actual_val(df_filtered, "likuiditas"))
    with m3:
        val_dcc = df_filtered["DCC"].mean() if "DCC" in df_filtered.columns and not df_filtered.empty else None
        st.metric("Days Cash Coverage (DCC)", f"{val_dcc:.1f} Hari" if pd.notnull(val_dcc) else "-")
    with m4:
        val_ikf = df_filtered["IKF"].mean() if "IKF" in df_filtered.columns and not df_filtered.empty else None
        st.metric("Indeks Kemampuan Fiskal (IKF)", f"{val_ikf:.3f}" if pd.notnull(val_ikf) else "-")

    st.markdown("<br>", unsafe_allow_html=True)

    m5, m6, m7, m8 = st.columns(4)
    with m5:
        val_tlrhp = df_filtered["TLRHP Y-1_Sesuai"].mean() if "TLRHP Y-1_Sesuai" in df_filtered.columns and not df_filtered.empty else None
        st.metric("Penyelesaian TLRHP (%)", f"{val_tlrhp:.2f}%" if pd.notnull(val_tlrhp) else "-")
    with m6:
        val_rugi = df_filtered["%lunas_rugi(t-1)"].mean() if "%lunas_rugi(t-1)" in df_filtered.columns and not df_filtered.empty else None
        st.metric("Penyelesaian Ganti Rugi (%)", f"{val_rugi:.2f}%" if pd.notnull(val_rugi) else "-")
    with m7:
        if not df_filtered.empty and "Opini Y" in df_filtered.columns:
            opini_aktual = df_filtered["Opini Y"].iloc[0] if len(df_filtered) == 1 else df_filtered["Opini Y"].mode()[0]
        else:
            opini_aktual = "-"
        st.metric("Opini BPK Aktual (Opini Y)", opini_aktual)
    with m8:
        if not df_filtered.empty and "Prediksi Opini BPK (ML)" in df_filtered.columns:
            opini_pred = df_filtered["Prediksi Opini BPK (ML)"].iloc[0] if len(df_filtered) == 1 else df_filtered["Prediksi Opini BPK (ML)"].mode()[0]
        else:
            opini_pred = "-"
        st.metric("Prediksi Opini BPK (ML)", opini_pred)

    if model_accuracy is not None:
        with st.expander("ℹ️ Detail Model Logistic Regression & Perbandingan Prediksi"):
            st.write(f"**Akurasi Model Training Logistic Regression:** {model_accuracy * 100:.2f}%")
            
            col_exp1, col_exp2 = st.columns([1, 2])
            with col_exp1:
                st.markdown("**Laporan Klasifikasi:**")
                if class_report:
                    st.dataframe(pd.DataFrame(class_report).transpose().style.format("{:.2f}"))
            
            with col_exp2:
                st.markdown("**Tabel Hasil Prediksi vs Opini Aktual:**")
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

    st.subheader("🔍 Analisis Scatterplot Korelasi Belanja vs Indikator Keuangan")

    if not df_filtered.empty:
        col_sc1, col_sc2 = st.columns(2)
        indikator_options = ["IKF", "DCC", "solvabilitas", "likuiditas"]

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
            fig_lik = px.line(df_trend_fin, x="Tahun", y="likuiditas", markers=True, title="Grafik Likuiditas per Tahun")
            fig_lik.update_traces(line_color="#2ecc71", line_width=3)
            st.plotly_chart(fig_lik, use_container_width=True)
            
        with col_t2:
            fig_solv = px.line(df_trend_fin, x="Tahun", y="solvabilitas", markers=True, title="Grafik Solvabilitas per Tahun")
            fig_solv.update_traces(line_color="#e74c3c", line_width=3)
            st.plotly_chart(fig_solv, use_container_width=True)
            
        with col_t3:
            fig_dcc = px.line(df_trend_fin, x="Tahun", y="DCC", markers=True, title="Grafik Days Cash Coverage (DCC) per Tahun")
            fig_dcc.update_traces(line_color="#9b59b6", line_width=3)
            st.plotly_chart(fig_dcc, use_container_width=True)
            
        with col_t4:
            fig_ikf = px.line(df_trend_fin, x="Tahun", y="IKF", markers=True, title="Grafik IKF per Tahun")
            fig_ikf.update_traces(line_color="#f39c12", line_width=3)
            st.plotly_chart(fig_ikf, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: VISUALISASI KLASTER PEMDA (K-MEANS)
# ---------------------------------------------------------
with tab3:
    st.subheader("🎯 Klasterisasi Pemda Menggunakan K-Means")

    # Filter Khusus Tahun di Tab 3
    if "Tahun" in df.columns:
        year_options_tab3 = ["Semua Tahun"] + sorted([str(y) for y in df["Tahun"].dropna().unique()])
        selected_year_tab3 = st.radio(
            "🗓️ Filter Tahun Khusus Klasterisasi:",
            options=year_options_tab3,
            index=0,
            horizontal=True,
            key="radio_tahun_tab3"
        )
        
        if selected_year_tab3 != "Semua Tahun":
            try:
                df_tab3 = df[df["Tahun"] == int(selected_year_tab3)].copy()
            except ValueError:
                df_tab3 = df[df["Tahun"] == selected_year_tab3].copy()
        else:
            df_tab3 = df.copy()
    else:
        df_tab3 = df.copy()

    st.markdown("---")

    # Seluruh variabel individual termasuk Opini Y
    all_individual_vars = [
        "IKF", "DCC", "solvabilitas", "likuiditas", 
        "b_peg", "b_mdl", "TLRHP Y-1_Sesuai", "%lunas_rugi(t-1)", "Opini Y"
    ]
    available_ind_vars = [v for v in all_individual_vars if v in df_tab3.columns]

    col_opt1, col_opt2 = st.columns([2, 1])
    
    with col_opt1:
        # Multiselect variabel individual termasuk Opini Y
        selected_kmeans_vars = st.multiselect(
            "📌 Pilih variabel individual untuk Algoritma K-Means (Pilih minimal 2):",
            options=available_ind_vars,
            default=["IKF", "DCC", "solvabilitas", "likuiditas", "Opini Y"],
            key="multiselect_kmeans_ind"
        )

    with col_opt2:
        n_clusters = st.slider("Jumlah Kluster (k):", min_value=2, max_value=5, value=3, key="slider_k_means")

    if len(selected_kmeans_vars) < 2:
        st.warning("⚠️ Harap pilih **minimal 2 variabel** untuk mengeksekusi model K-Means.")
    else:
        # Menyiapkan variabel numerik khusus untuk kalkulasi K-Means
        kmeans_calc_cols = []
        for v in selected_kmeans_vars:
            if v == "Opini Y":
                kmeans_calc_cols.append("Opini Y_Kode")
            else:
                kmeans_calc_cols.append(v)

        df_km = df_tab3.dropna(subset=kmeans_calc_cols).copy()
        
        if len(df_km) >= n_clusters:
            # 1. Standardisasi Data dari Variabel Terpilih
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_km[kmeans_calc_cols])
            
            # 2. Algoritma K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_km["Cluster_KMeans"] = kmeans.fit_predict(scaled_data)
            df_km["Cluster_Label"] = df_km["Cluster_KMeans"].apply(lambda x: f"Kluster {x+1}")

            st.markdown("---")

            # Menggunakan 2 variabel pertama terpilih sebagai Sumbu X dan Sumbu Y secara otomatis
            x_km = selected_kmeans_vars[0]
            y_km = selected_kmeans_vars[1]

            # Visualisasi Scatterplot Hasil Klasterisasi
            fig_km = px.scatter(
                df_km,
                x=x_km,
                y=y_km,
                color="Cluster_Label",
                hover_data=["Pemda", "Tahun", "Opini Y", "Prediksi Opini BPK (ML)"],
                title=f"Scatterplot Hasil Klasterisasi K-Means ({x_km} vs {y_km})",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_km.update_traces(marker=dict(size=12, opacity=0.85))
            st.plotly_chart(fig_km, use_container_width=True)

            # Tabel Detail Pemetaan Hasil Kluster K-Means
            with st.expander("📋 Tabel Hasil Klasterisasi K-Means Pemda"):
                cols_display = ["Pemda", "Tahun", "Cluster_Label", "Opini Y", "Prediksi Opini BPK (ML)"] + [v for v in selected_kmeans_vars if v != "Opini Y"]
                existing_cols_display = [c for c in cols_display if c in df_km.columns]
                st.dataframe(df_km[existing_cols_display], use_container_width=True)
        else:
            st.info("Jumlah data tidak cukup untuk menjalankan algoritma K-Means dengan k=" + str(n_clusters))
