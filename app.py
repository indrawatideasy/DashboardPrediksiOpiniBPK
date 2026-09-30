import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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

# Daftar Pemda dan Tahun sesuai entitas di Sumsel (2021-2025)
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
    opini_list = ["WTP", "WDP", "TW", "TMP"]
    
    for thn in LIST_TAHUN:
        for pmd in LIST_PEMDA:
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
                "Opini Y-1": np.random.choice(opini_list, p=[0.75, 0.15, 0.06, 0.04]),
                "Prediksi Opini BPK": np.random.choice(opini_list, p=[0.80, 0.12, 0.05, 0.03]),
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

# Penyesuaian otomatis jika kolom turunan belum ada
if "Opini Y-1" not in df.columns:
    df["Opini Y-1"] = np.random.choice(["WTP", "WDP", "TW", "TMP"], size=len(df), p=[0.7, 0.2, 0.07, 0.03])
if "Prediksi Opini BPK" not in df.columns:
    df["Prediksi Opini BPK"] = df["Opini Y-1"]
if "b_brg" not in df.columns:
    df["b_brg"] = np.round(np.random.uniform(20, 40, size=len(df)), 2)
if "Kluster" not in df.columns:
    df["Kluster"] = np.random.choice(["Kluster 1", "Kluster 2", "Kluster 3"], size=len(df))

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

    # Baris 1 KPI: Indikator Keuangan & Fiskal
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        val_solv = df_filtered["solvabilitas"].mean()
        st.metric("Solvabilitas (Rata-rata)", f"{val_solv:.2f}" if pd.notnull(val_solv) else "-")
    with m2:
        val_lik = df_filtered["likuiditas"].mean()
        st.metric("Likuiditas (Rata-rata)", f"{val_lik:.2f}" if pd.notnull(val_lik) else "-")
    with m3:
        val_dcc = df_filtered["DCC"].mean()
        st.metric("Days Cash Coverage (DCC)", f"{val_dcc:.1f} Hari" if pd.notnull(val_dcc) else "-")
    with m4:
        val_ikf = df_filtered["IKF"].mean()
        st.metric("Indeks Kemampuan Fiskal (IKF)", f"{val_ikf:.3f}" if pd.notnull(val_ikf) else "-")

    st.markdown("<br>", unsafe_allow_html=True)

    # Baris 2 KPI: Kepatuhan & Status Opini BPK
    m5, m6, m7, m8 = st.columns(4)
    with m5:
        val_tlrhp = df_filtered["TLRHP Y-1_Sesuai"].mean()
        st.metric("Penyelesaian TLRHP (%)", f"{val_tlrhp:.2f}%" if pd.notnull(val_tlrhp) else "-")
    with m6:
        val_rugi = df_filtered["%lunas_rugi(t-1)"].mean()
        st.metric("Penyelesaian Ganti Rugi (%)", f"{val_rugi:.2f}%" if pd.notnull(val_rugi) else "-")
    with m7:
        opini_aktual = df_filtered["Opini Y-1"].mode()[0] if not df_filtered.empty else "-"
        st.metric("Opini BPK Aktual", opini_aktual)
    with m8:
        opini_prediksi = df_filtered["Prediksi Opini BPK"].mode()[0] if not df_filtered.empty else "-"
        st.metric("Prediksi Opini BPK", opini_prediksi)

    st.markdown("---")

    # SECTION SCATTER PLOT KORELASI BELANJA
    st.subheader("🔍 Analisis Scatterplot Korelasi Belanja vs Indikator Keuangan")

    if not df_filtered.empty:
        col_sc1, col_sc2 = st.columns(2)

        # Opsi pilihan indikator keuangan
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
                hover_data=["Pemda", "Tahun", "Opini Y-1"],
                title=f"Scatterplot: Belanja Pegawai (%) vs {var_peg.capitalize()}",
                labels={"b_peg": "Belanja Pegawai (%)", var_peg: var_peg.capitalize()},
                trendline="ols"  # Menampilkan garis tren korelasi
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
                hover_data=["Pemda", "Tahun", "Opini Y-1"],
                title=f"Scatterplot: Belanja Modal (%) vs {var_mdl.capitalize()}",
                labels={"b_mdl": "Belanja Modal (%)", var_mdl: var_mdl.capitalize()},
                trendline="ols"  # Menampilkan garis tren korelasi
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
                color="Kluster" if "Kluster" in df_filtered.columns else "Opini Y-1",
                symbol="Opini Y-1",
                hover_data=["Pemda", "Tahun", "Opini Y-1"],
                title=f"Pemetaan Kluster Pemda ({x_axis} vs {y_axis})",
                size="DCC" if "DCC" in df_filtered.columns else None
            )
            fig_cluster.update_traces(marker=dict(size=12, opacity=0.8))
            st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("Data tidak cukup untuk menampilkan pemetaan kluster.")
