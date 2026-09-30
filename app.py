import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------------
# Configuration & Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analisis Opini Pemda",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Ringkasan Kinerja & Indikator Utama Pemda")
st.markdown("Visualisasi indikator keuangan, kepatuhan audit, status opini BPK, serta tren proporsi alokasi belanja.")

# ---------------------------------------------------------
# Sidebar: Upload File & 2 Radio Buttons Filter
# ---------------------------------------------------------
st.sidebar.header("📁 Filter Data")

uploaded_file = st.sidebar.file_uploader("Unggah File Excel/CSV", type=["xlsx", "xls", "csv"])

@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n = 150
    years = [2021, 2022, 2023, 2024]
    pemda_list = [f"Pemda Kab/Kota {chr(65+i)}" for i in range(10)]
    opini_list = ["WTP", "WDP", "TW", "TMP"]
    
    data = {
        "Tahun": np.random.choice(years, size=n),
        "Pemda": np.random.choice(pemda_list, size=n),
        "IKF": np.round(np.random.uniform(0.2, 1.8, size=n), 3),
        "DCC": np.round(np.random.uniform(30, 180, size=n), 2),  # Days Cash Coverage
        "solvabilitas": np.round(np.random.uniform(0.8, 3.5, size=n), 2),
        "likuiditas": np.round(np.random.uniform(1.0, 5.0, size=n), 2),
        "b_peg": np.round(np.random.uniform(25, 45, size=n), 2),  # Belanja Pegawai (%)
        "b_brg": np.round(np.random.uniform(20, 40, size=n), 2),  # Belanja Barang (%)
        "b_mdl": np.round(np.random.uniform(10, 30, size=n), 2),  # Belanja Modal (%)
        "TLRHP Y-1_Sesuai": np.round(np.random.uniform(50, 98, size=n), 2),
        "%lunas_rugi(t-1)": np.round(np.random.uniform(20, 95, size=n), 2),
        "Opini Y-1": np.random.choice(opini_list, size=n, p=[0.70, 0.18, 0.08, 0.04]),
        "Prediksi Opini BPK": np.random.choice(opini_list, size=n, p=[0.75, 0.15, 0.06, 0.04])
    }
    return pd.DataFrame(data)

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
    st.sidebar.info("Menggunakan sampel data bawaan.")
    df = generate_sample_data()

# Penyesuaian otomatis jika beberapa kolom kustom belum ada di file Excel
if "Opini Y-1" not in df.columns:
    df["Opini Y-1"] = np.random.choice(["WTP", "WDP", "TW", "TMP"], size=len(df), p=[0.7, 0.2, 0.07, 0.03])
if "Prediksi Opini BPK" not in df.columns:
    df["Prediksi Opini BPK"] = df["Opini Y-1"]  # default samakan jika tidak ada kolom prediksi
if "b_brg" not in df.columns:
    df["b_brg"] = np.round(np.random.uniform(20, 40, size=len(df)), 2)

df_filtered = df.copy()

# ---------------------------------------------------------
# SIDEBAR RADIO BUTTON FILTERS (Hanya 2 Radio Button)
# ---------------------------------------------------------

# 1. Radio Button Filter Tahun
if "Tahun" in df.columns:
    available_years = ["Semua Tahun"] + sorted([str(y) for y in df["Tahun"].dropna().unique()])
    selected_year = st.sidebar.radio("🗓️ Pilih Tahun:", options=available_years, index=0)
    
    if selected_year != "Semua Tahun":
        try:
            df_filtered = df_filtered[df_filtered["Tahun"] == int(selected_year)]
        except ValueError:
            df_filtered = df_filtered[df_filtered["Tahun"] == selected_year]

# 2. Radio Button Filter Pemda
if "Pemda" in df.columns:
    available_pemda = ["Semua Pemda"] + sorted(list(df["Pemda"].dropna().unique()))
    selected_pemda = st.sidebar.radio("🏛️ Pilih Pemda:", options=available_pemda, index=0)
    
    if selected_pemda != "Semua Pemda":
        df_filtered = df_filtered[df_filtered["Pemda"] == selected_pemda]

# ---------------------------------------------------------
# HALAMAN UTAMA: METRIK & RINGKASAN INDIKATOR
# ---------------------------------------------------------
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
    # Mengambil mode (opini paling dominan) jika filter semua pemda, atau nilai persis jika 1 pemda
    opini_aktual = df_filtered["Opini Y-1"].mode()[0] if not df_filtered.empty else "-"
    st.metric("Opini BPK Aktual", opini_aktual)

with m8:
    opini_prediksi = df_filtered["Prediksi Opini BPK"].mode()[0] if not df_filtered.empty else "-"
    st.metric("Prediksi Opini BPK", opini_prediksi)

st.markdown("---")

# ---------------------------------------------------------
# HALAMAN UTAMA: GRAFIK TREN PROPORSI BELANJA (DARI TAHUN KE TAHUN)
# ---------------------------------------------------------
st.subheader("📈 Tren Korelasi Proporsi Belanja (Dari Tahun ke Tahun)")

# Agregasi data per tahun
if "Tahun" in df_filtered.columns and not df_filtered.empty:
    df_trend = df_filtered.groupby("Tahun")[["b_mdl", "b_brg", "b_peg"]].mean().reset_index()
    
    col_g1, col_g2, col_g3 = st.columns(3)
    
    # 1. Grafik Belanja Modal
    with col_g1:
        fig_mdl = px.line(
            df_trend,
            x="Tahun",
            y="b_mdl",
            markers=True,
            title="Belanja Modal (b_mdl)",
            labels={"b_mdl": "Proporsi (%)", "Tahun": "Tahun"}
        )
        fig_mdl.update_traces(line_color="#1abc9c", line_width=3)
        st.plotly_chart(fig_mdl, use_container_width=True)
        
    # 2. Grafik Belanja Barang
    with col_g2:
        fig_brg = px.line(
            df_trend,
            x="Tahun",
            y="b_brg",
            markers=True,
            title="Belanja Barang (b_brg)",
            labels={"b_brg": "Proporsi (%)", "Tahun": "Tahun"}
        )
        fig_brg.update_traces(line_color="#3498db", line_width=3)
        st.plotly_chart(fig_brg, use_container_width=True)
        
    # 3. Grafik Belanja Pegawai
    with col_g3:
        fig_peg = px.line(
            df_trend,
            x="Tahun",
            y="b_peg",
            markers=True,
            title="Belanja Pegawai (b_peg)",
            labels={"b_peg": "Proporsi (%)", "Tahun": "Tahun"}
        )
        fig_peg.update_traces(line_color="#e67e22", line_width=3)
        st.plotly_chart(fig_peg, use_container_width=True)
else:
    st.info("Data tidak cukup untuk menampilkan tren tahunan.")

st.markdown("---")

# Tabel Detail Ringkas
st.subheader("📋 Detail Data Terfilter")
st.dataframe(df_filtered, use_container_width=True)
