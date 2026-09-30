import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analisis Opini Pemda",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Analisis Kinerja & Kepatuhan Pemda Berdasarkan Opini")
st.markdown("Visualisasi komparatif indikator keuangan, kapasitas fiskal, dan tindak lanjut rekomendasi audit antar kelompok **Opini Y-1**.")

# ---------------------------------------------------------
# Sidebar & Data Loading
# ---------------------------------------------------------
st.sidebar.header("📁 Filter Data")

uploaded_file = st.sidebar.file_uploader("Unggah Dataset (CSV / Excel)", type=["csv", "xlsx"])

@st.cache_data
def generate_dummy_data():
    np.random.seed(42)
    n = 300
    opini_list = ["WTP", "WDP", "TW", "TMP"]
    years = [2022, 2023, 2024]
    pemda_list = [f"Pemda Kab/Kota {i+1}" for i in range(10)]
    
    data = {
        "Tahun": np.random.choice(years, size=n),
        "Pemda": np.random.choice(pemda_list, size=n),
        "Opini Y-1": np.random.choice(opini_list, size=n, p=[0.65, 0.20, 0.10, 0.05]),
        "IKF": np.round(np.random.uniform(0.2, 1.8, size=n), 3),
        "DCC": np.round(np.random.uniform(5, 45, size=n), 2),
        "solvabilitas": np.round(np.random.uniform(0.8, 3.5, size=n), 2),
        "likuiditas": np.round(np.random.uniform(1.0, 5.0, size=n), 2),
        "b_peg": np.round(np.random.uniform(25, 55, size=n), 2),
        "b_mdl": np.round(np.random.uniform(10, 35, size=n), 2),
        "TLRHP Y-1_Sesuai": np.round(np.random.uniform(50, 98, size=n), 2),
        "%lunas_rugi(t-1)": np.round(np.random.uniform(20, 95, size=n), 2)
    }
    return pd.DataFrame(data)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("File berhasil diunggah!")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file: {e}")
        df = generate_dummy_data()
else:
    st.sidebar.info("Menggunakan **Dummy Data** sampel.")
    df = generate_dummy_data()

# ---------------------------------------------------------
# Sidebar Radio Button Filters (Hanya Tahun dan Pemda)
# ---------------------------------------------------------
df_filtered = df.copy()

# 1. Radio Button Tahun
if "Tahun" in df.columns:
    available_years = ["Semua Tahun"] + sorted([str(y) for y in df["Tahun"].dropna().unique()])
    selected_year = st.sidebar.radio("🗓️ Pilih Tahun:", options=available_years, index=0)
    
    if selected_year != "Semua Tahun":
        # Konversi ke tipe data numerik sesuai data asli jika memungkinkan
        try:
            df_filtered = df_filtered[df_filtered["Tahun"] == int(selected_year)]
        except ValueError:
            df_filtered = df_filtered[df_filtered["Tahun"] == selected_year]

# 2. Radio Button Nama Pemda
if "Pemda" in df.columns:
    available_pemda = ["Semua Pemda"] + sorted(list(df["Pemda"].dropna().unique()))
    selected_pemda = st.sidebar.radio("🏛️ Pilih Pemda:", options=available_pemda, index=0)
    
    if selected_pemda != "Semua Pemda":
        df_filtered = df_filtered[df_filtered["Pemda"] == selected_pemda]

# Map Warna Opini BPK
color_map = {
    "WTP": "#2ecc71",  # Hijau
    "WDP": "#f39c12",  # Oranye
    "TW": "#e74c3c",   # Merah
    "TMP": "#95a5a6"   # Abu-abu
}

# ---------------------------------------------------------
# Summary KPI Cards
# ---------------------------------------------------------
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_lik = df_filtered["likuiditas"].mean()
    st.metric("Rata-rata Likuiditas", f"{avg_lik:.2f}" if pd.notnull(avg_lik) else "-")

with col2:
    avg_solv = df_filtered["solvabilitas"].mean()
    st.metric("Rata-rata Solvabilitas", f"{avg_solv:.2f}" if pd.notnull(avg_solv) else "-")

with col3:
    avg_tlrhp = df_filtered["TLRHP Y-1_Sesuai"].mean()
    st.metric("Rata-rata TLRHP Sesuai (%)", f"{avg_tlrhp:.2f}%" if pd.notnull(avg_tlrhp) else "-")

with col4:
    avg_rugi = df_filtered["%lunas_rugi(t-1)"].mean()
    st.metric("Rata-rata Lunas Rugi (%)", f"{avg_rugi:.2f}%" if pd.notnull(avg_rugi) else "-")

st.markdown("---")

# ---------------------------------------------------------
# Tabs Section
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📈 Komparasi Rasio & Kepatuhan", 
    "⚖️ Profil Fiskal & Belanja", 
    "📑 Data Explorer & Korelasi"
])

# ---------------------------------------------------------
# TAB 1: Komparasi Rasio & Kepatuhan
# ---------------------------------------------------------
with tab1:
    st.subheader("Distribusi Indikator Utama Berdasarkan Opini")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        metric_choice = st.selectbox(
            "Pilih Variabel Distribusi (Boxplot):",
            ["likuiditas", "solvabilitas", "IKF", "DCC"]
        )
        fig_box = px.box(
            df_filtered, 
            x="Opini Y-1", 
            y=metric_choice, 
            color="Opini Y-1",
            color_discrete_map=color_map,
            points="all",
            title=f"Distribusi {metric_choice.upper()} per Kategori Opini"
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_right:
        st.write("**Rata-Rata Tingkat Kepatuhan (TLRHP & Pelunasan Kerugian)**")
        df_kepatuhan = df_filtered.groupby("Opini Y-1")[["TLRHP Y-1_Sesuai", "%lunas_rugi(t-1)"]].mean().reset_index()
        
        fig_kepatuhan = go.Figure()
        fig_kepatuhan.add_trace(go.Bar(
            x=df_kepatuhan["Opini Y-1"], 
            y=df_kepatuhan["TLRHP Y-1_Sesuai"], 
            name="TLRHP Y-1 Sesuai (%)",
            marker_color="#3498db"
        ))
        fig_kepatuhan.add_trace(go.Bar(
            x=df_kepatuhan["Opini Y-1"], 
            y=df_kepatuhan["%lunas_rugi(t-1)"], 
            name="% Lunas Rugi (t-1)",
            marker_color="#9b59b6"
        ))
        fig_kepatuhan.update_layout(
            barmode="group",
            title="Perbandingan Kepatuhan Audit per Opini",
            yaxis_title="Persentase (%)"
        )
        st.plotly_chart(fig_kepatuhan, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Profil Fiskal & Belanja
# ---------------------------------------------------------
with tab2:
    st.subheader("Hubungan Antar Variabel Keuangan dan Alokasi Belanja")
    
    c1, c2 = st.columns(2)
    
    with c1:
        fig_scatter = px.scatter(
            df_filtered,
            x="IKF",
            y="DCC",
            color="Opini Y-1",
            size="likuiditas",
            hover_data=df_filtered.columns,
            color_discrete_map=color_map,
            title="Pemetaan IKF vs DCC (Ukuran Bubble: Likuiditas)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with c2:
        df_belanja = df_filtered.groupby("Opini Y-1")[["b_peg", "b_mdl"]].mean().reset_index()
        fig_belanja = go.Figure()
        fig_belanja.add_trace(go.Bar(
            x=df_belanja["Opini Y-1"], 
            y=df_belanja["b_peg"], 
            name="Rata-rata Belanja Pegawai",
            marker_color="#e67e22"
        ))
        fig_belanja.add_trace(go.Bar(
            x=df_belanja["Opini Y-1"], 
            y=df_belanja["b_mdl"], 
            name="Rata-rata Belanja Modal",
            marker_color="#1abc9c"
        ))
        fig_belanja.update_layout(
            barmode="group",
            title="Perbandingan Alokasi Belanja Pegawai vs Belanja Modal",
            yaxis_title="Nilai / Proporsi Belanja"
        )
        st.plotly_chart(fig_belanja, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Data Explorer & Correlation
# ---------------------------------------------------------
with tab3:
    st.subheader("Matriks Korelasi Antar Indikator Numerik")
    
    numeric_cols = ["IKF", "DCC", "solvabilitas", "likuiditas", "b_peg", "b_mdl", "TLRHP Y-1_Sesuai", "%lunas_rugi(t-1)"]
    valid_cols = [c for c in numeric_cols if c in df_filtered.columns]
    
    if len(valid_cols) > 1:
        corr = df_filtered[valid_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Matriks Korelasi (Pearson)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    st.subheader("Tabel Data Detail Pemda")
    st.dataframe(df_filtered, use_container_width=True)
