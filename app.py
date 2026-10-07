import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# Configuration & Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Advisory Keuangan Pemda",
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

st.title("📊 Dashboard Advisory Keuangan Pemda")

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
            
            ikf = np.round(np.random.uniform(0.2, 1.8), 3)
            ikf_y1 = np.round(ikf * np.random.uniform(0.9, 1.1), 3)
            grw_ikf = np.round(((ikf - ikf_y1) / ikf_y1) * 100, 2)

            dcc = np.round(np.random.uniform(30, 180), 2)
            dcc_y1 = np.round(dcc * np.random.uniform(0.85, 1.15), 2)
            grw_dcc = np.round(((dcc - dcc_y1) / dcc_y1) * 100, 2)

            solv = np.round(np.random.uniform(0.8, 3.5), 2)
            solv_y1 = np.round(solv * np.random.uniform(0.9, 1.1), 2)
            grw_solv = np.round(((solv - solv_y1) / solv_y1) * 100, 2)

            lik = np.round(np.random.uniform(1.0, 5.0), 2)
            lik_y1 = np.round(lik * np.random.uniform(0.9, 1.1), 2)
            grw_lik = np.round(((lik - lik_y1) / lik_y1) * 100, 2)

            b_peg = np.round(np.random.uniform(25, 45), 2)
            b_peg_y1 = np.round(b_peg * np.random.uniform(0.95, 1.05), 2)
            grw_bpeg = np.round(((b_peg - b_peg_y1) / b_peg_y1) * 100, 2)

            b_barjas = np.round(np.random.uniform(20, 40), 2)
            b_barjas_y1 = np.round(b_barjas * np.random.uniform(0.95, 1.05), 2)
            grw_bbarjas = np.round(((b_barjas - b_barjas_y1) / b_barjas_y1) * 100, 2)

            b_mdl = np.round(np.random.uniform(10, 30), 2)
            b_mdl_y1 = np.round(b_mdl * np.random.uniform(0.9, 1.1), 2)
            grw_bmdl = np.round(((b_mdl - b_mdl_y1) / b_mdl_y1) * 100, 2)

            records.append({
                "Tahun": thn,
                "Pemda": pmd,
                "b_barjas": b_barjas,
                "b_barjas Y-1": b_barjas_y1,
                "%Grwthb_barjas": grw_bbarjas,
                "IKF": ikf,
                "IKF Y-1": ikf_y1,
                "%GrwthIKF": grw_ikf,
                "DCC": dcc,
                "DCC Y-1": dcc_y1,
                "%GrwthDCC": grw_dcc,
                "solvabilitas": solv,
                "Solv Y-1": solv_y1,
                "%GrwthSolv": grw_solv,
                "likuiditas": lik,
                "Likuiditas Y-1": lik_y1,
                "%GrwtLikuid": grw_lik,
                "b_peg": b_peg,
                "b_peg Y-1": b_peg_y1,
                "%Grwthb_peg": grw_bpeg,
                "b_mdl": b_mdl,
                "b_modal Y-1": b_mdl_y1,
                "%GrwthB_modal": grw_bmdl,
                "TLRHP Y-1_Sesuai": np.round(np.random.uniform(50, 98), 2),
                "%lunas_rugi(t-1)": np.round(np.random.uniform(20, 95), 2),
                "Opini Y-1": OPINI_MAP[np.random.choice([1, 2, 3])],
                "Opini (Y)": OPINI_MAP[opini_code]
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

# Clean Nama Kolom
df.columns = df.columns.str.strip()

# Standardisasi Penamaan Kolom Opini (Y)
if "Opini (Y)" not in df.columns:
    if "Opini Y" in df.columns:
        df["Opini (Y)"] = df["Opini Y"]
    elif "Opini Y-1" in df.columns:
        df["Opini (Y)"] = df["Opini Y-1"]

if "Opini Y_Kode" not in df.columns:
    if set(df["Opini (Y)"].dropna().unique()).issubset({1, 2, 3}):
        df["Opini Y_Kode"] = df["Opini (Y)"].astype(int)
        df["Opini (Y)"] = df["Opini Y_Kode"].map(OPINI_MAP)
    else:
        reverse_map = {"WDP": 1, "WTP PSH": 2, "WTP": 3}
        df["Opini Y_Kode"] = df["Opini (Y)"].map(reverse_map).fillna(3).astype(int)

if "b_barjas" in df.columns and "b_brg" not in df.columns:
    df["b_brg"] = df["b_barjas"]

# ---------------------------------------------------------
# SIDEBAR RADIO BUTTON FILTERS (GLOBAL) - INISIALISASI DINI
# ---------------------------------------------------------
df_filtered = df.copy()

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
# ALGORITMA LOGISTIC REGRESSION + HYPER TUNING (STRATIFIED)
# ---------------------------------------------------------
feature_cols = [
    "IKF", "DCC", "solvabilitas", "likuiditas", 
    "b_peg", "b_mdl", "TLRHP Y-1_Sesuai", "%lunas_rugi(t-1)"
]

available_features = [col for col in feature_cols if col in df.columns]

model_accuracy = None
class_report = None
best_params = None

if len(available_features) == len(feature_cols) and "Opini Y_Kode" in df.columns:
    df_clean = df.dropna(subset=available_features + ["Opini Y_Kode"]).copy()
    
    if len(df_clean) > 5 and df_clean["Opini Y_Kode"].nunique() >= 2:
        try:
            X = df_clean[available_features]
            y = df_clean["Opini Y_Kode"].astype(int)
            
            min_class_samples = y.value_counts().min()
            n_splits = max(2, min(5, min_class_samples))
            
            stratified_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
            base_model = LogisticRegression(max_iter=1000, random_state=42)
            
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=stratified_cv,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            preds_code = best_model.predict(df[available_features].fillna(0))
            df["Prediksi_Kode"] = preds_code
            df["Prediksi Opini BPK (ML)"] = df["Prediksi_Kode"].map(OPINI_MAP)
            
            y_pred = best_model.predict(X)
            model_accuracy = accuracy_score(y, y_pred)
            
            unique_classes = sorted(y.unique())
            target_labels = [f"{OPINI_MAP[c]} ({c})" for c in unique_classes if c in OPINI_MAP]
            
            class_report = classification_report(
                y, y_pred, 
                target_names=target_labels if len(target_labels) == len(unique_classes) else None, 
                output_dict=True
            )
        except Exception as err:
            st.sidebar.warning(f"Model Logistic Regression gagal dilatih: {err}")
            df["Prediksi Opini BPK (ML)"] = df["Opini (Y)"]
    else:
        df["Prediksi Opini BPK (ML)"] = df["Opini (Y)"]
else:
    df["Prediksi Opini BPK (ML)"] = df["Opini (Y)"]

# Update df_filtered setelah kolom prediksi ditambahkan
if selected_year != "Semua Tahun":
    try:
        df_filtered = df[df["Tahun"] == int(selected_year)]
    except ValueError:
        df_filtered = df[df["Tahun"] == selected_year]
else:
    df_filtered = df.copy()

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

    def compute_metric_with_delta(col_curr, col_y1=None, col_grwth=None, is_dcc=False):
        if col_curr not in df.columns or df_filtered.empty:
            return None, None, True
        
        val_curr = df_filtered[col_curr].mean()
        
        if col_grwth in df_filtered.columns and not df_filtered[col_grwth].dropna().empty:
            delta = df_filtered[col_grwth].mean()
        elif col_y1 in df_filtered.columns and not df_filtered[col_y1].dropna().empty:
            val_prev = df_filtered[col_y1].mean()
            delta = val_curr - val_prev
        else:
            delta = None
        
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
        v, d, g = compute_metric_with_delta("solvabilitas", "Solv Y-1", "%GrwthSolv")
        v_str = f"{v:.2f}" if v is not None else "-"
        render_custom_card("Solvabilitas (Aktual)", v_str, d, g)
        
    with b1_c2:
        v, d, g = compute_metric_with_delta("likuiditas", "Likuiditas Y-1", "%GrwtLikuid")
        v_str = f"{v:.2f}" if v is not None else "-"
        render_custom_card("Likuiditas (Aktual)", v_str, d, g)
        
    with b1_c3:
        v, d, g = compute_metric_with_delta("DCC", "DCC Y-1", "%GrwthDCC", is_dcc=True)
        v_str = f"{v:.1f} Hari" if v is not None else "-"
        render_custom_card("Days Cash Coverage (DCC)", v_str, d, g, unit=" Hari")

    st.markdown("<br>", unsafe_allow_html=True)

    # BARIS 2: Indeks Kemampuan Fiskal, TLRHP, Penyelesaian Ganti Rugi
    b2_c1, b2_c2, b2_c3 = st.columns(3)
    with b2_c1:
        v, d, g = compute_metric_with_delta("IKF", "IKF Y-1", "%GrwthIKF")
        v_str = f"{v:.3f}" if v is not None else "-"
        render_custom_card("Indeks Kemampuan Fiskal (IKF)", v_str, d, g)
        
    with b2_c2:
        v, d, g = compute_metric_with_delta("TLRHP Y-1_Sesuai")
        v_str = f"{v:.2f}%" if v is not None else "-"
        render_custom_card("Penyelesaian TLRHP (%)", v_str, d, g, unit="%")
        
    with b2_c3:
        v, d, g = compute_metric_with_delta("%lunas_rugi(t-1)")
        v_str = f"{v:.2f}%" if v is not None else "-"
        render_custom_card("Penyelesaian Ganti Rugi (%)", v_str, d, g, unit="%")

    st.markdown("<br>", unsafe_allow_html=True)

    # BARIS 3: Opini BPK Aktual & Prediksi ML
    b3_c1, b3_c2 = st.columns(2)
    with b3_c1:
        if not df_filtered.empty and "Opini (Y)" in df_filtered.columns:
            opini_aktual = df_filtered["Opini (Y)"].iloc[0] if len(df_filtered) == 1 else df_filtered["Opini (Y)"].mode()[0]
        else:
            opini_aktual = "-"
        render_opini_card("Opini BPK Aktual (Opini Y)", opini_aktual)
        
    with b3_c2:
        if not df_filtered.empty and "Prediksi Opini BPK (ML)" in df_filtered.columns:
            opini_pred = df_filtered["Prediksi Opini BPK (ML)"].iloc[0] if len(df_filtered) == 1 else df_filtered["Prediksi Opini BPK (ML)"].mode()[0]
        else:
            opini_pred = "-"
        render_opini_card("Prediksi Opini BPK (ML)", opini_pred)

    if model_accuracy is not None:
        with st.expander("ℹ️ Detail Model Logistic Regression (Stratified Tuning) & Perbandingan Prediksi"):
            st.write(f"**Akurasi Model Terbaik:** {model_accuracy * 100:.2f}%")
            if best_params:
                st.info(f"⚙️ **Hyperparameter Terbaik (GridSearch + StratifiedKFold):** {best_params}")
            
            col_exp1, col_exp2 = st.columns([1, 2])
            with col_exp1:
                st.markdown("**Laporan Klasifikasi:**")
                if class_report:
                    st.dataframe(pd.DataFrame(class_report).transpose().style.format("{:.2f}"))
            
            with col_exp2:
                st.markdown("**Tabel Hasil Prediksi vs Opini Aktual:**")
                show_cols = ["Pemda", "Tahun", "Opini (Y)", "Prediksi Opini BPK (ML)"]
                existing_show_cols = [c for c in show_cols if c in df_filtered.columns]
                
                df_compare = df_filtered[existing_show_cols].copy()
                df_compare["Status Evaluasi"] = np.where(
                    df_compare["Opini (Y)"] == df_compare["Prediksi Opini BPK (ML)"], 
                    "✅ Sesuai", 
                    "❌ Beda"
                )
                st.dataframe(df_compare, use_container_width=True)

    st.markdown("---")

    # SECTION SCATTERPLOT KORELASI BELANJA
    st.subheader("🔍 Analisis Scatterplot Korelasi Belanja vs Indikator Keuangan")

    if not df_filtered.empty:
        col_c_sel, col_dummy = st.columns([2, 2])
        with col_c_sel:
            selected_korelasi_var = st.selectbox(
                "📌 Pilih Variabel Korelasi (Indikator Keuangan):",
                options=["IKF", "DCC", "likuiditas", "solvabilitas"],
                index=0,
                key="combo_korelasi_var"
            )

        col_sc1, col_sc2, col_sc3 = st.columns(3)

        with col_sc1:
            fig_sc_peg = px.scatter(
                df_filtered,
                x="Tahun",
                y="b_peg" if "b_peg" in df_filtered.columns else "IKF",
                color=selected_korelasi_var,
                size=selected_korelasi_var,
                hover_data=["Pemda", "Opini (Y)", "Prediksi Opini BPK (ML)"],
                title=f"Tahun vs Belanja Pegawai (%) (Korelasi: {selected_korelasi_var.upper()})",
                labels={"Tahun": "Tahun", "b_peg": "Proporsi Belanja Pegawai (%)", selected_korelasi_var: selected_korelasi_var.upper()},
                color_continuous_scale="Viridis"
            )
            fig_sc_peg.update_traces(marker=dict(opacity=0.85))
            st.plotly_chart(fig_sc_peg, use_container_width=True)

        with col_sc2:
            col_barjas = "b_barjas" if "b_barjas" in df_filtered.columns else "b_brg"
            fig_sc_brg = px.scatter(
                df_filtered,
                x="Tahun",
                y=col_barjas,
                color=selected_korelasi_var,
                size=selected_korelasi_var,
                hover_data=["Pemda", "Opini (Y)", "Prediksi Opini BPK (ML)"],
                title=f"Tahun vs Belanja Barang/Jasa (%) (Korelasi: {selected_korelasi_var.upper()})",
                labels={"Tahun": "Tahun", col_barjas: "Proporsi Belanja Barjas (%)", selected_korelasi_var: selected_korelasi_var.upper()},
                color_continuous_scale="Plasma"
            )
            fig_sc_brg.update_traces(marker=dict(opacity=0.85))
            st.plotly_chart(fig_sc_brg, use_container_width=True)

        with col_sc3:
            fig_sc_mdl = px.scatter(
                df_filtered,
                x="Tahun",
                y="b_mdl" if "b_mdl" in df_filtered.columns else "IKF",
                color=selected_korelasi_var,
                size=selected_korelasi_var,
                hover_data=["Pemda", "Opini (Y)", "Prediksi Opini BPK (ML)"],
                title=f"Tahun vs Belanja Modal (%) (Korelasi: {selected_korelasi_var.upper()})",
                labels={"Tahun": "Tahun", "b_mdl": "Proporsi Belanja Modal (%)", selected_korelasi_var: selected_korelasi_var.upper()},
                color_continuous_scale="Cividis"
            )
            fig_sc_mdl.update_traces(marker=dict(opacity=0.85))
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

    all_individual_vars = [
        "IKF", "DCC", "solvabilitas", "likuiditas", 
        "b_peg", "b_barjas", "b_mdl", "TLRHP Y-1_Sesuai", "%lunas_rugi(t-1)", "Opini (Y)"
    ]
    available_ind_vars = [v for v in all_individual_vars if v in df_tab3.columns]

    col_opt1, col_opt2 = st.columns([2, 1])
    
    with col_opt1:
        selected_kmeans_vars = st.multiselect(
            "📌 Pilih variabel individual untuk Algoritma K-Means (Pilih minimal 2):",
            options=available_ind_vars,
            default=["IKF", "DCC", "solvabilitas", "likuiditas", "Opini (Y)"] if "Opini (Y)" in available_ind_vars else available_ind_vars[:4],
            key="multiselect_kmeans_ind"
        )

    with col_opt2:
        n_clusters = st.slider("Jumlah Kluster (k):", min_value=2, max_value=5, value=3, key="slider_k_means")

    if len(selected_kmeans_vars) < 2:
        st.warning("⚠️ Harap pilih **minimal 2 variabel** untuk mengeksekusi model K-Means.")
    else:
        kmeans_calc_cols = []
        for v in selected_kmeans_vars:
            if v == "Opini (Y)":
                kmeans_calc_cols.append("Opini Y_Kode")
            else:
                kmeans_calc_cols.append(v)

        df_km = df_tab3.dropna(subset=kmeans_calc_cols).copy()
        
        if len(df_km) >= n_clusters:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_km[kmeans_calc_cols])
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_km["Cluster_KMeans"] = kmeans.fit_predict(scaled_data)
            df_km["Cluster_Label"] = df_km["Cluster_KMeans"].apply(lambda x: f"Kluster {x+1}")

            st.markdown("---")

            x_km = selected_kmeans_vars[0]
            y_km = selected_kmeans_vars[1]

            fig_km = px.scatter(
                df_km,
                x=x_km,
                y=y_km,
                color="Cluster_Label",
                hover_data=["Pemda", "Tahun", "Opini (Y)", "Prediksi Opini BPK (ML)"],
                title=f"Scatterplot Hasil Klasterisasi K-Means ({x_km} vs {y_km})",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_km.update_traces(marker=dict(size=12, opacity=0.85))
            st.plotly_chart(fig_km, use_container_width=True)

            with st.expander("📋 Tabel Hasil Klasterisasi K-Means Pemda"):
                cols_display = ["Pemda", "Tahun", "Cluster_Label", "Opini (Y)", "Prediksi Opini BPK (ML)"] + [v for v in selected_kmeans_vars if v != "Opini (Y)"]
                existing_cols_display = [c for c in cols_display if c in df_km.columns]
                st.dataframe(df_km[existing_cols_display], use_container_width=True)
        else:
            st.info("Jumlah data tidak cukup untuk menjalankan algoritma K-Means dengan k=" + str(n_clusters))
