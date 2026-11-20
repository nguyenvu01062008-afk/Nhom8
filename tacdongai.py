"""
AI Academic Impact Predictor — Nhóm 8
Phân tích & dự đoán tác động của GenAI đến GPA sinh viên bằng Decision Tree Regressor.

Chạy:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# ======================================================================
# 0. CẤU HÌNH CHUNG
# ======================================================================
BASE_DIR = Path(__file__).parent
DATA_DIRS = [BASE_DIR / "data", BASE_DIR]
# Thứ tự ưu tiên tên file (cặp train, test). Hỗ trợ .csv và .xlsx.
FILE_CANDIDATES = [
    ("train.csv", "test.csv"),
    ("train.xlsx", "test.xlsx"),
    ("Nhóm8_TacDongAI_train.xlsx", "Nhóm8_TacDongAI_test.xlsx"),
]

# >>> SỬA CHO ĐÚNG VỚI TRANG KAGGLE CỦA BẠN <<<
DATASET_NAME = "AI Impact on Students (Kaggle)"
DATASET_URL = ""  # dán link trang Kaggle vào đây
DATASET_ORIGIN_NOTE = (
    "Nhóm ghi nhận dữ liệu liên quan đến sinh viên tại Melbourne (Úc). "
    "Thông tin về phương pháp thu thập (khảo sát thật hay dữ liệu tổng hợp) "
    "cần được đối chiếu với phần mô tả của tác giả trên Kaggle."
)

NUMERIC_FEATURES = [
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Tool_Diversity",
    "Traditional_Study_Hours",
    "Year_of_Study_Encoded",
    "Prompt_Engineering_Skill_Encoded",
    "Paid_Subscription_Encoded",
]
MAJOR_COLS = ["Major_Arts", "Major_Business", "Major_Humanities", "Major_Medical", "Major_STEM"]
USECASE_COLS = [
    "UseCase_Copywriting/Drafting",
    "UseCase_Debugging/Troubleshooting",
    "UseCase_Direct_Answer_Generation",
    "UseCase_Ideation",
    "UseCase_Summarizing_Reading",
]
FEATURE_COLS = NUMERIC_FEATURES + MAJOR_COLS + USECASE_COLS
TARGET_COL = "Post_Semester_GPA"

# Bảng màu giao diện sáng
C_TEXT = "#0f172a"
C_MUTED = "#475569"
C_PRIMARY = "#0369a1"
C_ACCENT = "#4f46e5"
C_GOOD = "#15803d"
C_BAD = "#b91c1c"
C_PURPLE = "#7e22ce"
C_WARN = "#d97706"

st.set_page_config(
    page_title="AI Impact Analytics | Nhóm 8",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
    .stApp {{ background: #f8fafc; color: {C_TEXT}; }}
    [data-testid="stSidebar"] {{ background: #eef2f7; }}
    .kpi-card {{
        background: #ffffff;
        border: 1px solid #cbd5e1; border-radius: 16px;
        padding: 18px; text-align: center; box-shadow: 0 4px 16px rgba(15,23,42,0.08);
        transition: transform .3s ease; height: 100%;
    }}
    .kpi-card:hover {{ transform: translateY(-4px); border-color: {C_PRIMARY}; }}
    .kpi-title {{ font-size: .8rem; color: {C_MUTED}; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
    .kpi-value {{
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, {C_PRIMARY}, {C_ACCENT});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .kpi-note {{ font-size: .8rem; margin-top: 5px; }}
    .hero-banner {{
        background: linear-gradient(90deg, #4338ca 0%, #4f46e5 50%, #0369a1 100%);
        padding: 1.6rem 2rem; border-radius: 20px;
        margin-bottom: 1.4rem; box-shadow: 0 6px 20px rgba(67,56,202,.25);
    }}
    .hero-title {{ font-size: 2rem; font-weight: 800; color: #ffffff; margin-bottom: .3rem; }}
    .hero-subtitle {{ font-size: 1rem; color: #e0e7ff; }}
    .paper-card {{
        background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid {C_PRIMARY};
        padding: 16px 20px; border-radius: 10px; margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(15,23,42,.05);
    }}
</style>
""",
    unsafe_allow_html=True,
)


def kpi(title: str, value: str, note: str = "", color: str = C_PRIMARY) -> str:
    return (
        f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note" style="color:{color};">{note}</div></div>'
    )


def style_fig(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_TEXT),
        height=height,
    )
    return fig


# ======================================================================
# 1. TẢI DỮ LIỆU
# ======================================================================
def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)


def _make_demo_data(n: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dữ liệu GIẢ LẬP chỉ để chạy thử giao diện khi chưa có file thật."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "Pre_Semester_GPA": np.round(rng.uniform(1.8, 4.0, n), 3),
            "Weekly_GenAI_Hours": np.round(rng.uniform(0, 35, n), 2),
            "Tool_Diversity": rng.integers(1, 6, n),
            "Traditional_Study_Hours": np.round(rng.uniform(2, 25, n), 2),
            "Year_of_Study_Encoded": rng.integers(1, 6, n),
            "Prompt_Engineering_Skill_Encoded": rng.integers(1, 4, n),
            "Paid_Subscription_Encoded": rng.integers(0, 2, n),
        }
    )
    majors = rng.choice(MAJOR_COLS, n)
    for m in MAJOR_COLS:
        df[m] = (majors == m).astype(int)
    usecases = rng.choice(USECASE_COLS, n)
    for u in USECASE_COLS:
        df[u] = (usecases == u).astype(int)
    delta = (
        0.015 * df["Traditional_Study_Hours"]
        + 0.08 * df["Prompt_Engineering_Skill_Encoded"]
        - 0.012 * np.maximum(0, df["Weekly_GenAI_Hours"] - 15)
        - 0.15 * df["UseCase_Direct_Answer_Generation"]
        + 0.06 * df["UseCase_Ideation"]
        - 0.07
        + rng.normal(0, 0.12, n)
    )
    df[TARGET_COL] = np.clip(np.round(df["Pre_Semester_GPA"] + delta, 3), 0.0, 4.0)
    cut = int(n * 0.8)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


@st.cache_data(show_spinner="Đang tải dữ liệu...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Trả về (train, test, loại_dữ_liệu, tên_file). loại: 'real' hoặc 'demo'."""
    for folder in DATA_DIRS:
        for tr_name, te_name in FILE_CANDIDATES:
            tr, te = folder / tr_name, folder / te_name
            if tr.exists() and te.exists():
                return _read_table(tr), _read_table(te), "real", f"{tr_name} / {te_name}"
    train, test = _make_demo_data()
    return train, test, "demo", "dữ liệu giả lập"


# ======================================================================
# 2. HUẤN LUYỆN & ĐÁNH GIÁ
# ======================================================================
def _evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray, pre: pd.Series) -> dict:
    y_pred = np.clip(y_pred, 0.0, 4.0)
    return {
        "Mô hình": name,
        "R² (GPA sau)": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R² (mức thay đổi GPA)": r2_score(y_true - pre, y_pred - pre),
    }


@st.cache_data(show_spinner="Đang huấn luyện mô hình...")
def train_models(train_df: pd.DataFrame, test_df: pd.DataFrame, max_depth: int):
    X_tr, y_tr = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_te, y_te = test_df[FEATURE_COLS], test_df[TARGET_COL]
    pre_te = test_df["Pre_Semester_GPA"]

    dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=10, random_state=42).fit(X_tr, y_tr)
    rf = RandomForestRegressor(
        n_estimators=150, max_depth=max_depth, min_samples_leaf=10, n_jobs=-1, random_state=42
    ).fit(X_tr, y_tr)
    lin = LinearRegression().fit(X_tr, y_tr)

    preds = {
        "Baseline (GPA sau = GPA trước)": pre_te.to_numpy(),
        "Hồi quy tuyến tính": lin.predict(X_te),
        "Decision Tree (mô hình chính)": dt.predict(X_te),
        "Random Forest (tham khảo)": rf.predict(X_te),
    }
    results = pd.DataFrame([_evaluate(n, y_te, p, pre_te) for n, p in preds.items()])

    depth_rows = []
    for d in range(2, 16):
        m = DecisionTreeRegressor(max_depth=d, min_samples_leaf=10, random_state=42).fit(X_tr, y_tr)
        depth_rows.append(
            {
                "max_depth": d,
                "Train RMSE": float(np.sqrt(mean_squared_error(y_tr, m.predict(X_tr)))),
                "Test RMSE": float(np.sqrt(mean_squared_error(y_te, m.predict(X_te)))),
            }
        )

    fi_df = (
        pd.DataFrame({"Đặc trưng": FEATURE_COLS, "Độ quan trọng (%)": dt.feature_importances_ * 100})
        .sort_values("Độ quan trọng (%)", ascending=False)
        .reset_index(drop=True)
    )
    return dt, rf, results, pd.DataFrame(depth_rows), fi_df


# ======================================================================
# 3. NẠP DỮ LIỆU + SIDEBAR
# ======================================================================
train_df, test_df, data_kind, data_file = load_data()

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in train_df.columns or c not in test_df.columns]
if missing:
    st.error(f"File dữ liệu thiếu các cột bắt buộc: {missing}")
    st.stop()

with st.sidebar:
    st.title("🎓 Bảng điều khiển")
    menu = st.radio(
        "Phân hệ:",
        [
            "🔮 Dự đoán GPA cá nhân",
            "📖 Về dữ liệu & cách tính GPA",
            "📊 Đánh giá mô hình",
            "🗂️ Khám phá dữ liệu",
            "📚 Bối cảnh & tài liệu tham khảo",
        ],
    )
    st.markdown("---")
    max_depth = st.slider("Độ sâu tối đa của cây (max_depth)", 3, 15, 10)
    st.caption("Mặc định 10: giá trị cho RMSE thấp nhất khi cross-validation 5-fold trên tập train (không dùng tập test để chọn). Cây sâu hơn dễ overfit — xem biểu đồ ở mục *Đánh giá mô hình*.")

dt_model, rf_model, results_df, depth_df, fi_df = train_models(train_df, test_df, max_depth)

# Banner chỉ hiện chữ "demo" khi thật sự không có file dữ liệu, để tránh hiểu nhầm.
banner_src = (
    f"Dữ liệu: <b>{DATASET_NAME}</b> — {len(train_df):,} mẫu train / {len(test_df):,} mẫu test"
    if data_kind == "real"
    else "⚠️ <b>CHẾ ĐỘ DEMO — dữ liệu giả lập, chưa có file dữ liệu thật trong thư mục data/</b>"
)
st.markdown(
    f"""
<div class="hero-banner">
  <div class="hero-title">🎓 AI ACADEMIC IMPACT PREDICTOR — NHÓM 8</div>
  <div class="hero-subtitle">Phân tích mối liên hệ giữa thói quen dùng GenAI và sự thay đổi GPA của sinh viên bằng <b>Decision Tree Regressor</b></div>
  <div style="margin-top:8px;font-size:.85rem;color:#bfdbfe;">📂 {banner_src}</div>
</div>
""",
    unsafe_allow_html=True,
)

dt_row = results_df[results_df["Mô hình"].str.startswith("Decision Tree")].iloc[0]
base_row = results_df.iloc[0]

# ======================================================================
# TAB 1: DỰ ĐOÁN
# ======================================================================
if menu.startswith("🔮"):
    col_in, col_out = st.columns([1.15, 1], gap="large")

    with col_in:
        st.subheader("📝 Thông tin học tập & thói quen dùng AI")
        c1, c2 = st.columns(2)
        with c1:
            pre_gpa = st.slider("GPA trước học kỳ (thang 4.0)", 0.0, 4.0, 3.20, 0.05)
            ai_hours = st.slider("Giờ dùng GenAI / tuần", 0.0, 40.0, 8.5, 0.5)
            trad_hours = st.slider("Giờ tự học truyền thống / tuần", 0.0, 36.0, 11.0, 0.5)
        with c2:
            tool_div = st.slider("Số loại công cụ AI đang dùng (1–5)", 1, 5, 3)
            year_map = {
                "Năm nhất": 1, "Năm hai": 2, "Năm ba": 3, "Năm cuối": 4, "Sau đại học": 5,
            }
            year_label = st.selectbox("Năm học", list(year_map), index=2)
            skill_map = {"Mới bắt đầu": 1, "Trung cấp": 2, "Nâng cao": 3}
            skill_label = st.selectbox("Kỹ năng viết prompt", list(skill_map), index=1)

        c3, c4 = st.columns(2)
        with c3:
            major_map = {
                "Kỹ thuật & Công nghệ (STEM)": "Major_STEM",
                "Kinh doanh (Business)": "Major_Business",
                "KHXH & Nhân văn (Humanities)": "Major_Humanities",
                "Y khoa (Medical)": "Major_Medical",
                "Nghệ thuật (Arts)": "Major_Arts",
            }
            major_label = st.selectbox("Khối ngành", list(major_map))
        with c4:
            usecase_map = {
                "Lên ý tưởng (Ideation)": "UseCase_Ideation",
                "Lập trình / Gỡ lỗi (Debugging)": "UseCase_Debugging/Troubleshooting",
                "Tóm tắt / Đọc tài liệu (Summarizing)": "UseCase_Summarizing_Reading",
                "Soạn thảo văn bản (Drafting)": "UseCase_Copywriting/Drafting",
                "Lấy đáp án trực tiếp (Direct Answer)": "UseCase_Direct_Answer_Generation",
            }
            usecase_label = st.selectbox("Mục đích dùng AI chính", list(usecase_map))
        paid_sub = st.toggle("Dùng gói AI trả phí", value=False)

    row = {c: 0 for c in FEATURE_COLS}
    row.update(
        Pre_Semester_GPA=pre_gpa,
        Weekly_GenAI_Hours=ai_hours,
        Tool_Diversity=tool_div,
        Traditional_Study_Hours=trad_hours,
        Year_of_Study_Encoded=year_map[year_label],
        Prompt_Engineering_Skill_Encoded=skill_map[skill_label],
        Paid_Subscription_Encoded=int(paid_sub),
    )
    row[major_map[major_label]] = 1
    row[usecase_map[usecase_label]] = 1
    input_df = pd.DataFrame([row])[FEATURE_COLS]

    pred = float(np.clip(dt_model.predict(input_df)[0], 0.0, 4.0))
    pred_rf = float(np.clip(rf_model.predict(input_df)[0], 0.0, 4.0))
    delta = pred - pre_gpa
    rmse = float(dt_row["RMSE"])

    with col_out:
        st.subheader("🎯 Kết quả dự đoán")
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                number={"font": {"size": 50, "color": C_PRIMARY}, "valueformat": ".2f"},
                delta={"reference": pre_gpa, "valueformat": "+.2f",
                       "increasing": {"color": "#16a34a"}, "decreasing": {"color": "#dc2626"}},
                title={"text": "GPA DỰ KIẾN SAU HỌC KỲ", "font": {"size": 16, "color": C_TEXT}},
                gauge={
                    "axis": {"range": [0, 4.0]},
                    "bar": {"color": C_ACCENT},
                    "steps": [
                        {"range": [0, 2.0], "color": "rgba(239,68,68,.25)"},
                        {"range": [2.0, 3.2], "color": "rgba(234,179,8,.30)"},
                        {"range": [3.2, 4.0], "color": "rgba(34,197,94,.28)"},
                    ],
                    "threshold": {"line": {"color": C_WARN, "width": 4}, "thickness": 0.75, "value": pre_gpa},
                },
            )
        )
        style_fig(fig, 300).update_layout(margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Vạch cam = GPA trước học kỳ. Sai số điển hình của mô hình trên tập test: ±{rmse:.2f} GPA (RMSE). "
            f"Random Forest dự đoán {pred_rf:.2f} để đối chiếu."
        )

        if abs(delta) < max(0.05, rmse * 0.25):
            st.info(f"⚖️ Thay đổi dự kiến {delta:+.2f} GPA — nhỏ hơn sai số của mô hình, nên xem như ổn định.")
        elif delta > 0:
            st.success(f"📈 Mô hình dự đoán GPA tăng {delta:+.2f}.")
        else:
            st.warning(f"📉 Mô hình dự đoán GPA giảm {delta:+.2f}.")

        out_of_range = [
            f"{label} ({v:g})"
            for label, col, v in [
                ("giờ dùng AI", "Weekly_GenAI_Hours", ai_hours),
                ("giờ tự học", "Traditional_Study_Hours", trad_hours),
                ("GPA trước", "Pre_Semester_GPA", pre_gpa),
            ]
            if not (train_df[col].min() <= v <= train_df[col].max())
        ]
        if out_of_range:
            st.warning("Giá trị nằm ngoài khoảng dữ liệu huấn luyện, dự đoán không đáng tin: " + ", ".join(out_of_range))

        with st.expander("Gợi ý theo quy tắc (không phải đầu ra của mô hình)"):
            if usecase_map[usecase_label] == "UseCase_Direct_Answer_Generation":
                st.write("- Dùng AI chủ yếu để lấy đáp án có thể làm giảm việc rèn tư duy; nhiều nghiên cứu khuyến nghị dùng AI để giải thích và phản biện.")
            if ai_hours > 20 and trad_hours < 8:
                st.write("- Giờ dùng AI cao trong khi giờ tự học thấp: nên cân đối lại.")
            if not any([usecase_map[usecase_label] == "UseCase_Direct_Answer_Generation", ai_hours > 20 and trad_hours < 8]):
                st.write("- Chưa có cảnh báo theo quy tắc nào.")

    st.markdown("---")
    st.markdown("#### 🔬 Phân tích 'nếu...': GPA dự đoán thay đổi thế nào khi chỉ đổi số giờ dùng AI?")
    grid = np.linspace(0, float(train_df["Weekly_GenAI_Hours"].max()), 60)
    sweep = pd.DataFrame(np.repeat(input_df.to_numpy(dtype=float), len(grid), axis=0), columns=FEATURE_COLS)
    sweep["Weekly_GenAI_Hours"] = grid
    curve = pd.DataFrame(
        {
            "Giờ dùng AI / tuần": grid,
            "Decision Tree": np.clip(dt_model.predict(sweep), 0, 4) - pre_gpa,
            "Random Forest": np.clip(rf_model.predict(sweep), 0, 4) - pre_gpa,
        }
    ).melt("Giờ dùng AI / tuần", var_name="Mô hình", value_name="Thay đổi GPA dự đoán")
    fig_c = px.line(curve, x="Giờ dùng AI / tuần", y="Thay đổi GPA dự đoán", color="Mô hình")
    fig_c.add_vline(x=ai_hours, line_dash="dash", line_color=C_WARN)
    st.plotly_chart(style_fig(fig_c, 330), use_container_width=True)
    st.caption("Cây quyết định cho đường bậc thang; Random Forest mượt hơn. Đây là mối liên hệ mô hình học được, chưa phải bằng chứng nhân quả.")

# ======================================================================
# TAB 2: VỀ DỮ LIỆU & GPA
# ======================================================================
elif menu.startswith("📖"):
    st.subheader("📖 Dữ liệu đến từ đâu và GPA nghĩa là gì?")

    st.markdown("#### 1. Nguồn dữ liệu")
    link = f" ([trang Kaggle]({DATASET_URL}))" if DATASET_URL else ""
    st.write(
        f"Bộ dữ liệu **{DATASET_NAME}**{link}, được nhóm tải về từ Kaggle (50.000 sinh viên), "
        f"tiền xử lý (Ordinal Encoding cho năm học và kỹ năng prompt, One-Hot Encoding cho ngành học và mục đích dùng AI) "
        f"rồi chia 80% train ({len(train_df):,}) / 20% test ({len(test_df):,})."
    )
    st.info(DATASET_ORIGIN_NOTE)
    st.warning(
        "Cần lưu ý: nếu tác giả không công bố phương pháp thu thập, hoặc bộ dữ liệu có dấu hiệu được tổng hợp, "
        "kết luận chỉ nên hiểu là *mô hình học được quy luật trong bộ dữ liệu này*, không phải kết luận về toàn bộ sinh viên."
    )

    st.markdown("#### 2. GPA ở đây là gì?")
    st.write(
        "- **Pre_Semester_GPA**: GPA của sinh viên *trước* học kỳ khảo sát, thang 4.0.\n"
        "- **Post_Semester_GPA**: GPA *sau* học kỳ đó, thang 4.0. Đây là biến mục tiêu mô hình cần dự đoán.\n"
        "- Mỗi sinh viên chỉ có **một giá trị GPA tổng**, không có điểm theo từng môn. "
        "Vì vậy mô hình không cho biết AI ảnh hưởng đến môn nào.\n"
        "- **Thay đổi GPA = Post − Pre** là đại lượng phản ánh tác động trong học kỳ."
    )
    st.caption("Phần mô tả trên suy ra từ tên cột và cách nhóm xử lý; hãy đối chiếu với data card của Kaggle trước khi nộp.")

    st.markdown("#### 3. Dữ liệu đầu vào (17 đặc trưng sau mã hóa)")
    dictionary = pd.DataFrame(
        [
            ("Pre_Semester_GPA", "GPA trước học kỳ", "Số thực, thang 0–4"),
            ("Weekly_GenAI_Hours", "Giờ dùng GenAI mỗi tuần", "Số thực"),
            ("Tool_Diversity", "Số loại công cụ AI sử dụng", "Số nguyên (1–5)"),
            ("Traditional_Study_Hours", "Giờ tự học truyền thống mỗi tuần", "Số thực"),
            ("Year_of_Study_Encoded", "Năm học", "Ordinal: 1 = năm nhất … 5 = sau đại học"),
            ("Prompt_Engineering_Skill_Encoded", "Kỹ năng viết prompt", "Ordinal: 1 = mới, 2 = trung cấp, 3 = nâng cao"),
            ("Paid_Subscription_Encoded", "Có dùng gói AI trả phí", "Nhị phân 0/1"),
            ("Major_Arts / Business / Humanities / Medical / STEM", "Khối ngành", "One-hot (5 cột 0/1)"),
            ("UseCase_Copywriting/Drafting, Debugging/Troubleshooting, Direct_Answer_Generation, Ideation, Summarizing_Reading",
             "Mục đích dùng AI chính", "One-hot (5 cột 0/1)"),
            ("Post_Semester_GPA", "GPA sau học kỳ (biến mục tiêu)", "Số thực, thang 0–4"),
        ],
        columns=["Cột", "Ý nghĩa", "Kiểu / mã hóa"],
    )
    st.dataframe(dictionary, use_container_width=True, hide_index=True)
    st.caption(
        "Bộ dữ liệu gốc còn các cột khác (ví dụ mức phụ thuộc AI, lo âu khi thi, điểm giữ kỹ năng, nguy cơ kiệt sức). "
        "Nhóm không đưa chúng vào mô hình vì nhiều khả năng chúng là *kết quả* của việc dùng AI chứ không phải đầu vào, "
        "đưa vào sẽ gây rò rỉ thông tin."
    )

    st.markdown("#### 4. Hạn chế cần nêu khi thuyết trình")
    st.write(
        "- Dữ liệu quan sát, không phải thí nghiệm có đối chứng: không thể kết luận AI *gây ra* thay đổi GPA.\n"
        "- GPA trước học kỳ gần như quyết định GPA sau học kỳ; phần đóng góp của AI là phần nhỏ còn lại "
        "(xem mục *Đánh giá mô hình*, cột R² của mức thay đổi GPA).\n"
        "- Khả năng khái quát hóa sang sinh viên Việt Nam hoặc trường khác chưa được kiểm chứng."
    )

# ======================================================================
# TAB 3: ĐÁNH GIÁ MÔ HÌNH
# ======================================================================
elif menu.startswith("📊"):
    st.subheader(f"📈 Đánh giá mô hình trên tập test ({len(test_df):,} mẫu)")

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("R² (GPA sau)", f"{dt_row['R² (GPA sau)']*100:.1f}%", "Decision Tree"), unsafe_allow_html=True)
    k2.markdown(kpi("RMSE", f"{dt_row['RMSE']:.3f}", "Sai số bình phương TB (điểm GPA)", C_PRIMARY), unsafe_allow_html=True)
    k3.markdown(kpi("MAE", f"{dt_row['MAE']:.3f}", "Sai số tuyệt đối TB (điểm GPA)", C_PURPLE), unsafe_allow_html=True)
    k4.markdown(
        kpi("R² (mức thay đổi GPA)", f"{dt_row['R² (mức thay đổi GPA)']*100:.1f}%", "Phần biến thiên của (Post − Pre) mà mô hình giải thích", C_GOOD),
        unsafe_allow_html=True,
    )
    st.info(
        f"Chỉ riêng việc chép lại GPA trước học kỳ (baseline) đã cho R² = {base_row['R² (GPA sau)']*100:.1f}%. "
        "Vì vậy R² của GPA sau dễ gây hiểu nhầm là rất cao; chỉ số **R² của mức thay đổi GPA** mới phản ánh "
        "mô hình học được gì ngoài GPA cũ."
    )

    candidates = results_df.iloc[1:]
    best = candidates.loc[candidates["RMSE"].idxmin()]
    if best["Mô hình"].startswith("Decision Tree"):
        st.success("Decision Tree có RMSE thấp nhất trong các mô hình được so sánh.")
    else:
        st.info(
            f"Trên tập test, **{best['Mô hình']}** có RMSE thấp nhất ({best['RMSE']:.3f}) so với Decision Tree "
            f"({dt_row['RMSE']:.3f}). Điều này cho thấy quan hệ trong dữ liệu gần tuyến tính. "
            "Decision Tree vẫn là mô hình chính của đề tài vì dễ diễn giải (đường đi quyết định, độ quan trọng đặc trưng)."
        )

    st.markdown("#### So sánh mô hình")
    st.dataframe(
        results_df.style.format({c: "{:.3f}" for c in results_df.columns if c != "Mô hình"}),
        use_container_width=True, hide_index=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### 🏆 Top 10 đặc trưng quan trọng (Decision Tree)")
        top = fi_df.head(10).sort_values("Độ quan trọng (%)")
        fig_fi = px.bar(
            top, x="Độ quan trọng (%)", y="Đặc trưng", orientation="h",
            text=top["Độ quan trọng (%)"].map(lambda v: f"{v:.1f}%"),
            color="Độ quan trọng (%)", color_continuous_scale="Viridis",
        )
        st.plotly_chart(style_fig(fig_fi, 420), use_container_width=True)
    with right:
        st.markdown("#### 🎯 GPA thực tế vs dự đoán (300 mẫu test)")
        sample = test_df.sample(min(300, len(test_df)), random_state=42).copy()
        sample["GPA dự đoán"] = dt_model.predict(sample[FEATURE_COLS])
        fig_sc = px.scatter(
            sample, x=TARGET_COL, y="GPA dự đoán", color="Weekly_GenAI_Hours",
            labels={TARGET_COL: "GPA thực tế", "Weekly_GenAI_Hours": "Giờ AI/tuần"},
            color_continuous_scale="Plasma",
        )
        fig_sc.add_shape(type="line", x0=0, y0=0, x1=4, y1=4, line=dict(color="#16a34a", dash="dash"))
        st.plotly_chart(style_fig(fig_sc, 420), use_container_width=True)

    st.markdown("#### Chọn độ sâu cây: Train vs Test RMSE")
    fig_d = px.line(
        depth_df.melt("max_depth", var_name="Tập", value_name="RMSE"),
        x="max_depth", y="RMSE", color="Tập", markers=True,
    )
    fig_d.add_vline(x=max_depth, line_dash="dash", line_color=C_WARN)
    st.plotly_chart(style_fig(fig_d, 340), use_container_width=True)
    st.caption("Khi Train RMSE tiếp tục giảm nhưng Test RMSE không giảm (hoặc tăng), cây đang bị overfit.")

# ======================================================================
# TAB 4: KHÁM PHÁ DỮ LIỆU
# ======================================================================
elif menu.startswith("🗂️"):
    st.subheader("🗂️ Khám phá dữ liệu")
    full = pd.concat([train_df, test_df], ignore_index=True)
    full["GPA_Change"] = full[TARGET_COL] - full["Pre_Semester_GPA"]

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(kpi("Số mẫu", f"{len(full):,}", f"{len(train_df):,} train / {len(test_df):,} test"), unsafe_allow_html=True)
    m2.markdown(kpi("GPA TB trước", f"{full['Pre_Semester_GPA'].mean():.2f}", "", C_PURPLE), unsafe_allow_html=True)
    m3.markdown(kpi("GPA TB sau", f"{full[TARGET_COL].mean():.2f}", "", C_GOOD), unsafe_allow_html=True)
    m4.markdown(kpi("Thay đổi TB", f"{full['GPA_Change'].mean():+.3f}", "Post − Pre", C_BAD), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Thay đổi GPA trung bình theo giờ dùng AI / tuần")
        bins = [-0.01, 5, 10, 15, 20, 25, 30, 1000]
        labels = ["0–5", "5–10", "10–15", "15–20", "20–25", "25–30", ">30"]
        grp = (
            full.assign(Nhom=pd.cut(full["Weekly_GenAI_Hours"], bins=bins, labels=labels))
            .groupby("Nhom", observed=True)["GPA_Change"].agg(["mean", "size"]).reset_index()
        )
        fig_g = px.bar(grp, x="Nhom", y="mean", text=grp["size"].map(lambda n: f"n={n:,}"),
                       labels={"Nhom": "Giờ AI/tuần", "mean": "Thay đổi GPA TB"})
        st.plotly_chart(style_fig(fig_g, 360), use_container_width=True)
    with b:
        st.markdown("#### Thay đổi GPA trung bình theo mục đích dùng AI")
        uc = full[USECASE_COLS].idxmax(axis=1).str.replace("UseCase_", "", regex=False)
        grp2 = full.assign(UseCase=uc).groupby("UseCase")["GPA_Change"].mean().reset_index().sort_values("GPA_Change")
        fig_u = px.bar(grp2, x="GPA_Change", y="UseCase", orientation="h",
                       labels={"GPA_Change": "Thay đổi GPA TB", "UseCase": ""})
        st.plotly_chart(style_fig(fig_u, 360), use_container_width=True)
    st.caption("Chênh lệch trung bình giữa các nhóm chưa kiểm soát các yếu tố khác (GPA nền, giờ tự học...), không nên đọc là nhân quả.")

    a2, b2 = st.columns(2, gap="large")
    with a2:
        st.markdown("#### Thay đổi GPA trung bình theo giờ tự học truyền thống / tuần")
        tb_bins = [-0.01, 5, 10, 15, 20, 1000]
        tb_labels = ["0–5", "5–10", "10–15", "15–20", ">20"]
        grp3 = (
            full.assign(Nhom=pd.cut(full["Traditional_Study_Hours"], bins=tb_bins, labels=tb_labels))
            .groupby("Nhom", observed=True)["GPA_Change"].agg(["mean", "size"]).reset_index()
        )
        fig_t = px.bar(grp3, x="Nhom", y="mean", text=grp3["size"].map(lambda n: f"n={n:,}"),
                       labels={"Nhom": "Giờ tự học/tuần", "mean": "Thay đổi GPA TB"})
        st.plotly_chart(style_fig(fig_t, 340), use_container_width=True)
    with b2:
        st.markdown("#### Lưu ý khi đọc dữ liệu")
        ceiling = (full[TARGET_COL] >= 4.0).mean() * 100
        st.warning(
            f"- **Hiệu ứng trần:** {ceiling:.1f}% sinh viên có GPA sau học kỳ chạm mức tối đa 4.0, "
            "nên sinh viên GPA cao khó tăng thêm và mô hình có xu hướng dự đoán thấp cho nhóm này.\n"
            f"- GPA trung bình tăng {full['GPA_Change'].mean():+.2f} ở toàn mẫu, kể cả nhóm dùng AI ít. "
            "Vì không có nhóm đối chứng, không thể tách phần tăng do AI khỏi các yếu tố khác."
        )

    st.markdown("#### Ma trận tương quan (biến số)")
    corr = full[NUMERIC_FEATURES + [TARGET_COL, "GPA_Change"]].corr()
    st.plotly_chart(
        style_fig(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), 520),
        use_container_width=True,
    )

    st.markdown("#### Mẫu dữ liệu & thống kê mô tả")
    st.dataframe(test_df.head(20), use_container_width=True)
    st.dataframe(
        full[["Pre_Semester_GPA", "Weekly_GenAI_Hours", "Tool_Diversity", "Traditional_Study_Hours", TARGET_COL]].describe().T,
        use_container_width=True,
    )

# ======================================================================
# TAB 5: BỐI CẢNH & TÀI LIỆU
# ======================================================================
else:
    st.subheader("📚 Bối cảnh: sinh viên dùng GenAI như thế nào?")
    st.caption(
        "Các số liệu và trích dẫn dưới đây nhập tay, KHÔNG lấy từ bộ dữ liệu Kaggle. "
        "Hãy mở từng liên kết để đối chiếu trước khi đưa vào báo cáo."
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("SV từng dùng GenAI", "86%", "Digital Education Council, 2024 (3.839 SV, 16 nước)"), unsafe_allow_html=True)
    c2.markdown(kpi("Dùng AI hàng tuần", "54%", "Digital Education Council, 2024", C_GOOD), unsafe_allow_html=True)
    c3.markdown(
        kpi("Giờ AI TB / tuần (dữ liệu Nhóm 8)",
            f"{pd.concat([train_df, test_df])['Weekly_GenAI_Hours'].mean():.1f}", "Tính từ bộ dữ liệu Kaggle", C_PURPLE),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    uses = pd.DataFrame(
        {
            "Mục đích": ["Tìm kiếm thông tin", "Kiểm tra ngữ pháp", "Tóm tắt tài liệu", "Diễn đạt lại văn bản", "Tạo bản nháp đầu tiên"],
            "Tỷ lệ SV (%)": [69, 42, 33, 28, 24],
        }
    ).sort_values("Tỷ lệ SV (%)")
    fig_use = px.bar(uses, x="Tỷ lệ SV (%)", y="Mục đích", orientation="h",
                     text=uses["Tỷ lệ SV (%)"].map(lambda v: f"{v}%"))
    st.markdown("#### Mục đích dùng AI phổ biến (Digital Education Council, 2024)")
    st.plotly_chart(style_fig(fig_use, 340), use_container_width=True)
    st.caption("Số liệu nhập tay từ báo cáo Global AI Student Survey 2024; cần đối chiếu lại với báo cáo gốc.")

    st.markdown("---")
    st.subheader("📑 Tài liệu tham khảo")
    papers = [
        {
            "title": "ChatGPT for good? On opportunities and challenges of large language models for education",
            "authors": "Kasneci, E., Sessler, K., Küchemann, S., et al. (2023)",
            "journal": "Learning and Individual Differences, 103, 102274",
            "summary": "Bài tổng quan về cơ hội và thách thức của mô hình ngôn ngữ lớn trong giáo dục; nhấn mạnh nguy cơ thông tin sai và sự cần thiết của tư duy phản biện. Không phải nghiên cứu thực nghiệm về GPA.",
            "link": "https://doi.org/10.1016/j.lindif.2023.102274",
        },
        {
            "title": "Students' voices on generative AI: perceptions, benefits, and challenges in higher education",
            "authors": "Chan, C. K. Y., & Hu, W. (2023)",
            "journal": "International Journal of Educational Technology in Higher Education, 20, 43",
            "summary": "Khảo sát nhận thức của sinh viên đại học Hồng Kông về GenAI: lợi ích (hỗ trợ cá nhân hóa, viết, động não) và lo ngại (độ chính xác, đạo đức, phụ thuộc). Đo nhận thức, không đo GPA.",
            "link": "https://doi.org/10.1186/s41239-023-00411-8",
        },
        {
            "title": "Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Effects of AI on Knowledge Worker Productivity and Quality",
            "authors": "Dell'Acqua, F., McFowland III, E., Mollick, E., et al. (2023)",
            "journal": "Harvard Business School Working Paper No. 24-013",
            "summary": "Thí nghiệm với chuyên viên tư vấn (không phải sinh viên): AI giúp nhanh và chất lượng hơn ở tác vụ trong năng lực của AI, nhưng làm kết quả kém đi ở tác vụ nằm ngoài ranh giới đó.",
            "link": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4573321",
        },
        {
            "title": "Guidance for generative AI in education and research",
            "authors": "Miao, F., & Holmes, W. (2023)",
            "journal": "UNESCO",
            "summary": "Hướng dẫn chính sách của UNESCO: tiếp cận lấy con người làm trung tâm, bảo vệ quyền tự chủ của người học.",
            "link": "https://unesdoc.unesco.org/ark:/48223/pf0000386693",
        },
    ]
    for i, p in enumerate(papers, 1):
        st.markdown(
            f"""
<div class="paper-card">
  <div style="font-size:1.05rem;font-weight:700;color:{C_PRIMARY};">{i}. {p['title']}</div>
  <div style="font-size:.88rem;color:{C_MUTED};margin:4px 0;"><b>Tác giả:</b> {p['authors']} | <i>{p['journal']}</i></div>
  <div style="font-size:.92rem;color:#1e293b;margin-top:6px;">{p['summary']}</div>
  <div style="margin-top:8px;font-size:.85rem;">🔗 <a href="{p['link']}" target="_blank" style="color:{C_ACCENT};">{p['link']}</a></div>
</div>
""",
            unsafe_allow_html=True,
        )
