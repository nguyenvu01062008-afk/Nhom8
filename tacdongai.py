import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# ==========================================
# 1. CẤU HÌNH TRANG & CSS TÙY CHỈNH (UI/UX ĐỈNH CAO)
# ==========================================
st.set_page_config(
    page_title="AI Impact Analytics | Nhóm 8",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Nền và Font chữ tổng thể */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Card hiển thị chỉ số KPI */
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: #38bdf8;
    }
    .kpi-title {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-delta {
        font-size: 0.85rem;
        margin-top: 5px;
    }

    /* Card bài báo & nghiên cứu khoa học */
    .article-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .article-card:hover {
        transform: translateY(-4px);
        border-color: #38bdf8;
        box-shadow: 0 8px 30px rgba(56, 189, 248, 0.2);
    }
    .article-tag {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    .article-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    .article-meta {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 12px;
    }
    .article-summary {
        font-size: 0.95rem;
        color: #cbd5e1;
        line-height: 1.6;
    }

    /* Header Banner */
    .hero-banner {
        background: linear-gradient(90deg, #1e1b4b 0%, #312e81 50%, #1e3a8a 100%);
        padding: 2.2rem;
        border-radius: 20px;
        border: 1px solid rgba(99, 102, 241, 0.4);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #c7d2fe;
    }

    /* Nút bấm */
    .stButton>button {
        background: linear-gradient(90deg, #0284c7, #4f46e5);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HÀM TẢI DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH
# ==========================================
FEATURE_COLS = [
    'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Tool_Diversity', 'Traditional_Study_Hours',
    'Year_of_Study_Encoded', 'Prompt_Engineering_Skill_Encoded', 'Paid_Subscription_Encoded',
    'Major_Arts', 'Major_Business', 'Major_Humanities', 'Major_Medical', 'Major_STEM',
    'UseCase_Copywriting/Drafting', 'UseCase_Debugging/Troubleshooting',
    'UseCase_Direct_Answer_Generation', 'UseCase_Ideation', 'UseCase_Summarizing_Reading'
]
TARGET_COL = 'Post_Semester_GPA'

@st.cache_data
def load_and_train_models():
    if os.path.exists("Nhóm8_TacDongAI_train.xlsx") and os.path.exists("Nhóm8_TacDongAI_test.xlsx"):
        train_df = pd.read_excel("Nhóm8_TacDongAI_train.xlsx")
        test_df = pd.read_excel("Nhóm8_TacDongAI_test.xlsx")
        data_source = "Dữ liệu thực tế Nhóm 8 (50.000 mẫu)"
    else:
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            'Pre_Semester_GPA': np.round(np.random.uniform(1.8, 4.0, n), 3),
            'Weekly_GenAI_Hours': np.round(np.random.uniform(0, 35, n), 2),
            'Tool_Diversity': np.random.randint(1, 6, n),
            'Traditional_Study_Hours': np.round(np.random.uniform(2, 25, n), 2),
            'Year_of_Study_Encoded': np.random.randint(1, 6, n),
            'Prompt_Engineering_Skill_Encoded': np.random.randint(1, 4, n),
            'Paid_Subscription_Encoded': np.random.randint(0, 2, n),
        })
        majors = np.random.choice(['Major_Arts', 'Major_Business', 'Major_Humanities', 'Major_Medical', 'Major_STEM'], n)
        for m in ['Major_Arts', 'Major_Business', 'Major_Humanities', 'Major_Medical', 'Major_STEM']:
            df[m] = (majors == m).astype(int)
            
        usecases = np.random.choice([
            'UseCase_Copywriting/Drafting', 'UseCase_Debugging/Troubleshooting',
            'UseCase_Direct_Answer_Generation', 'UseCase_Ideation', 'UseCase_Summarizing_Reading'
        ], n)
        for u in [
            'UseCase_Copywriting/Drafting', 'UseCase_Debugging/Troubleshooting',
            'UseCase_Direct_Answer_Generation', 'UseCase_Ideation', 'UseCase_Summarizing_Reading'
        ]:
            df[u] = (usecases == u).astype(int)
            
        gpa_delta = (
            0.015 * df['Traditional_Study_Hours']
            + 0.08 * df['Prompt_Engineering_Skill_Encoded']
            - 0.012 * np.maximum(0, df['Weekly_GenAI_Hours'] - 15)
            - 0.15 * df['UseCase_Direct_Answer_Generation']
            + 0.06 * df['UseCase_Ideation']
            + np.random.normal(0, 0.12, n)
        )
        df[TARGET_COL] = np.clip(np.round(df['Pre_Semester_GPA'] + gpa_delta, 3), 0.0, 4.0)
        
        train_df = df.iloc[:4000]
        test_df = df.iloc[4000:]
        data_source = "Dữ liệu mô phỏng cấu trúc chuẩn Nhóm 8"

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    dt_model = DecisionTreeRegressor(max_depth=8, min_samples_split=20, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    metrics = {
        'R2': r2_score(y_test, dt_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, dt_pred)),
        'MAE': mean_absolute_error(y_test, dt_pred)
    }

    fi_df = pd.DataFrame({
        'Đặc trưng': FEATURE_COLS,
        'Độ quan trọng (%)': dt_model.feature_importances_ * 100
    }).sort_values(by='Độ quan trọng (%)', ascending=False)

    return dt_model, metrics, fi_df, test_df, data_source

model, metrics, fi_df, test_df, data_source = load_and_train_models()

# ==========================================
# 3. GIAO DIỆN CHÍNH (HERO BANNER & SIDEBAR)
# ==========================================
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">🎓 AI ACADEMIC IMPACT PREDICTOR — NHÓM 8</div>
    <div class="hero-subtitle">
        Hệ thống phân tích & dự đoán tác động của Trí tuệ Nhân tạo (GenAI) đến kết quả GPA sinh viên sử dụng thuật toán <b>Decision Tree Regressor</b>
    </div>
    <div style="margin-top: 12px; font-size: 0.85rem; color: #93c5fd;">
        📂 Nguồn dữ liệu đang chạy: <b>{data_source}</b>
    </div>
</div>
""", unsafe_allow_html=True)

# Thanh điều hướng bên trái (Sidebar)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80)
    st.title("Bảng Điều Khiển")
    st.markdown("---")
    menu = st.radio(
        "Chọn phân hệ chức năng:",
        [
            "🔮 Dự đoán GPA Cá nhân hóa",
            "📊 Phân tích Mô hình Decision Tree",
            "📰 Góc Nhìn Khoa Học & Bài Báo AI",
            "🗂️ Khám phá Dữ liệu Nhóm 8"
        ]
    )
    st.markdown("---")
    st.info("💡 **Mẹo trình bày:** Bạn có thể thay đổi các thanh trượt ở mục Dự đoán để xem trực tiếp cách thời gian dùng AI và kỹ năng Prompt tác động đến GPA!")

# ==========================================
# TAB 1: DỰ ĐOÁN GPA CÁ NHÂN HÓA
# ==========================================
if menu == "🔮 Dự đoán GPA Cá nhân hóa":
    col_input, col_result = st.columns([1.15, 1], gap="large")

    with col_input:
        st.subheader("📝 Nhập thông tin học tập & Thói quen dùng AI")
        
        c1, c2 = st.columns(2)
        with c1:
            pre_gpa = st.slider("GPA trước học kỳ (Thang 4.0)", 0.0, 4.0, 3.20, 0.05)
            ai_hours = st.slider("Số giờ dùng AI / tuần", 0.0, 40.0, 8.5, 0.5)
            trad_hours = st.slider("Số giờ tự học truyền thống / tuần", 0.0, 35.0, 14.0, 0.5)
        
        with c2:
            tool_div = st.slider("Độ đa dạng công cụ AI (1-5 công cụ)", 1, 5, 3)
            
            year_map = {
                "Năm nhất (Freshman)": 1,
                "Năm hai (Sophomore)": 2,
                "Năm ba (Junior)": 3,
                "Năm cuối (Senior)": 4,
                "Sau đại học (Graduate)": 5
            }
            year_label = st.selectbox("Năm học hiện tại", list(year_map.keys()), index=2)
            year_val = year_map[year_label]

            skill_map = {
                "Mới bắt đầu (Beginner)": 1,
                "Trung cấp (Intermediate)": 2,
                "Nâng cao (Advanced)": 3
            }
            skill_label = st.selectbox("Kỹ năng viết câu lệnh (Prompt)", list(skill_map.keys()), index=1)
            skill_val = skill_map[skill_label]

        c3, c4 = st.columns(2)
        with c3:
            major_map = {
                "Kỹ thuật & Công nghệ (STEM)": "Major_STEM",
                "Kinh doanh (Business)": "Major_Business",
                "KHXH & Nhân văn (Humanities)": "Major_Humanities",
                "Y khoa (Medical)": "Major_Medical",
                "Nghệ thuật (Arts)": "Major_Arts"
            }
            major_label = st.selectbox("Khối ngành học", list(major_map.keys()))
            
        with c4:
            usecase_map = {
                "Lên ý tưởng sáng tạo (Ideation)": "UseCase_Ideation",
                "Lập trình / Gỡ lỗi (Debugging)": "UseCase_Debugging/Troubleshooting",
                "Tóm tắt / Đọc tài liệu (Summarizing)": "UseCase_Summarizing_Reading",
                "Soạn thảo văn bản (Copywriting)": "UseCase_Copywriting/Drafting",
                "Tạo câu trả lời trực tiếp (Direct Answer)": "UseCase_Direct_Answer_Generation"
            }
            usecase_label = st.selectbox("Mục đích dùng AI chính", list(usecase_map.keys()))

        paid_sub = st.toggle("Đăng ký gói AI trả phí (ChatGPT Plus, Claude Pro, Gemini Advanced...)", value=True)
        paid_val = 1 if paid_sub else 0

    input_dict = {col: 0 for col in FEATURE_COLS}
    input_dict['Pre_Semester_GPA'] = pre_gpa
    input_dict['Weekly_GenAI_Hours'] = ai
