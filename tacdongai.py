import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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
    # Kiểm tra xem file của Nhóm 8 có nằm trong thư mục không
    if os.path.exists("Nhóm8_TacDongAI_train.xlsx") and os.path.exists("Nhóm8_TacDongAI_test.xlsx"):
        train_df = pd.read_excel("Nhóm8_TacDongAI_train.xlsx")
        test_df = pd.read_excel("Nhóm8_TacDongAI_test.xlsx")
        data_source = "Dữ liệu thực tế Nhóm 8 (50.000 mẫu)"
    else:
        # Tạo dữ liệu chuẩn cấu trúc 18 cột của Nhóm 8 nếu chạy demo độc lập
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
            
        # Công thức mô phỏng quan hệ thực tế của tập dữ liệu
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
        data_source = "Dữ liệu mô phỏng cấu trúc chuẩn Nhóm 8 (Hãy đặt file Excel vào cùng thư mục để dùng 50.000 dòng gốc)"

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    # Huấn luyện Decision Tree Regressor
    dt_model = DecisionTreeRegressor(max_depth=8, min_samples_split=20, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    # Tính metrics
    metrics = {
        'R2': r2_score(y_test, dt_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, dt_pred)),
        'MAE': mean_absolute_error(y_test, dt_pred)
    }

    # Feature Importance
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
        ["🔮 Dự đoán GPA Cá nhân hóa", "📊 Phân tích Mô hình Decision Tree", "🗂️ Khám phá Dữ liệu Nhóm 8"]
    )
    st.markdown("---")
    st.info("💡 **Mẹo trình bày:** Bạn có thể thay đổi các thanh trượt bên phải để xem trực tiếp cách thời gian dùng AI và kỹ năng Prompt làm tăng hoặc giảm GPA!")

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

    # Tạo vector đầu vào X khớp 100% thứ tự 17 cột đã huấn luyện
    input_dict = {col: 0 for col in FEATURE_COLS}
    input_dict['Pre_Semester_GPA'] = pre_gpa
    input_dict['Weekly_GenAI_Hours'] = ai_hours
    input_dict['Tool_Diversity'] = tool_div
    input_dict['Traditional_Study_Hours'] = trad_hours
    input_dict['Year_of_Study_Encoded'] = year_val
    input_dict['Prompt_Engineering_Skill_Encoded'] = skill_val
    input_dict['Paid_Subscription_Encoded'] = paid_val
    input_dict[major_map[major_label]] = 1
    input_dict[usecase_map[usecase_label]] = 1

    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
    predicted_gpa = float(np.clip(model.predict(input_df)[0], 0.0, 4.0))
    delta_gpa = predicted_gpa - pre_gpa

    with col_result:
        st.subheader("🎯 Kết quả Dự đoán từ Mô hình")
        
        # Biểu đồ đồng hồ đo (Gauge Chart) cực đẹp
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted_gpa,
            number={'font': {'size': 52, 'color': '#38bdf8'}, 'valueformat': '.3f'},
            delta={'reference': pre_gpa, 'increasing': {'color': '#4ade80'}, 'decreasing': {'color': '#f87171'}, 'valueformat': '+.3f'},
            title={'text': "GPA DỰ KIẾN SAU HỌC KỲ", 'font': {'size': 18, 'color': '#e2e8f0'}},
            gauge={
                'axis': {'range': [0, 4.0], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': "#6366f1"},
                'bgcolor': "rgba(15, 23, 42, 0.6)",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 2.0], 'color': 'rgba(239, 68, 68, 0.25)'},
                    {'range': [2.0, 3.2], 'color': 'rgba(234, 179, 8, 0.25)'},
                    {'range': [3.2, 4.0], 'color': 'rgba(34, 197, 94, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': "#facc15", 'width': 4},
                    'thickness': 0.75,
                    'value': pre_gpa
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=320,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Khuyến nghị thông minh từ AI Coach
        if delta_gpa >= 0.05:
            st.success(f"🚀 **Tín hiệu Tích cực (+{delta_gpa:.3f} GPA):** Bạn đang khai thác AI rất hiệu quả làm đòn bẩy học tập kết hợp với nền tảng tự học vững chắc!")
        elif delta_gpa <= -0.05:
            st.warning(f"⚠️ **Cảnh báo Suy giảm ({delta_gpa:.3f} GPA):** Mô hình nhận thấy nguy cơ giảm điểm. Hãy kiểm tra lại việc lạm dụng AI trả lời trực tiếp hoặc tăng thời gian tự học truyền thống.")
        else:
            st.info(f"⚖️ **Trạng thái Ổn định ({delta_gpa:+.3f} GPA):** Phong độ học tập duy trì ổn định so với học kỳ trước.")

        # Bảng tóm tắt khuyến nghị hành động
        with st.expander("🔍 Phân tích chi tiết các yếu tố tác động đến bạn", expanded=True):
            if usecase_label == "Tạo câu trả lời trực tiếp (Direct Answer)":
                st.write("- ❌ **Mục đích sử dụng:** Dùng AI để lấy đáp án trực tiếp thường làm giảm khả năng ghi nhớ sâu khi đi thi.")
            else:
                st.write("- ✅ **Mục đích sử dụng:** Dùng AI hỗ trợ tư duy/lập trình/tóm tắt giúp tối ưu hiệu suất học tập.")
            
            if ai_hours > 20 and trad_hours < 8:
                st.write("- ⚠️ **Cân bằng thời gian:** Thời gian dùng AI quá cao trong khi giờ tự học truyền thống thấp dễ gây lệ thuộc.")
            else:
                st.write("- ✅ **Cân bằng thời gian:** Tỷ lệ phân bổ giữa tự học và dùng AI hợp lý.")

# ==========================================
# TAB 2: PHÂN TÍCH MÔ HÌNH DECISION TREE
# ==========================================
elif menu == "📊 Phân tích Mô hình Decision Tree":
    st.subheader("📈 Hiệu năng Thuật toán Decision Tree trên Tập Kiểm thử (Test 20%)")
    
    # 3 Card KPI
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Độ chính xác giải thích (R² Score)</div>
            <div class="kpi-value">{metrics['R2']*100:.2f}%</div>
            <div class="kpi-delta" style="color: #4ade80;">Mức độ phù hợp mô hình cao</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Sai số toàn phương trung bình (RMSE)</div>
            <div class="kpi-value">{metrics['RMSE']:.4f}</div>
            <div class="kpi-delta" style="color: #38bdf8;">Độ lệch chuẩn điểm GPA</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Sai số tuyệt đối trung bình (MAE)</div>
            <div class="kpi-value">{metrics['MAE']:.4f}</div>
            <div class="kpi-delta" style="color: #c084fc;">Chênh lệch trung bình thực tế</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("#### 🏆 Top 10 Yếu tố ảnh hưởng lớn nhất đến GPA (Feature Importance)")
        top_fi = fi_df.head(10).sort_values(by='Độ quan trọng (%)', ascending=True)
        fig_fi = px.bar(
            top_fi,
            x='Độ quan trọng (%)',
            y='Đặc trưng',
            orientation='h',
            text=top_fi['Độ quan trọng (%)'].apply(lambda x: f"{x:.2f}%"),
            color='Độ quan trọng (%)',
            color_continuous_scale='Viridis'
        )
        fig_fi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            showlegend=False,
            height=420
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with c_right:
        st.markdown("#### 🎯 So sánh GPA Thực tế vs GPA Dự đoán (Mẫu ngẫu nhiên)")
        sample_test = test_df.sample(min(300, len(test_df)), random_state=42).copy()
        sample_test['GPA_Du_Doan'] = model.predict(sample_test[FEATURE_COLS])
        
        fig_scatter = px.scatter(
            sample_test,
            x='Post_Semester_GPA',
            y='GPA_Du_Doan',
            color='Weekly_GenAI_Hours',
            labels={
                'Post_Semester_GPA': 'GPA Thực tế',
                'GPA_Du_Doan': 'GPA Mô hình Dự đoán',
                'Weekly_GenAI_Hours': 'Giờ dùng AI/tuần'
            },
            color_continuous_scale='Plasma'
        )
        fig_scatter.add_shape(
            type="line", x0=0, y0=0, x1=4, y1=4,
            line=dict(color="#4ade80", width=2, dash="dash")
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            height=420
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TAB 3: KHÁM PHÁ DỮ LIỆU NHÓM 8
# ==========================================
elif menu == "🗂️ Khám phá Dữ liệu Nhóm 8":
    st.subheader("📋 Cấu trúc Dữ liệu đã Mã hóa của Nhóm 8")
    st.write("Dữ liệu hiển thị dưới đây là tập kiểm thử (`Nhóm8_TacDongAI_test.xlsx`) đã hoàn tất tiền xử lý **Ordinal Encoding** và **One-Hot Encoding** với 18 cột số nguyên bản.")
    
    st.dataframe(test_df.head(20), use_container_width=True)
    
    st.markdown("#### 📊 Thống kê mô tả các biến định lượng chính")
    st.dataframe(
        test_df[['Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Tool_Diversity', 'Traditional_Study_Hours', 'Post_Semester_GPA']].describe().T,
        use_container_width=True
    )
