# 🎓 AI Academic Impact Predictor — Nhóm 8

Ứng dụng Streamlit phân tích mối liên hệ giữa thói quen dùng GenAI (ChatGPT, Gemini, Claude...) và sự thay đổi GPA của sinh viên, dùng **Decision Tree Regressor** (so sánh thêm với Hồi quy tuyến tính, Random Forest và baseline).

## Cấu trúc repo

```
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── train.csv      # hoặc train.xlsx (80%)
    └── test.csv       # hoặc test.xlsx  (20%)
```

Ứng dụng cũng nhận tên file cũ `Nhóm8_TacDongAI_train.xlsx` / `Nhóm8_TacDongAI_test.xlsx`.
Nếu không tìm thấy file, app chạy ở **chế độ DEMO** với dữ liệu giả lập và hiển thị cảnh báo đỏ.

## Chạy trên máy

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Triển khai miễn phí

Đẩy repo lên GitHub → vào [share.streamlit.io](https://share.streamlit.io) → chọn repo, file chính `app.py`.

## Dữ liệu

- Nguồn: bộ dữ liệu *AI Impact on Students* trên Kaggle (~50.000 sinh viên). **Điền link Kaggle vào biến `DATASET_URL` trong `app.py` và ghi rõ tên tác giả/giấy phép.**
- `Pre_Semester_GPA` / `Post_Semester_GPA`: GPA tổng của sinh viên trước/sau học kỳ, thang 4.0. Không có điểm theo từng môn.
- Đầu vào mô hình: 7 biến số/ordinal + 5 cột one-hot ngành học + 5 cột one-hot mục đích dùng AI = 17 đặc trưng.
- Các cột như mức phụ thuộc AI, lo âu khi thi, điểm giữ kỹ năng, nguy cơ kiệt sức **không** được đưa vào mô hình vì nhiều khả năng là kết quả, không phải đầu vào (tránh rò rỉ thông tin).
- Phần nguồn gốc địa lý (Melbourne, Úc) và cách thu thập dữ liệu cần được đối chiếu với mô tả của tác giả trên Kaggle; sửa `DATASET_ORIGIN_NOTE` cho khớp.

## Những điểm đã cải tiến so với bản đầu

- Không còn nhãn "dữ liệu thực tế" khi đang chạy dữ liệu giả lập.
- Thêm baseline "GPA sau = GPA trước" và chỉ số **R² của mức thay đổi GPA**, vì R² thông thường bị GPA cũ chi phối.
- Thêm so sánh mô hình, biểu đồ chọn `max_depth` (train vs test), cảnh báo khi nhập giá trị ngoài vùng dữ liệu, biểu đồ "nếu đổi số giờ AI".
- Tab *Về dữ liệu & cách tính GPA* (từ điển dữ liệu, giải thích GPA, hạn chế).
- Sửa trích dẫn: thay các số liệu không kiểm chứng được bằng số liệu tính từ dữ liệu; sửa tiêu đề bài báo.

## Hạn chế

Dữ liệu quan sát nên không kết luận được quan hệ nhân quả. Kết quả chưa được kiểm chứng trên sinh viên Việt Nam.
