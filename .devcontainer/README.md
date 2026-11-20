# 🎓 AI Academic Impact Predictor — Nhóm 8

Ứng dụng Streamlit phân tích mối liên hệ giữa thói quen dùng GenAI (ChatGPT, Gemini, Claude...) và sự thay đổi GPA của sinh viên, dùng **Decision Tree Regressor**, có so sánh với Hồi quy tuyến tính, Random Forest và một baseline.

## Cấu trúc repo

```
├── tacdongai.py        # file chính của app
├── requirements.txt
├── README.md
└── data/
    ├── train.csv       # 40.000 mẫu (80%)
    └── test.csv        # 10.000 mẫu (20%)
```

Nếu thiếu file trong `data/`, app chạy ở **chế độ DEMO** với dữ liệu giả lập và hiện cảnh báo đỏ ở sidebar. Khi chạy đúng, sidebar hiện ô xanh "Đang dùng dữ liệu thật" và banner ghi 40.000 train / 10.000 test.

## Chạy trên máy

```bash
pip install -r requirements.txt
streamlit run tacdongai.py
```

## Triển khai miễn phí

Đẩy repo lên GitHub → vào [share.streamlit.io](https://share.streamlit.io) → chọn repo, Main file path: `tacdongai.py`.

## Dữ liệu

- Nguồn: bộ dữ liệu *AI Impact on Students* trên Kaggle, 50.000 sinh viên. **Điền link Kaggle vào `DATASET_URL` trong `tacdongai.py`, ghi rõ tác giả và giấy phép.**
- `Pre_Semester_GPA` / `Post_Semester_GPA`: GPA tổng của mỗi sinh viên trước / sau học kỳ, thang 4.0. Không có điểm theo từng môn.
- Tiền xử lý: Ordinal Encoding cho năm học (1 = Freshman … 5 = Graduate) và kỹ năng prompt (1 = Beginner … 3 = Advanced); One-Hot Encoding cho ngành học và mục đích dùng AI. Tổng cộng 17 đặc trưng đầu vào.
- Dữ liệu không thiếu giá trị; mỗi sinh viên có đúng một ngành và một mục đích dùng AI chính.
- Các cột khác của bộ gốc (mức phụ thuộc AI, lo âu khi thi, điểm giữ kỹ năng, nguy cơ kiệt sức) không đưa vào mô hình vì nhiều khả năng là kết quả, không phải đầu vào.
- Nguồn gốc địa lý (Melbourne, Úc) và cách thu thập dữ liệu chưa xác minh được từ nguồn công khai. Hãy đối chiếu với phần mô tả của tác giả trên Kaggle và sửa `DATASET_ORIGIN_NOTE` cho khớp.

## Kết quả trên tập test (max_depth = 10, chọn bằng cross-validation trên tập train)

| Mô hình | R² (GPA sau) | RMSE | R² (mức thay đổi GPA) |
|---|---|---|---|
| Baseline: GPA sau = GPA trước | 0.682 | 0.277 | −1.231 |
| Hồi quy tuyến tính | 0.897 | 0.157 | 0.280 |
| **Decision Tree** | 0.889 | 0.164 | 0.220 |
| Random Forest | 0.903 | 0.153 | 0.319 |

Diễn giải: GPA trước học kỳ chiếm khoảng 95% độ quan trọng đặc trưng của cây; giờ dùng AI chỉ khoảng 0.7%. Decision Tree không phải mô hình tốt nhất ở đây (quan hệ gần tuyến tính), nhưng được giữ làm mô hình chính vì dễ diễn giải.

## Hạn chế

- Dữ liệu quan sát, không có nhóm đối chứng: không kết luận được AI *gây ra* thay đổi GPA.
- Hiệu ứng trần: khoảng 9.6% sinh viên có GPA sau học kỳ đạt tối đa 4.0.
- Chưa kiểm chứng trên sinh viên Việt Nam.
- Các số liệu và trích dẫn ở tab *Bối cảnh & tài liệu tham khảo* nhập tay, cần đối chiếu với nguồn gốc trước khi nộp.
