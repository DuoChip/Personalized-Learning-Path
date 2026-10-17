# Kết quả đã đo trên bộ dữ liệu cục bộ

## Phạm vi thực nghiệm

Run mô hình `20261017T150803777229Z_009410` hoàn tất ngày 17/10/2026 với seed 42. File fine cục bộ có 14.224 sinh viên và 898.933 lượt trả lời. Chia theo thời gian trong từng sinh viên tạo 624.837 lượt train, 128.789 validation và 145.307 test. Run EDA toàn bộ `eda_20261017T152225465617Z_c680fb` nằm trong `artifacts/` riêng và có `manifest.json` ghi phạm vi.

## Chất lượng dữ liệu gốc

- `problem.json` có 9.383 bài, 73 khóa học, 9.985 concept theo tên; 7.203 bài gắn nhiều concept. Bốn liên kết problem–concept trùng trong danh sách metadata không phải lượt trả lời trùng.
- `score` thiếu ở 543.847/898.933 lượt (60,50%); thêm 114 giá trị `score = -1.0` cần kiểm tra quy ước. `is_correct` hiện diện ở mọi lượt.
- Không phát hiện `log_id` trùng, bản ghi trùng hoàn toàn, foreign key `problem_id` sai, timestamp không hợp lệ hoặc nhãn đúng/sai bất thường trong lần quét toàn bộ.
- Có 8.168 chuỗi và 95.384 cặp thời gian đi lùi theo thứ tự gốc. Vì vậy bước chuẩn hóa cần sắp xếp ổn định từng chuỗi trước khi chia theo thời gian.
- Độ bao phủ nhãn Bloom ở cấp problem và lượt trả lời là 100% theo mã 1–6. Không thấy trường liên kết video tường minh trong metadata cục bộ.

## Dự đoán trên validation và test

| Mô hình | Tập | ROC-AUC | Accuracy | F1 | Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| Smoothed concept baseline | Validation | 0,735351 | 0,827276 | 0,903734 | 0,129782 |
| Smoothed concept baseline | Test | 0,738546 | 0,811255 | 0,894076 | 0,138044 |
| BKT | Validation | 0,663296 | 0,820435 | 0,901361 | 0,138928 |
| BKT | Test | 0,653821 | 0,803320 | 0,890934 | 0,148772 |

Trong test, 80,3% lượt có nhãn đúng. BKT dự đoán lớp “đúng” cho tất cả lượt ở ngưỡng 0,5: 116.728 true positive, 28.579 false positive và 0 true negative. Do đó accuracy của BKT chịu ảnh hưởng lớn của tỷ lệ lớp; ROC-AUC và Brier cho thêm góc nhìn. Baseline tốt hơn BKT trong bản trích xuất này. Không dùng test để chọn tham số.

## Artifacts và giới hạn

Run mô hình đã xuất 1.074.367 dòng mastery sinh viên–concept, 51.156 dòng kết quả Bloom, 266.497 dòng gap trong khóa học đích và 69.874 dòng lộ trình. Ba sinh viên đầu tiên có các Top-5 khác nhau. Run EDA xuất tám hình PNG và các bảng thống kê cục bộ để dùng trong báo cáo.

Các con số này chỉ mô tả file hiện có; không phải benchmark của bài báo và không chứng minh lộ trình làm tăng kết quả học tập. Điểm đề xuất, ngưỡng mastery và confidence là heuristic. Chưa có đánh giá với người học/giảng viên, prerequisite được xác minh hoặc tài nguyên video có liên kết.
