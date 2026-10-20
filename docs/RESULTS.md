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

## NCDM: full run đã đo

Run `20261020T153506501618Z_ncdm_a83c2a` dùng toàn bộ 898.933 interaction, granularity `middle`, seed 42 và đúng split IDs 624.837/128.789/145.307. Máy local không có CUDA/MPS nên run dùng CPU, batch 2.048, hidden size 64/32, dropout 0,2 và 5 epoch. Hidden size nhỏ hơn default 256/128 để hoàn thành thực nghiệm CPU; student proficiency, problem difficulty/discrimination, Q-mask và ràng buộc đơn điệu không thay đổi. Best checkpoint theo validation ROC-AUC là epoch 5.

| Model | Validation AUC | Test AUC | Test Brier | Balanced Accuracy | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smoothed concept | 0,698879 | 0,708021 | **0,143918** | 0,530712 | **0,891036** |
| BKT frozen | 0,644992 | 0,514144 | 0,158849 | 0,505103 | 0,890336 |
| NCDM | **0,787163** | **0,725446** | 0,153843 | **0,620862** | 0,853951 |
| BKT online (protocol bổ sung) | 0,768563 | 0,781523 | 0,132744 | 0,574480 | 0,890826 |

Trong protocol chính đóng băng train-state, NCDM có AUC và balanced accuracy cao hơn baseline middle/BKT frozen, đồng thời positive prediction rate 0,809913 và specificity 0,384268 cho thấy ít lệch về lớp dương hơn. Tuy nhiên, baseline có Brier, accuracy và F1 tốt hơn NCDM. Vì vậy full run cho thấy NCDM hữu ích hơn ở ranking/phân biệt hai lớp và learner profile concept-level, nhưng chưa tốt hơn tuyệt đối về calibration hoặc quyết định tại threshold 0,5.

`bkt_online` đạt test AUC/Brier tốt nhất nhưng đã nhận các label validation/test trước đó theo thứ tự thời gian. Đây là policy triển khai khác checkpoint NCDM tĩnh; chỉ báo cáo bổ sung, không dùng để tuyên bố BKT hay NCDM tốt hơn tuyệt đối. Kết quả baseline/BKT ở bảng NCDM dùng `middle` cho cả ba model, nên không so trực tiếp với bảng fine-grained cũ ở đầu tài liệu mà bỏ qua ảnh hưởng granularity.

Run có 203/2.513 problem không xuất hiện trong train, tương ứng 96 validation event và 109 test event được đánh dấu `cold_start_problem=1`. Tất cả 898.933 split rows và prediction event IDs đã được kiểm tra không trùng/lệch. Artifacts có 528.706 mastery rows, 240.710 gaps và 71.094 recommendations cho 14.224 sinh viên, với 4.242 mẫu Top-N khác nhau. Checkpoint reload cho kết quả lặp lại đúng.

## NCDM: smoke test đã đo

Run `20261020T153032172366Z_ncdm_e23e2a` dùng đúng 25 sinh viên trong `data/processed/smoke`, granularity `middle`, seed 42, CPU, 3 epoch và hidden size 64/32 để kiểm tra tích hợp. Đây **không phải full run** và không thay thế bảng full phía trên.

| Model | Validation ROC-AUC | Test ROC-AUC | Test Brier | Test Balanced Accuracy | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smoothed concept | 0,759170 | 0,355263 | 0,220907 | 0,500000 | 0,834783 |
| BKT frozen | 0,531488 | 0,460115 | 0,213445 | 0,500000 | 0,834783 |
| BKT online (bổ sung) | 0,509689 | 0,543860 | 0,210703 | 0,500000 | 0,834783 |
| NCDM | 0,615225 | 0,553591 | 0,283065 | 0,500000 | 0,834783 |

Test prevalence là 0,716418 và mọi model dự đoán lớp dương cho 100% event tại threshold 0,5, nên specificity bằng 0. Vì run chỉ có 134 test event và 116/401 problem là cold-start đối với train, không dùng kết quả này để tuyên bố NCDM tốt hơn; kết luận chính dùng full run ở phần trên.
