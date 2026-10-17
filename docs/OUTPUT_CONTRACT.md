# Hợp đồng đầu ra cho nhóm dashboard và frontend

Mỗi lần chạy tạo thư mục `artifacts/<run_id>/` mới; không ghi đè run cũ. Kiểm tra `manifest.json` có `status: complete` trước khi dùng. CSV dùng UTF-8, có header kể cả khi không có dòng dữ liệu. ID sinh viên là ID từ bộ dữ liệu gốc và cần được xử lý phù hợp khi chia sẻ.

## Run huấn luyện mô hình

| File | Các trường chính | Cách hiểu |
| --- | --- | --- |
| `config.yaml`, `manifest.json` | Cấu hình, seed, thời điểm UTC, nguồn dữ liệu, trạng thái | Tái lập run |
| `data_quality.json` | Thống kê từ bước chuẩn hóa | Kiểm tra nguồn dữ liệu đã xử lý |
| `split_indices.csv` | `log_id,user_id,sequence_position,split` | Một lượt thuộc train/validation/test |
| `predictions.csv` | `log_id,user_id,problem_id,split,model,is_correct,probability_correct` | Một dự đoán ở cấp câu hỏi cho mỗi mô hình trên validation/test |
| `metrics.csv` | `model,split,n,roc_auc,accuracy,f1,brier,tp,tn,fp,fn` | Chỉ số tại ngưỡng 0,5; AUC để trống nếu chỉ có một lớp |
| `student_mastery.csv` | `user_id,concept_id,concept_name_zh,mastery,probability_correct,observations,confidence,insufficient_evidence` | Trạng thái BKT cuối cùng cho concept đã quan sát |
| `cognitive_profile.csv` | `user_id,cognitive_dimension,observations,correct,accuracy` | Tỷ lệ đúng thực nghiệm theo mã Bloom |
| `knowledge_gaps.csv` | `user_id,course_id,concept_id,mastery,observations,evidence_status,priority` | Tối đa 20 gap trong khóa học đích |
| `learning_paths.csv` | `user_id,course_id,rank,activity_type,problem_id,concept_ids,score,reason` | Bài tập được xếp thứ tự; `concept_ids` phân tách bằng dấu `|` |
| `models/baseline.json`, `models/bkt.json` | Tham số mô hình | Suy luận không cần train lại |
| `plots/`, `logs/training.log` | PNG, chú thích, log | Biểu đồ và tiến trình chạy |

`mastery` là xác suất trạng thái kiến thức tiềm ẩn đã học. `probability_correct` còn xét tham số guess/slip nên là đại lượng khác. `confidence = n/(n+5)` và `insufficient_evidence = n<3` chỉ là heuristic hỗ trợ hiển thị. Concept chưa quan sát không có dòng trong `student_mastery.csv`; ở `knowledge_gaps.csv`, mastery của concept đó để trống và `evidence_status=unobserved`.

Frontend lọc theo `user_id`, sắp xếp lộ trình theo `rank` dạng số, rồi nối `concept_id` với `data/processed/<mode>/concepts.csv`. Nếu có bản dịch đã điền, nối `problem_id` với `data/translated/problem_text_en.csv`; ô tiếng Anh trống nghĩa là chưa dịch. Lệnh `demo` xuất hồ sơ và lộ trình dạng JSON, có thể cập nhật với một lượt trả lời mới.

## Run Data Quality & EDA

Notebook EDA tạo `artifacts/eda_<run_id>/` độc lập với mô hình:

| File | Nội dung |
| --- | --- |
| `manifest.json` | `run_id`, chế độ mẫu/toàn bộ, trạng thái và đường dẫn nguồn |
| `data_quality_report.json` | File inventory, schema/kiểu dữ liệu, số lượng, duplicate, ngoại lệ, độ bao phủ Bloom/video và chú thích biểu đồ |
| `missing_values.csv` | `dataset,field,missing_count,total_count,missing_pct`; tỷ lệ tính theo phần trăm |
| `dataset_statistics.csv` | `section,metric,value,scope` |
| `concept_statistics.csv` | `concept_id,name_zh,interactions,correct,error_rate` |
| `course_statistics.csv` | `course_id,interactions` |
| `monthly_statistics.csv` | `month,interactions` |
| `eda_summary.md` | Tóm tắt và khuyến nghị bằng tiếng Việt |
| `plots/eda/*.png`, `plots/eda/README.md` | Tối đa tám biểu đồ 220 dpi và diễn giải |

Trong `concept_statistics.csv`, một lượt trả lời liên kết nhiều concept được tính một lần **cho mỗi concept**; đây không phải duplicate ở cấp câu hỏi. `scope=sample` chỉ phản ánh số chuỗi sinh viên đã chọn, còn metadata bài tập vẫn được quét đầy đủ. Không so sánh tỷ lệ từ run mẫu như đại diện cho toàn bộ tập dữ liệu.
