# Hợp đồng đầu ra cho nhóm dashboard và frontend

Mỗi lần chạy tạo thư mục `artifacts/<run_id>/` mới; không ghi đè run cũ. Kiểm tra `manifest.json` có `status: complete` trước khi dùng. CSV dùng UTF-8, có header kể cả khi không có dòng dữ liệu. ID sinh viên là ID từ bộ dữ liệu gốc và cần được xử lý phù hợp khi chia sẻ.

## Run huấn luyện mô hình

| File | Các trường chính | Cách hiểu |
| --- | --- | --- |
| `config.yaml`, `manifest.json` | Cấu hình, seed, thời điểm UTC, nguồn dữ liệu, trạng thái | Tái lập run |
| `data_quality.json` | Thống kê từ bước chuẩn hóa | Kiểm tra nguồn dữ liệu đã xử lý |
| `split_indices.csv` | `log_id,user_id,sequence_position,split` | Một lượt thuộc train/validation/test |
| `predictions.csv` | Trường lõi: `log_id,user_id,problem_id,split,model,is_correct,probability_correct`; run NCDM thêm `evaluation_protocol,cold_start_problem` | Một dự đoán ở cấp câu hỏi cho mỗi mô hình trên validation/test |
| `metrics.csv` | `model,split,n,threshold,roc_auc,pr_auc,accuracy,balanced_accuracy,f1,precision,recall,specificity,brier,prevalence,positive_prediction_rate,tp,tn,fp,fn` | Chỉ số tại threshold đã khai báo; AUC để trống nếu chỉ có một lớp |
| `student_mastery.csv` | Trường BKT cũ và, với NCDM: `mastery_probability,mastery_level,observation_count,evidence_status,model_name,run_id` | Proficiency theo student–concept cùng mức bằng chứng |
| `cognitive_profile.csv` | `user_id,cognitive_dimension,observations,correct,accuracy` | Tỷ lệ đúng thực nghiệm theo mã Bloom |
| `knowledge_gaps.csv` | `user_id,course_id,concept_id,mastery,observations,evidence_status,priority` | Tối đa 20 gap trong khóa học đích |
| `learning_paths.csv` | `user_id,course_id,rank,activity_type,problem_id,concept_ids,score,reason` | Bài tập được xếp thứ tự; `concept_ids` phân tách bằng dấu `|` |
| `models/baseline.json`, `models/bkt.json` | Tham số mô hình | Suy luận không cần train lại |
| `models/best_ncdm.pt` | Model state, kiến trúc, best epoch và optimizer state | Load NCDM/inference hoặc resume không cần train từ đầu |
| `models/id_mappings.json`, `models/q_matrix.npz` | Mapping ổn định và Q-matrix COO nén | Khôi phục đúng ID/granularity khi inference |
| `training_history.csv` | `epoch,train_loss,validation_roc_auc,validation_brier` | Dữ liệu gốc của training curves |
| `plots/`, `logs/training.log` | PNG và log | Loss, validation AUC, ROC, PR, confusion matrix, probability distribution |

Trong run BKT, `mastery` là xác suất trạng thái tiềm ẩn đã học và `probability_correct` còn xét guess/slip. Trong run NCDM, `mastery` và `mastery_probability` đều lấy từ `sigmoid(student_proficiency)`; `probability_correct` để trống vì đại lượng đó chỉ có nghĩa khi kết hợp thêm problem, Q-row, difficulty, discrimination và prediction network. Không dùng xác suất đúng của một problem thay cho mastery.

`confidence = n/(n+5)`, `insufficient_evidence = n<3`, `mastery_level` và threshold mastery là heuristic hiển thị. NCDM có thể xuất proficiency cho concept chưa quan sát trong khóa học đích, nhưng dòng đó phải có `observation_count=0`, `insufficient_evidence=1`, `evidence_status=unobserved_model_estimate`; đây không phải năng lực đã được xác nhận. Run BKT cũ chỉ có dòng cho concept đã quan sát.

Frontend lọc theo `user_id`, sắp xếp lộ trình theo `rank` dạng số và đọc `manifest.json` để biết model/granularity. Với `fine`, nối `concept_id` với `data/processed/<mode>/concepts.csv`; với `middle/coarse`, dùng `concept_names` trong `models/id_mappings.json` (hiện là ID exercise/course). Nếu có bản dịch đã điền, nối `problem_id` với `data/translated/problem_text_en.csv`; ô tiếng Anh trống nghĩa là chưa dịch.

Lệnh `demo --model ncdm` load `best_ncdm.pt`, mappings và Q-matrix. NCDM không nhận `--new-problem-id/--new-is-correct` vì chưa có cơ chế update thời gian thực; frontend dùng BKT cho luồng đó hoặc chờ checkpoint NCDM retrain định kỳ.

## Chính sách so sánh trong run NCDM

`smoothed_concept`, `bkt_frozen` và `ncdm` là protocol chính: chỉ dùng train để fit/trạng thái, sau đó checkpoint đóng băng trên validation/test. `bkt_online` là kết quả bổ sung: dự đoán sự kiện trước, rồi mới cập nhật bằng label của sự kiện đó cho lần tiếp theo. Không so sánh `bkt_online` với NCDM như hai policy lịch sử giống nhau. Threshold mặc định là 0,5; nếu sau này chọn threshold khác, chỉ được fit quy tắc trên validation và phải ghi vào config/manifest.

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
