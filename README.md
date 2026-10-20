# Hệ thống gợi ý lộ trình học cá nhân hóa với MOOC-Radar

## 1. Giới thiệu dự án

Dự án xây dựng pipeline từ lịch sử làm bài MOOC đến hồ sơ người học và danh sách bài luyện tập được xếp hạng. Mã nguồn phục vụ assignment Modern Management Information Systems (Modern MIS); các bảng đầu ra được thiết kế để nhóm dashboard và frontend sử dụng tiếp. Ngôn ngữ thuyết minh chính là tiếng Việt, còn tên biến, lệnh và schema giữ nguyên tiếng Anh.

## 2. Bài toán và mục tiêu

Hệ thống cần nhận biết concept đã có bằng chứng chưa thành thạo, phân biệt concept chưa từng gặp, rồi đề xuất hoạt động phù hợp với khóa học đích. Các mục tiêu cụ thể là kiểm tra chất lượng dữ liệu, chuẩn hóa quan hệ problem–concept, dự đoán kết quả làm bài, ước lượng mastery, mô tả kết quả theo nhãn Bloom và xuất lộ trình có thể cập nhật khi có lượt trả lời mới.

## 3. Bộ dữ liệu MOOC-Radar

Bản cục bộ có bốn file trong `data/raw/`: `problem.json` dạng JSON Lines và ba JSON array `student-problem-{fine,middle,coarse}.json`. Bước chuẩn hóa dùng event ở bản `fine` và nối metadata problem; NCDM chọn granularity bằng config: `middle` dùng `exercise_id`, `fine` dùng quan hệ nhiều-nhiều trong `problem_concepts.csv`, còn `coarse` dùng `course_id`. Bản fine có 14.224 sinh viên, 898.933 lượt trả lời và 2.513 bài được tương tác. `problem.json` có 9.383 bài metadata, 73 khóa học và 9.985 tên concept riêng biệt. Xem [từ điển dữ liệu](docs/DATA_DICTIONARY.md) để biết trường và khóa nối thực tế. Dữ liệu gốc không bị chỉnh sửa.

## 4. Kiến trúc hệ thống

`src/moocradar/reading.py` đọc tuần tự các file lớn. Nhánh [Data Quality & EDA](notebooks/01_data_quality_eda.ipynb) phân tích **raw data trước modeling** và lưu run riêng. Nhánh mô hình chuẩn hóa dữ liệu, chia theo thời gian trong từng sinh viên, huấn luyện baseline và Bayesian Knowledge Tracing (BKT), đánh giá rồi xuất learner profile và learning path. [Tài liệu kiến trúc](docs/ARCHITECTURE.md) nêu rõ giả định và giới hạn đánh giá.

## 5. Các chức năng chính

- Kiểm tra missing values, duplicate, timestamp, thứ tự chuỗi gốc, foreign key, concept, Bloom và khả năng liên kết video.
- Xuất tối đa tám biểu đồ EDA tiếng Việt, số liệu CSV/JSON và `eda_summary.md`.
- Chuẩn hóa một lượt trả lời thành một dòng, giữ quan hệ nhiều-nhiều problem–concept.
- Baseline tỷ lệ đúng theo concept, BKT và NCDM PyTorch; lưu dự đoán, metrics, mappings, Q-matrix và checkpoint.
- Hồ sơ mastery theo concept, kết quả theo nhãn Bloom, gap có xét bằng chứng và Top-N bài luyện tập.
- Suy luận từ checkpoint với một lượt trả lời mới; glossary dịch metadata tiếng Trung ngoại tuyến.

## 6. Cấu trúc thư mục

```text
configs/                   Cấu hình mặc định
data/raw/                  Bốn file gốc, không commit vào Git
data/processed/            Dữ liệu chuẩn hóa theo chế độ smoke/full
data/translated/           Glossary dịch và bảng tiếng Anh tùy chọn
src/moocradar/             Loader, EDA, xử lý, mô hình, đánh giá, gợi ý, CLI
notebooks/                 Notebook EDA và notebook chạy mô hình trên Colab
tests/                     Kiểm thử các phần quan trọng
docs/                      Kiến trúc, từ điển dữ liệu, hợp đồng đầu ra, kết quả
artifacts/                 Mỗi lần chạy có một run_id riêng
```

## 7. Cài đặt môi trường

Yêu cầu Python 3.10 trở lên. Tại thư mục gốc dự án:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

Để mở notebook trên local, cài thêm `jupyterlab` nếu chưa có:

```bash
pip install jupyterlab
jupyter lab notebooks/01_data_quality_eda.ipynb
```

## 8. Chuẩn bị dữ liệu

Đặt bốn file gốc đúng tên trong `data/raw/`. Không tách, sửa hoặc dịch trực tiếp raw data. Có thể chỉ định vị trí khác bằng `--data-root` trên CLI hoặc các biến môi trường `MOOC_DATA_ROOT`, `MOOC_PROJECT_ROOT`, `MOOC_ARTIFACT_ROOT` trong notebook. File lớn được đọc theo từng chuỗi; không load toàn bộ JSON array chỉ để xem mẫu.

## 9. Chạy notebook Data Quality & EDA

**Chạy bước này trước huấn luyện.** Mở [`notebooks/01_data_quality_eda.ipynb`](notebooks/01_data_quality_eda.ipynb) bằng Jupyter hoặc Colab, kiểm tra `PROJECT_ROOT`/`DATA_ROOT`, chọn `MODE = "sample"` để thử nhanh hoặc `MODE = "full"` cho số liệu báo cáo, rồi chạy cell từ trên xuống. Notebook độc lập với checkpoint mô hình và không xóa bản ghi.

Có thể chạy cùng chức năng không cần notebook:

```bash
python -m moocradar eda --mode sample --sample-students 25
python -m moocradar eda --mode full
```

Mỗi lệnh tạo `artifacts/eda_<run_id>/` gồm `data_quality_report.json`, `missing_values.csv`, `dataset_statistics.csv`, `eda_summary.md`, các bảng thống kê bổ sung và `plots/eda/`. Tỷ lệ từ run mẫu chỉ mô tả những chuỗi được chọn.

## 10. Pipeline xử lý dữ liệu

Sau khi xem EDA, chạy chuẩn hóa:

```bash
python -m moocradar prepare --data-root data/raw --output-dir data/processed/full
```

Lệnh này tạo `problems.csv`, `concepts.csv`, `problem_concepts.csv`, `interactions.csv` và `data_quality.json`. Dữ liệu gốc `score` được giữ nguyên, kể cả null hoặc giá trị bất thường; modeling dùng `is_correct` gốc. Chuỗi được sắp xếp ổn định theo `submit_time` trước bước chia tập. Nếu cần chạy lại nhiều phiên bản xử lý, dùng `--output-dir` khác để giữ kết quả cũ.

Glossary dịch được tạo riêng bằng `python -m moocradar translation extract`; sau khi điền cột `english` và rà soát, dùng `python -m moocradar translation apply`. Chưa có bản dịch tự động; không gửi ID sinh viên ra dịch vụ bên ngoài.

## 11. Huấn luyện và đánh giá mô hình

```bash
python -m moocradar train --processed-dir data/processed/full --artifact-root artifacts --seed 42
```

Hoặc chạy chuẩn hóa và huấn luyện liền nhau:

```bash
python -m moocradar run --mode smoke_test --smoke-students 25
python -m moocradar run --mode full_run
```

Baseline dùng thống kê train theo concept; BKT học tham số chung và giữ trạng thái theo sinh viên–concept. Giao thức warm-start chia theo thời gian trong mỗi sinh viên. `metrics.csv` có ROC-AUC, accuracy, F1, Brier và confusion counts trên validation/test; không dùng test để chọn tham số. Xem [kết quả đo thực tế](docs/RESULTS.md). Trong run fine-grained cũ, BKT kém baseline và dự đoán toàn lớp “đúng” ở ngưỡng 0,5; run NCDM mới bổ sung metrics mất cân bằng và protocol đóng băng để diễn giải đúng hơn.

### Chạy NCDM

Nhóm thử NCDM vì mô hình học trực tiếp proficiency của từng sinh viên theo concept, difficulty theo concept của từng problem và discrimination của problem. Q-matrix giới hạn tương tác vào concept được yêu cầu; các lớp dự đoán có trọng số không âm để xác suất không giảm khi proficiency tăng. Đây là NeuralCDM, không phải MLP tùy ý. So với mã tham khảo, cơ chế projection trọng số được đóng gói trong lớp `NonNegativeLinear`, còn kích thước hidden, dropout và granularity được đưa vào config để phù hợp Colab.

```bash
python -m moocradar train \
  --model ncdm \
  --processed-dir data/processed/full \
  --artifact-root artifacts \
  --granularity middle \
  --batch-size 512 \
  --learning-rate 0.002 \
  --epochs 20 \
  --patience 4 \
  --device auto \
  --seed 42
```

`middle` là mặc định vì 898.933 event hiện có ánh xạ đầy đủ sang 580 exercise concept đang hoạt động, nhỏ hơn và ít thưa hơn 9.985 fine concept trong metadata. Nếu dùng `fine`, adapter vẫn giữ nhiều concept trên cùng problem. Một event luôn chỉ tạo một mẫu dự đoán. `id_mappings.json` và `q_matrix.npz` được dùng chung cho train/validation/test/inference; label tương lai không phải input.

Training dùng mini-batch, GPU CUDA nếu có và CPU fallback, validation mỗi epoch, early stopping theo validation ROC-AUC, progress bar và checkpoint tốt nhất. `--resume` có thể nạp cả model/optimizer state phù hợp. Test không được dùng để chọn epoch hoặc hyperparameter.

Run NCDM xuất bốn nhóm dự đoán: `smoothed_concept`, `bkt_frozen`, `ncdm` theo protocol chính đóng băng sau train, và `bkt_online` bổ sung theo protocol prequential. BKT online được dự đoán trước rồi mới nhận label validation/test; không gộp policy này với kết luận hơn-kém tuyệt đối. Metrics gồm ROC-AUC, PR-AUC, accuracy, balanced accuracy, F1, precision, recall, specificity, Brier, prevalence, positive prediction rate và confusion counts. Threshold hiện cố định 0,5 và được ghi trong manifest.

## 12. Tạo learner profile

Với BKT, `student_mastery.csv` lưu mastery cho concept đã gặp. Với NCDM, `mastery_probability`/`mastery` là `sigmoid(student_proficiency)` của model, không phải xác suất đúng của một câu hỏi. Run NCDM xuất concept đã quan sát và concept thuộc khóa học đích; `evidence_status` phân biệt `observed_model_estimate` và `unobserved_model_estimate`, còn `observation_count` chỉ đếm train evidence. `mastery_level`, ngưỡng 0,7, `confidence` và cờ dưới ba quan sát đều là heuristic, chưa được kiểm chứng sư phạm.

## 13. Đề xuất lộ trình học

`knowledge_gaps.csv` phân biệt `observed_gap` và `unobserved`. `learning_paths.csv` chỉ đề xuất bài tập có liên kết concept trực tiếp trong khóa học đích. NCDM tái sử dụng đúng module gap/ranking này; không tự suy luận prerequisite hoặc liên kết video. Vì standard NCDM là learner state tĩnh theo checkpoint, dữ liệu mới chỉ được hấp thụ khi retrain định kỳ. BKT vẫn giữ chức năng cập nhật theo từng response.

Để xuất JSON cho một sinh viên từ checkpoint:

```bash
python -m moocradar demo --model ncdm --run-dir artifacts/<run_id> --processed-dir data/processed/full --user-id U_1002476
```

Với run BKT, dùng `--model bkt --new-problem-id Pm_668265 --new-is-correct 0` để cập nhật tuần tự. Cả hai model chấp nhận `--target-course-id C_697360`. NCDM từ chối update tức thời và chỉ load checkpoint đóng băng.

## 14. Chạy trên Google Colab

Notebook EDA hỗ trợ local/Colab; trên Colab nó mount Drive, cài requirements và dùng các biến đường dẫn có thể chỉnh. Sau khi đánh giá chất lượng, mở [`notebooks/moocradar_colab.ipynb`](notebooks/moocradar_colab.ipynb), đặt `PROJECT_ROOT`, chọn `MODEL = "bkt"` hoặc `"ncdm"`, rồi chọn `smoke_test`/`full_run`. Notebook kiểm tra GPU/RAM, lưu artifacts trực tiếp lên Drive và hỗ trợ `TRAIN=False` + `SAVED_RUN_DIR` để inference từ checkpoint mà không train lại. NCDM dùng GPU nếu có nhưng không bắt buộc.

## 15. Artifacts được lưu

Run EDA và run mô hình có `run_id` khác nhau; không ghi đè âm thầm run trước. Xem [hợp đồng đầu ra](docs/OUTPUT_CONTRACT.md) để biết schema từng file, ý nghĩa của `mastery`/`probability_correct`, cách xử lý ô trống và vị trí hình. Chỉ dùng run có `manifest.json` báo `status: complete`.

## 16. Hướng dẫn tích hợp frontend

Lọc bảng lộ trình theo `user_id`, sắp xếp `rank` dạng số và đọc granularity từ manifest. Với fine, nối `concept_id` với `concepts.csv`; với middle/coarse, dùng `concept_names` trong `models/id_mappings.json`. Có thể hiển thị `concept_name_zh` trước khi bản dịch tiếng Anh được kiểm duyệt. Chỉ BKT hỗ trợ cập nhật tức thời qua `demo`/`infer`; NCDM dùng checkpoint định kỳ. Không dùng bảng mastery để tính lại metrics test.

## 17. Hạn chế và hướng phát triển

Không có liên kết video hoặc prerequisite được xác minh trong bản cục bộ; lộ trình hiện chỉ gồm bài tập. Nhãn Bloom số chưa có ánh xạ tên chính thức. Dịch tiếng Anh cần glossary được điền/rà soát. BKT dùng tham số chung và trung bình xác suất giữa concept. NCDM không mô hình hóa biến đổi năng lực theo thời gian, không hỗ trợ sinh viên/problem hoàn toàn mới và các dimension không có quan sát vẫn chỉ là model estimate. Full run đã hoàn tất nhưng vẫn cần đánh giá calibration, cold-start và hiệu quả với người học trước khi chọn model mặc định. Kiểm thử:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src
```

## 18. Tài liệu tham khảo

- [Repository MOOC-Radar](https://github.com/THU-KEG/MOOC-Radar)
- [Yu và cộng sự, bài báo MoocRadar](https://arxiv.org/abs/2304.02205)
- [Implementation NeuralCDM chính thức](https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD)
- [Wang và cộng sự, Neural Cognitive Diagnosis](https://arxiv.org/abs/1908.08733)
