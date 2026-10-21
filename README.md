# Hệ thống gợi ý lộ trình học cá nhân hóa với MOOC-Radar

## 1. Giới thiệu dự án

Dự án xây dựng pipeline từ lịch sử làm bài MOOC đến hồ sơ người học và danh sách bài luyện tập được xếp hạng. Mã nguồn phục vụ assignment Modern Management Information Systems (Modern MIS); các bảng đầu ra được thiết kế để nhóm dashboard và frontend sử dụng tiếp. Ngôn ngữ thuyết minh chính là tiếng Việt, còn tên biến, lệnh và schema giữ nguyên tiếng Anh.

## 2. Bài toán và mục tiêu

Hệ thống cần nhận biết concept đã có bằng chứng chưa thành thạo, phân biệt concept chưa từng gặp, rồi đề xuất hoạt động phù hợp với khóa học đích. Các mục tiêu cụ thể là kiểm tra chất lượng dữ liệu, chuẩn hóa quan hệ problem–concept, dự đoán kết quả làm bài, ước lượng mastery, mô tả kết quả theo nhãn Bloom và xuất lộ trình có thể cập nhật khi có lượt trả lời mới.

### Phạm vi đã triển khai

```text
Lịch sử làm bài + danh mục bài tập + mapping bài–concept
    -> learner profile theo concept
    -> knowledge gaps
    -> Top-N bài luyện tập được xếp hạng
```

| Khả năng | Trạng thái hiện tại |
| --- | --- |
| Dự đoán xác suất sinh viên làm đúng một bài | Đã triển khai và có metrics validation/test |
| Tạo hồ sơ năng lực theo concept | Đã triển khai |
| Phân biệt gap có bằng chứng và concept chưa quan sát | Đã triển khai |
| Tạo Top-N bài luyện tập theo gap và khóa học | Đã triển khai |
| Hai sinh viên cùng khóa nhận lộ trình khác nhau | Đã kiểm tra trên artifact chính thức |
| Giải thích lý do đề xuất | Đã có ở mức liên kết bài–concept–gap |
| Load checkpoint để inference không cần train lại | Đã triển khai |
| Cập nhật sau response mới | BKT hỗ trợ trong phiên inference; NCDM cần retrain định kỳ |
| Lịch học theo deadline, ví dụ kế hoạch thi sau 3 tháng | Chưa triển khai |
| Tối ưu theo mục tiêu `fast_track`, `deep_learning` hoặc `exam_preparation` | Chưa triển khai |
| Chứng minh lộ trình làm tăng kết quả học tập | Chưa có thí nghiệm người học |

Vì vậy, “learning path” trong phiên bản hiện tại là **danh sách bài luyện tập cá nhân hóa theo knowledge gaps**, không phải lịch học theo tuần/tháng hoặc kế hoạch tối ưu theo deadline.

## 3. Bộ dữ liệu MOOC-Radar

Bản cục bộ có bốn file trong `data/raw/`: `problem.json` dạng JSON Lines và ba JSON array `student-problem-{fine,middle,coarse}.json`. Bước chuẩn hóa dùng event ở bản `fine` và nối metadata problem; NCDM chọn granularity bằng config: `middle` dùng `exercise_id`, `fine` dùng quan hệ nhiều-nhiều trong `problem_concepts.csv`, còn `coarse` dùng `course_id`. Bản fine có 14.224 sinh viên, 898.933 lượt trả lời và 2.513 bài được tương tác. `problem.json` có 9.383 bài metadata, 73 khóa học và 9.985 tên concept riêng biệt. Xem [từ điển dữ liệu](docs/DATA_DICTIONARY.md) để biết trường và khóa nối thực tế. Dữ liệu gốc không bị chỉnh sửa.

## 4. Kiến trúc hệ thống

`src/moocradar/reading.py` đọc tuần tự các file lớn. Nhánh [Data Quality & EDA](notebooks/01_data_quality_eda.ipynb) phân tích **raw data trước modeling** và lưu run riêng. Nhánh mô hình chuẩn hóa dữ liệu, chia theo thời gian trong từng sinh viên, huấn luyện baseline, Bayesian Knowledge Tracing (BKT) và NCDM, đánh giá rồi xuất learner profile và learning path. [Tài liệu thuật toán](docs/ALGORITHMS.md) giải thích vai trò của từng phương pháp; [tài liệu kiến trúc](docs/ARCHITECTURE.md) nêu rõ giả định và giới hạn đánh giá.

## 5. Các chức năng chính

- Kiểm tra missing values, duplicate, timestamp, thứ tự chuỗi gốc, foreign key, concept, Bloom và khả năng liên kết video.
- Xuất tối đa tám biểu đồ EDA tiếng Việt, số liệu CSV/JSON và `eda_summary.md`.
- Chuẩn hóa một lượt trả lời thành một dòng, giữ quan hệ nhiều-nhiều problem–concept.
- Baseline tỷ lệ đúng theo concept, BKT và NCDM PyTorch; lưu dự đoán, metrics, mappings, Q-matrix và checkpoint.
- Hồ sơ mastery theo concept, kết quả theo nhãn Bloom, gap có xét bằng chứng và Top-N bài luyện tập.
- Load checkpoint để xuất learner profile và recommendation; BKT có thể nhận thêm một response trong phiên inference. Glossary dịch metadata tiếng Trung được xử lý ngoại tuyến.

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
artifacts/                 Run chính thức, run EDA độc lập và thư mục OLD lưu lịch sử
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

### Vai trò của các phương pháp

| Phương pháp | Vai trò trong dự án |
| --- | --- |
| **BKT Frozen** (`bkt_frozen`) | Thử nghiệm đối chứng: dùng tham số và trạng thái học từ train, không cập nhật trạng thái bằng nhãn validation/test. |
| **BKT Online** (`bkt_online`) | Chính sách vận hành tuần tự: dự đoán response hiện tại trước, sau đó dùng kết quả đã quan sát để cập nhật trạng thái kiến thức và đề xuất nội dung tiếp theo. |
| **NCDM** (`ncdm`) | Phương pháp bổ sung để học proficiency của sinh viên theo concept, difficulty/discrimination của bài và tạo learner profile nhiều chiều. |

BKT Frozen và BKT Online dùng cùng cơ chế BKT; khác biệt nằm ở chính sách cập nhật trạng thái. BKT Online chỉ dùng lịch sử đã quan sát, không dùng nhãn hiện tại trước khi dự đoán hoặc nhãn tương lai. NCDM hiện dùng checkpoint tĩnh và cần retrain/fine-tune định kỳ để hấp thụ dữ liệu mới. Xem giải thích đầy đủ tại [docs/ALGORITHMS.md](docs/ALGORITHMS.md).

### Hiểu nhanh đầu vào và đầu ra

Ba file processed cung cấp đầu vào chính: `interactions.csv` chứa lịch sử và nhãn đúng/sai, `problems.csv` chứa metadata bài, còn `problem_concepts.csv` chứa mapping nhiều-nhiều ở mức fine. Run `middle` hiện tại dùng `exercise_id` làm concept. Từ đó pipeline tạo hai nhóm kết quả:

- `predictions.csv` và `metrics.csv`: dùng để đánh giá xác suất làm đúng từng bài của baseline, BKT Frozen, BKT Online và NCDM.
- `student_mastery.csv` → `knowledge_gaps.csv` → `learning_paths.csv`: dùng proficiency theo concept của NCDM để tạo hồ sơ, tìm điểm yếu và xếp hạng bài luyện tập.

`probability_correct` là xác suất đúng của một **bài cụ thể**; `mastery_probability` là ước lượng năng lực trên một **concept**; `score` trong learning path chỉ là điểm xếp hạng recommendation. Không dùng ba giá trị này thay thế lẫn nhau. Xem ví dụ bằng dữ liệu thật và bảng chọn output tại [docs/ALGORITHMS.md](docs/ALGORITHMS.md#luồng-đầu-vào-và-đầu-ra-trong-dự-án).

```bash
python -m moocradar train --processed-dir data/processed/full --artifact-root artifacts --seed 42
```

Hoặc chạy chuẩn hóa và huấn luyện liền nhau:

```bash
python -m moocradar run --mode smoke_test --smoke-students 25
python -m moocradar run --mode full_run
```

Baseline dùng thống kê train theo concept; BKT học tham số chung và giữ trạng thái theo sinh viên–concept. Dữ liệu được chia theo thời gian trong mỗi sinh viên. `metrics.csv` có ROC-AUC, accuracy, F1, Brier và confusion counts trên validation/test; không dùng test để chọn tham số. Xem [kết quả đo thực tế](docs/RESULTS.md). Các run fine-grained trước đây đã được lưu trong `artifacts/OLD/`; kết luận chính hiện dùng artifact NCDM full-run chính thức với cùng granularity `middle` cho các model so sánh.

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

### Bằng chứng cá nhân hóa trong artifact chính thức

Hai sinh viên dưới đây cùng thuộc khóa `C_697360`, nhưng có knowledge gaps khác nhau:

| Sinh viên | Concept yếu có bằng chứng | Mastery |
| --- | --- | ---: |
| `U_1002476` | `Ex_535154` | 0,494601 |
| `U_1002476` | `Ex_535156` | 0,542724 |
| `U_10125756` | `Ex_535315` | 0,478074 |
| `U_10125756` | `Ex_535316` | 0,540771 |

Kết quả là hai Top-5 khác nhau:

| Rank | `U_1002476` | `U_10125756` |
| ---: | --- | --- |
| 1 | `Pm_668281` – `Ex_535156` | `Pm_668450` – `Ex_535316` |
| 2 | `Pm_1256155` – `Ex_535150` | `Pm_668451` – `Ex_535316` |
| 3 | `Pm_668270` – `Ex_535150` | `Pm_668452` – `Ex_535316` |
| 4 | `Pm_668272` – `Ex_535151` | `Pm_668445` – `Ex_535315` |
| 5 | `Pm_668277` – `Ex_535154` | `Pm_668446` – `Ex_535315` |

Ví dụ này chứng minh pipeline tạo lộ trình khác nhau từ profile khác nhau ngay cả khi cùng khóa học. Tính phù hợp kỹ thuật được kiểm tra bằng việc bài đề xuất thuộc khóa đích và có mapping trực tiếp tới concept gap. Đây chưa phải bằng chứng rằng lộ trình tối ưu về mặt sư phạm hoặc làm tăng kết quả học tập.

### Những gì chưa phải là “lộ trình” trong phiên bản hiện tại

Hệ thống chưa nhận các đầu vào như `exam_date`, `hours_per_week`, `learning_goal` hoặc phạm vi kiến thức của một kỳ thi. Do đó, hệ thống chưa thể tạo kế hoạch dạng “12 tuần để thi”, chưa tối ưu số giờ học và chưa thể khẳng định một lộ trình là “học nhanh”. Giảm `top_n` chỉ làm danh sách ngắn hơn, không chứng minh đó là kế hoạch nhanh nhất.

Để hỗ trợ mục tiêu này trong tương lai, có thể bổ sung một `Goal-based Learning Planner` sau bước knowledge-gap detection:

```text
Learner profile + knowledge gaps
    + mục tiêu + deadline + số giờ/tuần + phạm vi thi
    -> kế hoạch theo tuần
    -> checkpoint đánh giá tiến độ
    -> điều chỉnh các tuần còn lại
```

Planner còn cần metadata chưa có đầy đủ trong MOOC-Radar như thời lượng hoạt động, syllabus kỳ thi và prerequisite đã được xác minh. Nếu dùng giá trị ước lượng, artifact phải đánh dấu rõ là heuristic.

Để xuất JSON cho một sinh viên từ checkpoint:

```bash
python -m moocradar demo --model ncdm --run-dir artifacts/20261020T160618189951Z_ncdm_3d00a8 --processed-dir data/processed/full --user-id U_1002476
```

Với run BKT, dùng `--model bkt --new-problem-id Pm_668265 --new-is-correct 0` để cập nhật tuần tự. Cả hai model chấp nhận `--target-course-id C_697360`. NCDM từ chối update tức thời và chỉ load checkpoint đóng băng.

## 14. Chạy trên Google Colab

Notebook EDA hỗ trợ local/Colab; trên Colab nó mount Drive, cài requirements và dùng các biến đường dẫn có thể chỉnh. Sau khi đánh giá chất lượng, mở [`notebooks/moocradar_colab.ipynb`](notebooks/moocradar_colab.ipynb), đặt `PROJECT_ROOT`, chọn `MODEL = "bkt"` hoặc `"ncdm"`, rồi chọn `smoke_test`/`full_run`. Notebook kiểm tra GPU/RAM, lưu artifacts trực tiếp lên Drive và hỗ trợ `TRAIN=False` + `SAVED_RUN_DIR` để inference từ checkpoint mà không train lại. NCDM dùng GPU nếu có nhưng không bắt buộc.

Kết quả chạy trên Colab được lưu tại [thư mục Google Drive của dự án](https://drive.google.com/drive/folders/1o07j0q612DXwAVOrbIzeioMYRTqwKZ44?usp=sharing).

## 15. Artifacts được lưu

Artifact mô hình chính thức của dự án là `artifacts/20261020T160618189951Z_ncdm_3d00a8/`. Run này có `status: complete`, chứa checkpoint NCDM tốt nhất, mappings, Q-matrix, predictions, metrics, mastery, gaps, learning paths và plots. Theo quy ước dự án, frontend, demo và báo cáo kết quả phải được cấu hình trỏ vào run này.

`artifacts/eda_20261017T155226873628Z_da4915/` là run EDA độc lập ở chế độ sample 25 sinh viên; không phải run huấn luyện và không thay thế thống kê full trong `data_quality.json` của artifact chính thức. Các lần huấn luyện/thử nghiệm trước nằm trong `artifacts/OLD/`, chỉ dùng để truy vết lịch sử và không được chọn làm dữ liệu mặc định của dự án.

Run EDA và run mô hình có `run_id` khác nhau. Xem [hợp đồng đầu ra](docs/OUTPUT_CONTRACT.md) để biết schema từng file, ý nghĩa của `mastery`/`probability_correct`, cách xử lý ô trống và vị trí hình. Chỉ dùng run có `manifest.json` báo `status: complete`.

## 16. Hướng dẫn tích hợp frontend

Frontend load `artifacts/20261020T160618189951Z_ncdm_3d00a8/manifest.json`, sau đó lọc bảng lộ trình theo `user_id` và sắp xếp `rank` dạng số. Với fine, nối `concept_id` với `concepts.csv`; với middle/coarse, dùng `concept_names` trong `models/id_mappings.json`. Có thể hiển thị `concept_name_zh` trước khi bản dịch tiếng Anh được kiểm duyệt. Chỉ BKT hỗ trợ cập nhật tức thời qua `demo`/`infer`; NCDM dùng checkpoint định kỳ. Không dùng bảng mastery để tính lại metrics test và không tự động quét `artifacts/OLD/`.

## 17. Hạn chế và hướng phát triển

Không có liên kết video hoặc prerequisite được xác minh trong bản cục bộ; lộ trình hiện chỉ gồm bài tập. Hệ thống chưa có goal-based planner, lịch theo deadline hoặc tối ưu thời gian học. Nhãn Bloom số chưa có ánh xạ tên chính thức. Dịch tiếng Anh cần glossary được điền/rà soát. BKT dùng tham số chung và trung bình xác suất giữa concept. NCDM không mô hình hóa biến đổi năng lực theo thời gian, không hỗ trợ sinh viên/problem hoàn toàn mới và các dimension không có quan sát vẫn chỉ là model estimate. Full run đã hoàn tất nhưng vẫn cần đánh giá calibration, cold-start, mức độ phù hợp bởi chuyên gia và hiệu quả với người học trước khi chọn model mặc định. Kiểm thử:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src
```

## 18. Tài liệu tham khảo

- [Repository MOOC-Radar](https://github.com/THU-KEG/MOOC-Radar)
- [Yu và cộng sự, bài báo MoocRadar](https://arxiv.org/abs/2304.02205)
- [Implementation NeuralCDM chính thức](https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD)
- [Wang và cộng sự, Neural Cognitive Diagnosis](https://arxiv.org/abs/1908.08733)

- [Các thuật toán và chính sách sử dụng mô hình](docs/ALGORITHMS.md): BKT Frozen, BKT Online và NCDM.
- [Giải thích NCDM](docs/NCDM_EXPLAINED.md): trực giác, đầu vào, Q-matrix, kiến trúc, mastery và giới hạn.
