# Các thuật toán và chính sách sử dụng mô hình

Tài liệu này giải thích vai trò của **BKT Frozen**, **BKT Online** và **NCDM** trong hệ thống. `BKT Frozen` và `BKT Online` không phải hai thuật toán BKT được huấn luyện độc lập. Chúng dùng cùng cơ chế Bayesian Knowledge Tracing nhưng khác chính sách cập nhật trạng thái trong giai đoạn validation/test hoặc khi vận hành.

## Tổng quan

| Tên trong artifacts | Vai trò | Có cập nhật từ response mới? | Mục đích chính |
| --- | --- | --- | --- |
| `bkt_frozen` | Thử nghiệm đối chứng | Không cập nhật trong validation/test | So sánh dự đoán từ cùng lượng thông tin train |
| `bkt_online` | Chính sách vận hành tuần tự | Có, sau khi đã dự đoán response hiện tại | Theo dõi kiến thức và điều chỉnh gợi ý theo lịch sử đã quan sát |
| `ncdm` | Chẩn đoán nhận thức theo concept | Không cập nhật tức thời; cần checkpoint mới | Học learner profile nhiều chiều và đặc điểm riêng của bài tập |

## Luồng đầu vào và đầu ra trong dự án

Có thể hình dung pipeline thành hai nhánh đầu ra khác nhau:

```text
interactions + problem/concept mapping
                |
                +--> dự đoán từng lượt làm bài
                |      -> predictions.csv
                |      -> metrics.csv + plots
                |
                +--> NCDM student proficiency theo concept
                       -> student_mastery.csv
                       -> knowledge_gaps.csv
                       -> learning_paths.csv
```

Nhánh dự đoán dùng để **đánh giá model**. Nhánh mastery dùng để **xây learner profile và recommendation**. Không lấy trực tiếp `probability_correct` trong `predictions.csv` làm mastery.

### Các file đầu vào

| File | Trường được dùng | Vai trò |
| --- | --- | --- |
| `data/processed/full/interactions.csv` | `log_id`, `user_id`, `problem_id`, `is_correct`, `submit_time`, `sequence_position` | Lịch sử làm bài và thứ tự thời gian |
| `data/processed/full/problems.csv` | `problem_id`, `exercise_id`, `course_id` | Xác định bài thuộc concept/course nào ở granularity đã chọn |
| `data/processed/full/problem_concepts.csv` | `problem_id`, `concept_id` | Quan hệ nhiều-nhiều ở granularity `fine` |
| `split_indices.csv` | `log_id`, `split` | Giữ đúng event train/validation/test |

Run chính thức dùng granularity `middle`, vì vậy `exercise_id` trong `problems.csv` đóng vai trò `concept_id`. `problem_concepts.csv` vẫn là nguồn mapping khi chọn `fine`, nhưng không được trộn với mapping `middle` trong cùng run.

### Đầu vào và đầu ra theo từng phương pháp

| Phương pháp | Đầu vào khi dự đoán | Có đọc nhãn của lượt hiện tại trước dự đoán? | Đầu ra trực tiếp |
| --- | --- | --- | --- |
| Smoothed concept | Concept của bài và tỷ lệ đúng đã học từ train | Không | `probability_correct` |
| BKT Frozen | Sinh viên, concept của bài và BKT state sau train | Không | `probability_correct`; state không đổi |
| BKT Online | Sinh viên, concept của bài và state từ các response đã quan sát trước đó | Không | `probability_correct`, sau đó state được cập nhật khi có nhãn thật |
| NCDM | Student index, problem index và Q-row của bài | Không | `probability_correct`; đồng thời checkpoint chứa proficiency vector của sinh viên |

Trong quá trình **training**, `is_correct` của tập train là target để học tham số. Trong validation/test, pipeline phải tạo `probability_correct` trước; `is_correct` sau đó chỉ dùng để tính metrics. Riêng BKT Online còn dùng nhãn vừa quan sát để cập nhật state cho event tiếp theo.

### Ví dụ một dự đoán thực tế

Trong artifact chính thức, event `1002476_5940136` của sinh viên `U_1002476` trên bài `Pm_668288` có nhãn thật `is_correct=1`. Trước khi dùng nhãn này để đánh giá, các model đã tạo:

| Model | `probability_correct` | Cách hiểu |
| --- | ---: | --- |
| Smoothed concept | 0,70985000 | Xác suất từ thống kê concept trên train |
| BKT Frozen | 0,77128203 | Xác suất từ state BKT đóng băng sau train |
| BKT Online | 0,77128203 | Xác suất từ lịch sử quan sát đến trước event này |
| NCDM | 0,82692498 | Xác suất từ student embedding, problem parameters và Q-row |

Các giá trị trên trả lời câu hỏi “sinh viên có khả năng làm đúng **bài này** bao nhiêu?”. Chúng không trả lời trực tiếp “sinh viên đã thành thạo **concept** bao nhiêu?”.

### Từ NCDM mastery đến learning path

Artifact chính thức tạo luồng sau:

```text
NCDM checkpoint
  -> mastery_probability theo student–concept
  -> so với mastery_target = 0,7
  -> knowledge gap
  -> tìm problem cùng course có liên kết trực tiếp với gap
  -> xếp hạng Top-N learning path
```

Ví dụ với `U_1002476`:

1. `student_mastery.csv` ghi concept `Ex_535149` có `mastery_probability=0,576124`, mức `developing`, với 3 quan sát train.
2. Với concept `Ex_535154`, mastery là `0,494601`, thấp hơn target `0,7`, nên `knowledge_gaps.csv` ghi `observed_gap` và priority `0,205399`.
3. `learning_paths.csv` chọn các bài có mapping trực tiếp tới concept còn yếu. Ví dụ bài `Pm_668281` được xếp hạng 1 để luyện `Ex_535156`.

Đây là ba loại đại lượng khác nhau:

| Đại lượng | Cấp độ | Dùng để làm gì? |
| --- | --- | --- |
| `probability_correct` | Một sinh viên × một bài | Đánh giá dự đoán hoặc hiển thị khả năng làm đúng bài cụ thể |
| `mastery_probability` | Một sinh viên × một concept | Hiển thị learner profile và tìm concept yếu |
| `score` trong `learning_paths.csv` | Một sinh viên × một hoạt động | Xếp hạng bài nên luyện tiếp; không phải xác suất làm đúng |

### Chọn output theo nhu cầu

| Nhu cầu | File/đầu ra nên dùng |
| --- | --- |
| So sánh chất lượng model | `metrics.csv` và `plots/` |
| Kiểm tra model dự đoán từng event | `predictions.csv` |
| Hiển thị hồ sơ năng lực theo concept | `student_mastery.csv` |
| Hiển thị các concept cần hỗ trợ | `knowledge_gaps.csv` |
| Hiển thị Top-N bài nên học tiếp | `learning_paths.csv` |
| Dự đoán lại bằng NCDM | `best_ncdm.pt` + `id_mappings.json` + `q_matrix.npz` |

Trong artifact NCDM chính thức, các dòng `bkt_frozen` và `bkt_online` trong `predictions.csv` phục vụ đánh giá/đối chiếu. Các file profile, gap và path của run này được sinh từ **NCDM**, thể hiện qua `model_name=ncdm` trong `student_mastery.csv`. BKT cập nhật online là một luồng inference riêng và không ghi đè checkpoint hoặc artifact NCDM.

## 1. Nền tảng chung của BKT

Bayesian Knowledge Tracing biểu diễn trạng thái kiến thức của một sinh viên đối với từng concept bằng xác suất tiềm ẩn đã học. Bản hiện thực trong dự án dùng bốn tham số chung:

- `initial`: xác suất sinh viên đã biết concept trước quan sát đầu tiên.
- `learn`: xác suất chuyển từ chưa biết sang đã biết sau một cơ hội luyện tập.
- `guess`: xác suất trả lời đúng dù concept chưa được nắm vững.
- `slip`: xác suất trả lời sai dù concept đã được nắm vững.

Từ mastery hiện tại, BKT tính xác suất trả lời đúng có xét `guess` và `slip`. Sau khi có kết quả thật, mô hình dùng định lý Bayes để cập nhật mastery rồi áp dụng xác suất học. Với bài yêu cầu nhiều concept, pipeline hiện tại lấy trung bình xác suất của các concept liên kết để tạo đúng một dự đoán ở cấp bài tập. Đây là giả định triển khai của dự án.

### Đầu vào và đầu ra của BKT

Đầu vào gồm `student_id`, `problem_id`, các concept của bài, thứ tự thời gian và nhãn đúng/sai của những response **đã được quan sát**. Đầu ra chính gồm:

- `probability_correct`: xác suất trả lời đúng bài hiện tại.
- mastery theo cặp sinh viên–concept.
- knowledge gaps và danh sách bài luyện tập được xếp hạng từ mastery.

Nhãn của response hiện tại chỉ được dùng sau khi dự đoán response đó. Nhãn tương lai không được dùng để dự đoán quá khứ.

## 2. BKT Frozen

`bkt_frozen` là **protocol đối chứng**, không phải một thuật toán BKT mới. Mô hình học tham số và xây trạng thái sinh viên từ tập train. Khi đánh giá validation/test, trạng thái này được giữ cố định:

```text
train history -> BKT state sau train -> đóng băng
                                      -> dự đoán validation/test
```

Nhãn validation/test không được đưa trở lại trạng thái để cải thiện những dự đoán tiếp theo. Protocol này trả lời câu hỏi: nếu các mô hình chỉ được phép học từ train, checkpoint nào dự đoán tốt hơn trên các event về sau?

Ưu điểm của BKT Frozen là tạo điều kiện so sánh gần hơn với checkpoint NCDM tĩnh. Hạn chế là nó không phản ánh đầy đủ lợi thế tự nhiên của BKT trong một hệ thống tuần tự, nơi response mới thường được dùng để cập nhật learner state.

## 3. BKT Online

`bkt_online` là chính sách vận hành theo thời gian. Với mỗi response, hệ thống thực hiện đúng thứ tự:

```text
state_t
  -> dự đoán P(correct_t)
  -> quan sát kết quả thật y_t
  -> cập nhật state_(t+1)
  -> cập nhật learner profile, gap và nội dung tiếp theo
```

Do dự đoán được ghi trước khi đọc nhãn hiện tại, đây không phải rò rỉ nhãn của chính event đang dự đoán. Tuy nhiên, dự đoán ở thời điểm sau được hưởng lợi từ các nhãn validation/test đã xuất hiện trước đó. Vì vậy kết quả `bkt_online` thể hiện một chính sách sử dụng lịch sử khác với BKT Frozen và NCDM tĩnh.

Trong sản phẩm, BKT Online phù hợp với luồng adaptive: khi sinh viên hoàn thành thêm bài, learner state được cập nhật và hệ thống có thể tạo lại knowledge gaps hoặc đề xuất nội dung tiếp theo. Chỉ sử dụng các response đã thực sự quan sát; không dùng thông tin tương lai.

## 4. NCDM

Neural Cognitive Diagnosis Model là phương pháp bổ sung để chẩn đoán năng lực sinh viên theo nhiều concept. Mô hình học đồng thời:

- proficiency của mỗi sinh viên trên từng concept;
- difficulty của mỗi bài trên từng concept;
- discrimination của từng bài;
- tương tác phi tuyến giữa năng lực và đặc điểm bài tập.

Q-matrix cho biết bài tập yêu cầu những concept nào. Với sinh viên `s`, bài `e` và Q-row `Q_e`, pipeline tạo biểu diễn tương tác theo dạng:

```text
discrimination_e * (proficiency_s - difficulty_e) * Q_e
```

Biểu diễn này đi qua prediction network có trọng số không âm. Ràng buộc đơn điệu giúp đảm bảo rằng khi proficiency tăng trên concept liên quan, xác suất làm đúng không bị giảm chỉ do trọng số âm tùy ý.

### Đầu vào

- ID sinh viên và ID bài tập đã được ánh xạ ổn định.
- Nhãn đúng/sai của interaction trong tập train.
- Q-matrix problem–concept, có thể chứa nhiều concept cho một bài.
- Granularity concept được ghi trong config; run hiện tại dùng `middle`.

### Đầu ra

- `probability_correct` cho một cặp sinh viên–bài tập.
- vector student proficiency theo concept.
- `mastery_probability = sigmoid(student_proficiency)` cho từng sinh viên–concept.
- knowledge gaps và learning paths được tạo bằng module recommendation hiện có.

Mastery NCDM phải lấy từ student proficiency, không lấy xác suất đúng của một bài rồi coi là mastery. `observation_count` và `evidence_status` đi kèm giúp phân biệt estimate có dữ liệu quan sát với estimate ở concept chưa đủ bằng chứng.

NCDM chuẩn trong dự án là mô hình tĩnh theo checkpoint. Nó không tự cập nhật learner state ngay sau một response như BKT Online; muốn hấp thụ dữ liệu mới cần retrain hoặc fine-tune và phát hành checkpoint mới.

## 5. Cách so sánh và sử dụng

Protocol so sánh chính dùng `smoothed_concept`, `bkt_frozen` và `ncdm`, vì cả ba chỉ học từ train trước khi dự đoán validation/test. `bkt_online` được báo cáo riêng như kết quả của chính sách adaptive. Không nên gộp các protocol này rồi kết luận một mô hình tốt hơn tuyệt đối.

Trong hệ thống cá nhân hóa, ba vai trò có thể cùng tồn tại:

- BKT Frozen kiểm tra chất lượng dự đoán trong điều kiện train-state được giữ nguyên.
- BKT Online phục vụ cập nhật trạng thái tuần tự và đề xuất nội dung tiếp theo sau mỗi response.
- NCDM bổ sung hồ sơ năng lực nhiều chiều theo concept và có thể được cập nhật theo chu kỳ huấn luyện.

Việc chọn mô hình phải dựa trên mục tiêu sử dụng và số liệu thực nghiệm. Xem [RESULTS.md](RESULTS.md) để đọc kết quả đã đo, [NCDM_EXPLAINED.md](NCDM_EXPLAINED.md) để xem chi tiết kiến trúc NCDM và [OUTPUT_CONTRACT.md](OUTPUT_CONTRACT.md) để tích hợp artifacts.
