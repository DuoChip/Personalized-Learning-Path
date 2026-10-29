# Giải thích thuật toán Neural Cognitive Diagnosis Model (NCDM)

## 1. NCDM là thuật toán gì?

NCDM, viết tắt của **Neural Cognitive Diagnosis Model**, là mô hình chẩn đoán nhận thức sử dụng mạng neural. Mô hình có hai nhiệm vụ chính:

1. Dự đoán xác suất một sinh viên trả lời đúng một bài tập.
2. Ước lượng mức độ thành thạo của sinh viên đối với từng concept kiến thức.

Khác với mô hình phân loại thông thường, NCDM không chỉ học quan hệ giữa `student_id` và `problem_id`. Mô hình còn sử dụng Q-matrix để biết mỗi bài tập yêu cầu những concept nào. Vì vậy, biểu diễn năng lực của sinh viên có ý nghĩa theo từng concept thay vì chỉ là một vector ẩn khó giải thích.

Ví dụ, nếu hệ thống có ba concept:

- Đại số tuyến tính.
- Xác suất.
- Lập trình Python.

NCDM có thể biểu diễn năng lực của một sinh viên dưới dạng:

```text
[0.82, 0.35, 0.64]
```

Điều này được diễn giải là mô hình ước lượng sinh viên có proficiency 0,82 ở Đại số tuyến tính, 0,35 ở Xác suất và 0,64 ở Python. Đây là **ước lượng của mô hình**, không phải chứng nhận rằng sinh viên chắc chắn đã thành thạo các concept đó.

## 2. Cognitive Diagnosis là gì?

Cognitive Diagnosis, hay chẩn đoán nhận thức, là bài toán suy ra trạng thái kiến thức tiềm ẩn của người học từ các response quan sát được.

Một response thường gồm:

```text
sinh viên + bài tập + các concept cần thiết + đúng/sai
```

Nhãn đúng/sai quan sát được, nhưng mastery theo concept không được quan sát trực tiếp. NCDM học mastery gián tiếp bằng cách tối ưu khả năng dự đoán các response trong tập train.

NCDM thuộc họ mô hình cognitive diagnosis tĩnh. Mỗi checkpoint biểu diễn trạng thái năng lực được học từ dữ liệu train tại thời điểm huấn luyện. NCDM chuẩn không tự cập nhật mastery tức thời sau từng response mới như BKT.

## 3. Đầu vào của NCDM

### 3.1. Dữ liệu interaction

Mỗi mẫu dự đoán tương ứng với một lượt sinh viên làm một bài tập:

| Trường | Ý nghĩa |
| --- | --- |
| `student_id` | Sinh viên thực hiện bài tập |
| `problem_id` | Bài tập được thực hiện |
| `is_correct` | Nhãn đúng/sai, chỉ dùng làm target khi huấn luyện hoặc đánh giá |
| `concept_ids` | Các concept cần thiết để giải bài tập |

Trong dự án MOOC-Radar, các trường thực tế được lấy từ:

- `data/processed/<mode>/interactions.csv`.
- `data/processed/<mode>/problems.csv`.
- `data/processed/<mode>/problem_concepts.csv`.

Một lượt trả lời vẫn là **một mẫu ở cấp bài tập**, kể cả khi bài tập liên kết với nhiều concept.

### 3.2. ID mappings

PyTorch embedding cần chỉ số nguyên liên tục. Vì vậy các ID gốc được ánh xạ thành index:

```text
U_1002476  -> student index 0
Pm_668265  -> problem index 15
Ex_535149  -> concept index 7
```

Một bộ mapping duy nhất được dùng cho train, validation, test và inference. Điều này ngăn cùng một ID bị ánh xạ thành các index khác nhau giữa các split.

Mappings được lưu tại:

```text
models/id_mappings.json
```

### 3.3. Q-matrix

Q-matrix mô tả quan hệ giữa bài tập và concept:

```text
Q[problem, concept] = 1  nếu bài tập cần concept đó
Q[problem, concept] = 0  nếu không cần
```

Ví dụ:

| Problem | Đại số | Xác suất | Python |
| --- | ---: | ---: | ---: |
| P1 | 1 | 1 | 0 |
| P2 | 0 | 1 | 0 |
| P3 | 0 | 1 | 1 |

Problem P1 cần cả Đại số và Xác suất. Đây là cách NCDM bảo toàn quan hệ nhiều-nhiều giữa problem và concept.

Trong dự án, Q-matrix được lưu nén dạng sparse COO tại:

```text
models/q_matrix.npz
```

Khi training, pipeline chỉ tạo tensor Q dày cho mini-batch hiện tại, thay vì giữ một tensor dày rất lớn cho toàn bộ interaction.

## 4. Granularity của concept

Pipeline hỗ trợ ba mức concept:

| Granularity | Concept được dùng | Đặc điểm |
| --- | --- | --- |
| `coarse` | `course_id` | Nhỏ, tổng quát nhưng ít chi tiết |
| `middle` | `exercise_id` | Cân bằng giữa độ chi tiết và chi phí tính toán |
| `fine` | Concept trong `problem_concepts.csv` | Chi tiết và hỗ trợ nhiều concept trên một problem, nhưng thưa hơn |

Full run hiện tại dùng `middle`. Tập full có 580 middle concepts đang hoạt động và không có interaction bị mất do thiếu Q-row. Cùng granularity được dùng cho baseline, BKT và NCDM trong một run so sánh.

## 5. Các thành phần của mô hình

### 5.1. Student proficiency

Mỗi sinh viên có một embedding với số chiều bằng số concept:

```text
student_embedding[student_id]
```

Embedding thô có thể nhận mọi giá trị thực. NCDM đưa nó qua hàm sigmoid:

```text
proficiency = sigmoid(student_embedding)
```

Do đó mỗi giá trị proficiency nằm trong khoảng `(0, 1)` và có thể dùng làm mastery estimate theo concept.

### 5.2. Exercise difficulty

Mỗi problem có một vector difficulty theo concept:

```text
difficulty = sigmoid(problem_difficulty[problem_id])
```

Ví dụ, cùng một problem có thể khó hơn ở concept Xác suất nhưng dễ hơn ở concept Python.

### 5.3. Exercise discrimination

Discrimination đo khả năng phân biệt sinh viên có năng lực cao và thấp của một problem:

```text
discrimination = sigmoid(problem_discrimination[problem_id]) * scale
```

Problem có discrimination lớn sẽ phản ứng mạnh hơn với chênh lệch giữa proficiency và difficulty.

### 5.4. Q-mask

Trước khi đưa vào prediction network, NCDM tính interaction:

```text
x = discrimination * (proficiency - difficulty) * Q
```

Phép nhân với Q là theo từng phần tử. Những concept không liên quan đến problem có `Q = 0`, nên không đóng góp trực tiếp vào prediction của problem đó.

Ví dụ:

```text
proficiency = [0.8, 0.4, 0.7]
difficulty  = [0.5, 0.6, 0.3]
Q           = [1,   1,   0]
```

Khi đó:

```text
(proficiency - difficulty) * Q
= [0.3, -0.2, 0.0]
```

Concept thứ ba không liên quan nên bị loại khỏi interaction của problem này.

### 5.5. Prediction network

Vector interaction được đưa qua các lớp neural phi tuyến:

```text
x
 -> positive linear layer
 -> sigmoid
 -> dropout
 -> positive linear layer
 -> sigmoid
 -> probability_correct
```

Các lớp này giúp mô hình học tương tác phi tuyến giữa năng lực sinh viên và đặc điểm bài tập.

## 6. Tính đơn điệu và khả năng diễn giải

Một yêu cầu quan trọng của NCDM là **monotonicity**:

> Nếu mọi yếu tố khác giữ nguyên, proficiency của sinh viên tăng trên một concept cần thiết thì xác suất trả lời đúng không được giảm.

Pipeline thực hiện ràng buộc này bằng cách giữ trọng số của các prediction layers không âm. Sau mỗi optimizer step, trọng số âm được chiếu về 0. Forward pass cũng clamp trọng số để không âm.

Nếu sử dụng một MLP thông thường với trọng số âm tùy ý, proficiency tăng có thể làm xác suất đúng giảm. Khi đó embedding sinh viên sẽ khó được diễn giải như mastery. Vì vậy ràng buộc đơn điệu là thành phần cốt lõi phân biệt NCDM với một mạng neural phân loại tùy ý.

## 7. Cơ chế training

Training diễn ra theo các bước sau:

1. Chia interaction theo thời gian trong từng sinh viên thành train, validation và test.
2. Xây ID mappings và Q-matrix dùng chung cho mọi split.
3. Chỉ dùng train interactions để cập nhật tham số NCDM.
4. Trong mỗi mini-batch, lấy student index, problem index và Q-row tương ứng.
5. Tính `probability_correct` bằng forward pass.
6. So sánh probability với nhãn `is_correct` bằng binary cross-entropy loss.
7. Backpropagation và cập nhật bằng Adam.
8. Chiếu trọng số prediction layers về miền không âm.
9. Sau mỗi epoch, đo ROC-AUC và Brier trên validation.
10. Lưu checkpoint có validation ROC-AUC tốt nhất.
11. Chỉ dùng test sau khi quá trình chọn checkpoint kết thúc.

Nhãn validation và test không được dùng để cập nhật checkpoint NCDM.

Checkpoint tốt nhất được lưu tại:

```text
models/best_ncdm.pt
```

Checkpoint chứa:

- Model state.
- Kiến trúc và kích thước embedding.
- Epoch tốt nhất.
- Validation metric.
- Training config.
- Optimizer state để có thể resume.

## 8. Đầu ra dự đoán

Với một bộ ba:

```text
student_id + problem_id + required_concepts
```

NCDM trả về:

```text
P(correct | student, problem, required_concepts)
```

Ví dụ:

```text
0.76
```

Giá trị này nghĩa là checkpoint dự đoán xác suất trả lời đúng là 76%. Nó không có nghĩa sinh viên có mastery 76% ở tất cả concept của problem.

Predictions validation/test được lưu trong `predictions.csv` với các trường chính:

```text
log_id
user_id
problem_id
split
model
is_correct
probability_correct
evaluation_protocol
cold_start_problem
```

## 9. Đầu ra cognitive diagnosis

Mastery được lấy trực tiếp từ student proficiency:

```text
mastery_probability = sigmoid(student_embedding)[concept]
```

Ví dụ:

| user_id | concept_id | mastery_probability | observations | evidence_status |
| --- | --- | ---: | ---: | --- |
| U1 | C1 | 0.82 | 8 | `observed_model_estimate` |
| U1 | C2 | 0.35 | 0 | `unobserved_model_estimate` |

Cả hai giá trị đều là model estimates. Tuy nhiên, C1 có tám train observations hỗ trợ, trong khi C2 chưa có bằng chứng trực tiếp. Frontend không nên hiển thị hai giá trị này với mức độ chắc chắn giống nhau.

Các trường chính trong `student_mastery.csv` gồm:

| Trường | Ý nghĩa |
| --- | --- |
| `user_id` | ID sinh viên gốc |
| `concept_id` | ID concept theo granularity của run |
| `mastery_probability` | Proficiency NCDM trong khoảng 0–1 |
| `mastery_level` | Nhãn heuristic: needs support/developing/mastered |
| `observation_count` | Số train interactions liên quan đến concept |
| `insufficient_evidence` | Bằng 1 nếu dưới ba observations |
| `evidence_status` | Có hoặc chưa có bằng chứng interaction trực tiếp |
| `model_name` | `ncdm` |
| `run_id` | Run đã tạo estimate |

`mastery_level`, ngưỡng 0,7, confidence và điều kiện dưới ba observations đều là heuristic. Chúng chưa phải ngưỡng đã được kiểm chứng sư phạm.

## 10. Từ mastery đến learning path

Luồng tích hợp trong dự án là:

```text
NCDM checkpoint
    -> student proficiency theo concept
    -> learner profile
    -> knowledge gaps
    -> xếp hạng bài luyện tập
    -> Top-N learning path
```

NCDM chỉ thay nguồn mastery. Module knowledge-gap detection và recommendation hiện có được tái sử dụng.

Recommendation chỉ chọn problem có liên kết concept trực tiếp trong dữ liệu và thuộc khóa học đích. Pipeline không tự tạo prerequisite, video hoặc quan hệ concept–activity không có trong dữ liệu.

## 11. NCDM khác BKT như thế nào?

| Đặc điểm | BKT | NCDM |
| --- | --- | --- |
| Loại mô hình | Xác suất tuần tự | Cognitive diagnosis bằng neural network |
| Trạng thái kiến thức | Cập nhật sau từng response | Học trong quá trình training |
| Biểu diễn problem | Chủ yếu qua concept và tham số BKT chung | Difficulty theo concept và discrimination riêng cho problem |
| Tương tác phi tuyến | Hạn chế | Có prediction network phi tuyến |
| Cập nhật thời gian thực | Có trong pipeline hiện tại | Chưa có trong standard NCDM |
| Sinh viên mới | Có thể khởi tạo từ prior | Không có embedding đã học |
| Problem mới | Có thể dựa vào concept state | Difficulty/discrimination chưa được học |

Không nên kết luận NCDM tốt hơn BKT chỉ vì NCDM phức tạp hơn. Hai mô hình có chính sách sử dụng lịch sử khác nhau và cần được báo cáo bằng protocol rõ ràng.

Trong run NCDM hiện tại:

- `bkt_frozen` là thử nghiệm đối chứng: chỉ dùng trạng thái sau train và không cập nhật bằng validation/test, nên phù hợp hơn cho so sánh chính với NCDM.
- `bkt_online` là policy vận hành tuần tự: dự đoán trước rồi cập nhật bằng từng response đã quan sát để điều chỉnh trạng thái và đề xuất tiếp theo.
- NCDM là phương pháp bổ sung để chẩn đoán proficiency theo concept và dùng checkpoint đóng băng cho validation/test.

BKT Frozen và BKT Online dùng cùng cơ chế BKT; chúng khác nhau ở chính sách cập nhật trạng thái, không phải là hai thuật toán được train độc lập. NCDM hiện không thay thế khả năng cập nhật theo từng response của BKT Online. Xem [ALGORITHMS.md](ALGORITHMS.md) để đọc mô tả thống nhất về vai trò của ba phương pháp.

## 12. Cách đánh giá NCDM

Pipeline báo cáo:

- ROC-AUC: khả năng xếp response đúng cao hơn response sai.
- PR-AUC: hữu ích khi tỷ lệ lớp mất cân bằng.
- Accuracy: tỷ lệ phân loại đúng ở threshold đã chọn.
- Balanced Accuracy: trung bình recall của hai lớp.
- F1-score: cân bằng giữa precision và recall của lớp đúng.
- Precision và Recall.
- Specificity: tỷ lệ nhận diện đúng response sai.
- Brier Score: sai số bình phương của xác suất; càng thấp càng tốt.
- Confusion Matrix.
- Class prevalence.
- Positive prediction rate.

Threshold hiện tại là 0,5 và được ghi trong config/manifest. Nếu thay threshold, quy tắc lựa chọn chỉ được fit trên validation, không được chọn trên test.

Artifact chính thức `20261020T160618189951Z_ncdm_3d00a8` cho thấy NCDM có test ROC-AUC 0,723662 và balanced accuracy 0,636039, cao hơn baseline middle ở hai chỉ số này. Tuy nhiên, test Brier 0,164394, Accuracy 0,742435 và F1 0,835025 thấp hơn baseline. Điều này cho thấy NCDM phân biệt/ranking tốt hơn trong run này nhưng chưa calibration hoặc phân loại tại threshold 0,5 tốt hơn tuyệt đối.

Xem số liệu chi tiết tại [RESULTS.md](RESULTS.md).

## 13. Các hạn chế cần hiểu đúng

### NCDM không phải mô hình knowledge tracing thời gian thực

Student proficiency chỉ thay đổi khi checkpoint được retrain hoặc fine-tune. Hàm inference không nhận một nhãn mới rồi cập nhật mastery ngay lập tức.

### Mastery vẫn là ước lượng

Giá trị proficiency nằm trong khoảng 0–1 nhưng không tự động trở thành xác suất sư phạm đã được calibration. Cần kiểm định với chuyên gia, bài kiểm tra chuẩn hoặc dữ liệu can thiệp.

### Cold-start

Sinh viên mới không có student embedding đã học. Problem mới không có difficulty và discrimination đã học. Pipeline đánh dấu problem không xuất hiện trong train bằng `cold_start_problem=1`.

### Concept chưa quan sát

NCDM có thể trả ra một giá trị embedding cho concept chưa được sinh viên làm trực tiếp. Giá trị này phải được hiển thị là `unobserved_model_estimate`, không phải mastery đã xác nhận.

### Granularity ảnh hưởng kết quả

Kết quả middle và fine không thể được xem là cùng một bài toán hoàn toàn giống nhau. Khi so sánh model, phải giữ cùng granularity hoặc giải thích rõ khác biệt.

## 14. Ví dụ chạy và inference

Training:

```bash
PYTHONPATH=src python -m moocradar train \
  --model ncdm \
  --processed-dir data/processed/full \
  --artifact-root artifacts \
  --granularity middle \
  --batch-size 512 \
  --learning-rate 0.002 \
  --epochs 20 \
  --patience 4 \
  --hidden-dims 256,128 \
  --dropout 0.5 \
  --device auto \
  --seed 42
```

Load checkpoint và tạo learner profile:

```bash
PYTHONPATH=src python -m moocradar demo \
  --model ncdm \
  --run-dir artifacts/<run_id> \
  --processed-dir data/processed/full \
  --user-id <user_id>
```

Nếu truyền `--new-problem-id` và `--new-is-correct` cho NCDM, CLI sẽ từ chối vì hệ thống chưa triển khai update NCDM thời gian thực. Dùng BKT cho luồng cập nhật từng response hoặc retrain NCDM định kỳ.

## 15. Tài liệu tham khảo

- [Neural Cognitive Diagnosis for Intelligent Education Systems](https://arxiv.org/abs/1908.08733)
- [Implementation NeuralCDM chính thức](https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD)
- [MOOC-Radar](https://github.com/THU-KEG/MOOC-Radar)
