# Kiến trúc hệ thống

## Hai luồng xử lý độc lập

```text
data/raw/*.json
  ├─> src/moocradar/eda.py + eda_plots.py
  │     └─> artifacts/eda_<run_id>/
  └─> streaming validation và chuẩn hóa
        └─> data/processed/{smoke|full}/
              └─> chia thời gian theo sinh viên (70/15/15)
                    ├─> baseline + Bayesian Knowledge Tracing (BKT)
                    └─> NCDM + sparse Q-matrix + stable ID mappings
                          └─> dự đoán, metrics, mastery, gaps, learning paths
                                └─> artifacts/<run_id>/
```

Notebook [Data Quality & EDA](../notebooks/01_data_quality_eda.ipynb) dùng trực tiếp bộ đọc tuần tự trong `src/moocradar/reading.py`. Nó kiểm tra thứ tự gốc và chất lượng dữ liệu **trước** khi chuẩn hóa, tạo run EDA riêng và không cần checkpoint mô hình. Notebook [huấn luyện trên Colab](../notebooks/moocradar_colab.ipynb) điều phối pipeline mô hình. Cả hai đều không sửa dữ liệu gốc.

## Chuẩn hóa và đánh giá

Mỗi lượt trả lời được giữ thành một dòng trong `interactions.csv`. Quan hệ nhiều-nhiều giữa bài và concept nằm ở `problem_concepts.csv`. Một lượt có thể cập nhật trạng thái của nhiều concept, nhưng chỉ có một dự đoán ở cấp câu hỏi cho mỗi mô hình. Xác suất trả lời đúng của BKT được lấy bằng trung bình xác suất trên các concept liên kết. Đây là giả định của cách hiện thực, không phải kết luận của bài báo.

Baseline dùng tỷ lệ đúng theo concept từ **train**, làm mượt bằng 10 quan sát giả ở tỷ lệ đúng toàn cục của train. BKT ước lượng bốn tham số chung (`initial`, `learn`, `guess`, `slip`) bằng likelihood trên tối đa 800 chuỗi sinh viên–concept lấy từ train. Nếu dưới 10 chuỗi đủ điều kiện, chương trình dùng giá trị mặc định đã ghi trong mã. Trạng thái mastery được lưu riêng cho từng sinh viên–concept.

Với sinh viên có từ năm lượt trở lên, dữ liệu được sắp xếp ổn định theo `submit_time`, rồi chia khoảng 70% train, 15% validation và phần còn lại test. Sinh viên có ít hơn năm lượt chỉ thuộc train.

## Vai trò của BKT Frozen, BKT Online và NCDM

`bkt_frozen` là protocol đối chứng. BKT học tham số và trạng thái từ train, sau đó giữ nguyên trạng thái khi dự đoán validation/test. Nhãn ở hai split này không được dùng để cập nhật trạng thái. Nhờ đó, BKT Frozen có điều kiện thông tin gần với checkpoint NCDM tĩnh trong protocol so sánh chính.

`bkt_online` là policy vận hành tuần tự. Với từng event, hệ thống dự đoán **trước**, rồi mới nhận nhãn thật để cập nhật trạng thái cho event tiếp theo. Vì vậy mô hình chỉ dùng lịch sử đã quan sát và không rò rỉ nhãn hiện tại; tuy nhiên các dự đoán về sau có thêm thông tin từ những event validation/test trước đó. Learner state mới có thể được dùng để cập nhật knowledge gaps và đề xuất nội dung tiếp theo.

NCDM là nhánh chẩn đoán năng lực theo concept. Mô hình tạo proficiency nhiều chiều từ checkpoint và tái sử dụng pipeline gap/recommendation. Standard NCDM trong dự án không cập nhật tức thời sau mỗi response; dữ liệu mới được hấp thụ khi retrain hoặc fine-tune định kỳ.

```text
BKT Frozen: train state -> freeze -> validation/test predictions
BKT Online: state_t -> predict_t -> observe y_t -> update state_(t+1) -> next recommendation
NCDM:       train -> checkpoint -> concept proficiency -> gaps -> learning path
```

BKT Frozen và BKT Online không phải hai model family khác nhau; chúng là hai chính sách dùng lịch sử của cùng BKT. Chi tiết đầu vào/đầu ra và cách diễn giải nằm tại [ALGORITHMS.md](ALGORITHMS.md).

## Nhánh NCDM

Adapter NCDM tái sử dụng `interactions.csv`, `problems.csv` và `problem_concepts.csv`. Mapping student/problem/concept được tạo một lần trên toàn bộ tập ID, sau đó giữ nguyên giữa mọi split và inference. Việc xây mapping không đọc label làm feature. Q-matrix được lưu dạng COO nén trong `models/q_matrix.npz`; DataLoader chỉ tạo dense Q-row cho mini-batch hiện tại.

Granularity là một phần của config. `middle` ánh xạ mỗi problem sang `exercise_id` và là mặc định vì tập full có 580 middle concept đang hoạt động, Q-row đầy đủ và footprint phù hợp Colab. `fine` dùng các concept metadata và bảo toàn nhiều-nhiều; `coarse` dùng `course_id`. Một run dùng cùng granularity cho baseline, BKT và NCDM.

NCDM học `sigmoid(student_embedding)` như proficiency theo concept, `sigmoid(problem_difficulty)` theo concept và discrimination dương cho từng problem. Tương tác đầu vào là `discrimination × (proficiency - difficulty) × Q`. Các lớp phi tuyến phía sau dùng trọng số không âm và sigmoid, nên tăng proficiency trên concept liên quan không làm giảm xác suất đúng. Hidden size/dropout có thể cấu hình; thay đổi này không thay cấu trúc nhận thức của NeuralCDM chuẩn.

Protocol so sánh chính đóng băng tất cả trạng thái sau train: baseline train-only, `bkt_frozen` và checkpoint NCDM. `bkt_online` được xuất riêng để giữ chính sách BKT prequential của pipeline trước đây; validation/test label chỉ được dùng sau khi prediction tương ứng đã ghi. Metrics và prediction luôn giữ cùng event IDs, nhưng không diễn giải `bkt_online` như policy tương đương NCDM.

Mastery NCDM lấy trực tiếp từ student proficiency vector. `observation_count` chỉ đếm train evidence. Concept chưa quan sát nhưng thuộc khóa học đích có thể có model estimate, song được gắn `unobserved_model_estimate` và không được coi là đã xác nhận. Gap/ranking tái sử dụng module hiện có, không thêm prerequisite/video suy đoán. Standard NCDM không cập nhật tức thời; checkpoint cần retrain định kỳ, trong khi BKT inference vẫn độc lập và có thể update từng event.

Với BKT Online, learner state có thể phản ánh toàn bộ lịch sử đã quan sát, kể cả các lượt test xảy ra trước thời điểm xuất profile. Với NCDM, `student_mastery.csv` lấy từ checkpoint chỉ được huấn luyện bằng train. Không dùng các bảng mastery/lộ trình làm đầu vào cho phép đánh giá dự đoán test. Seed, split indices, cấu hình, tham số và dự đoán đều được lưu theo `run_id`.

Artifact mô hình được dự án sử dụng là `artifacts/20261020T160618189951Z_ncdm_3d00a8/`. Thư mục `artifacts/OLD/` chứa các run cũ để truy vết, không tham gia inference hoặc frontend mặc định. EDA có namespace `artifacts/eda_<run_id>/` riêng và không bị coi là một checkpoint mô hình.

## Hồ sơ và gợi ý

Mastery là xác suất trạng thái kiến thức tiềm ẩn đã học; `probability_correct` là xác suất trả lời đúng có xét guess/slip. Concept chưa từng gặp có mastery chưa biết, không bị gán bằng 0. Khoảng trống kiến thức gồm concept đã quan sát nhưng dưới ngưỡng mục tiêu và concept chưa quan sát trong khóa học đích; nhóm thứ hai có ưu tiên thấp hơn. Hệ thống xuất tối đa 20 khoảng trống cho mỗi sinh viên.

Hoạt động được đề xuất là bài tập có liên kết trực tiếp với concept trong khóa học đích. Điểm xếp hạng ưu tiên bằng chứng chưa thành thạo, số concept liên quan và bài chưa làm. Nếu không chỉ định `target_course_id`, hệ thống dùng khóa học ở lượt quan sát mới nhất. Ngưỡng và trọng số đều là heuristic; không suy luận prerequisite hay liên kết video từ thứ tự chương hoặc `context_id`.

## Dịch metadata

Lệnh `translation extract` trích chuỗi tiếng Trung duy nhất từ metadata sang glossary ngoại tuyến; `translation apply` tạo bảng tiếng Anh đi kèm và giữ nguyên các ID. Chỉ khi có bản dịch đã điền, bảng tiếng Anh mới có nội dung. Không gửi thông tin sinh viên tới dịch vụ bên ngoài.
