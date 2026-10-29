# Kết quả đã đo trên bộ dữ liệu cục bộ

## Artifact chính thức của dự án

Kết quả mô hình được dự án sử dụng nằm tại:

```text
artifacts/20261020T160618189951Z_ncdm_3d00a8/
```

`manifest.json` của run này có `status: complete`, seed 42, granularity `middle` và bốn nhóm dự đoán: `smoothed_concept`, `bkt_frozen`, `bkt_online`, `ncdm`. Đây là nguồn chính thức cho báo cáo metrics, checkpoint inference, learner profile, knowledge gaps, learning paths và frontend.

Các run huấn luyện/thử nghiệm trước đã được chuyển vào `artifacts/OLD/`. Chúng chỉ phục vụ truy vết lịch sử, không được dùng để thay số liệu của run chính thức hoặc được frontend tự động chọn. Run EDA được quản lý riêng và không tính là checkpoint mô hình.

## Dữ liệu và cách chia tập

Run chính thức dùng toàn bộ 898.933 interaction của 14.224 sinh viên trên 2.513 problem đã được tương tác. Dữ liệu được sắp xếp theo thời gian trong từng sinh viên và chia thành:

| Split | Số interaction |
| --- | ---: |
| Train | 624.837 |
| Validation | 128.789 |
| Test | 145.307 |

Metadata có 9.383 problem, 73 khóa học và 9.985 fine concept. NCDM dùng 580 middle concept đang hoạt động; không có interaction bị loại vì thiếu Q-row. Có 203 problem không xuất hiện trong train và được đánh dấu là cold-start khi cần.

`data_quality.json` trong run chính thức ghi nhận 543.847 giá trị `score` bị thiếu và 114 giá trị score bất thường, nhưng `is_correct` đầy đủ cho 898.933 interaction. Không phát hiện `log_id` trùng, foreign key problem sai, nhãn `is_correct` bất thường hoặc timestamp không hợp lệ trong dữ liệu đã chuẩn hóa.

## Cấu hình NCDM chính thức

Run yêu cầu device `auto` và manifest ghi nhận thiết bị thực tế là CPU. Cấu hình huấn luyện gồm batch size 512, learning rate 0,002, hidden dimensions 256/128, dropout 0,5, tối đa 20 epoch và patience 4. Early stopping kết thúc sau 7 epoch; checkpoint tốt nhất theo validation ROC-AUC là epoch 3.

Q-matrix, ID mappings và checkpoint tương ứng được lưu tại `models/q_matrix.npz`, `models/id_mappings.json` và `models/best_ncdm.pt` trong cùng run. Không ghép checkpoint hoặc mappings từ run trong `artifacts/OLD/`.

## Cách đọc tên mô hình

- `bkt_frozen` là thử nghiệm đối chứng: trạng thái BKT sau train được giữ nguyên trên validation/test.
- `bkt_online` là chính sách vận hành tuần tự: mỗi event được dự đoán trước khi nhãn của event đó cập nhật trạng thái cho lần tiếp theo.
- `ncdm` là mô hình chẩn đoán năng lực theo concept dùng checkpoint tĩnh; dữ liệu mới chưa làm thay đổi proficiency cho đến khi retrain/fine-tune.

BKT Frozen và BKT Online dùng cùng cơ chế BKT nhưng khác chính sách sử dụng lịch sử. Vì `bkt_online` được hưởng lợi từ các response validation/test đã quan sát trước đó, kết quả này được báo cáo riêng và không được xem là điều kiện thông tin tương đương NCDM. Xem [ALGORITHMS.md](ALGORITHMS.md) để biết cơ chế và mục đích của từng phương pháp.

## Kết quả dự đoán chính thức

| Model | Validation AUC | Test AUC | Test Brier | Test Balanced Accuracy | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smoothed concept | 0,698879 | 0,708021 | **0,143918** | 0,530712 | **0,891036** |
| BKT frozen | 0,644992 | 0,514144 | 0,158849 | 0,505103 | 0,890336 |
| NCDM | **0,790925** | **0,723662** | 0,164394 | **0,636039** | 0,835025 |
| BKT online (protocol bổ sung) | 0,768563 | 0,781523 | 0,132744 | 0,574480 | 0,890826 |

Trong protocol chính đóng băng train-state, NCDM có validation/test AUC và test balanced accuracy cao hơn baseline middle và BKT Frozen. Test specificity của NCDM là 0,460653 và positive prediction rate là 0,757913, trong khi prevalence lớp đúng là 0,803320. Điều này cho thấy NCDM ít nghiêng về dự đoán toàn lớp dương hơn baseline và BKT Frozen.

Tuy nhiên, baseline có test Brier, accuracy và F1 tốt hơn NCDM; BKT Frozen cũng có Brier thấp hơn NCDM trong run này. Vì vậy kết quả chỉ hỗ trợ nhận định NCDM hữu ích hơn ở ranking, nhận diện hai lớp cân bằng và learner profile concept-level. Không có cơ sở để tuyên bố NCDM tốt hơn tuyệt đối về calibration hoặc quyết định tại threshold 0,5.

`bkt_online` đạt test AUC 0,781523 và Brier 0,132744, nhưng đã nhận các label validation/test trước đó theo thứ tự thời gian. Đây là policy triển khai adaptive khác checkpoint NCDM tĩnh và chỉ được báo cáo bổ sung.

## Learner profile và recommendation

Artifact chính thức chứa:

| Đầu ra | Số dòng dữ liệu |
| --- | ---: |
| `student_mastery.csv` | 528.706 |
| `knowledge_gaps.csv` | 240.717 |
| `learning_paths.csv` | 71.094 |
| Sinh viên có learning path | 14.224 |

Mọi split row và prediction event ID đã được kiểm tra căn chỉnh trong pipeline. Checkpoint, Q-matrix và mappings cùng nằm trong một run để inference có thể lặp lại mà không huấn luyện lại. Các con số trên mô tả artifact hiện có; chúng không chứng minh lộ trình làm tăng kết quả học tập thực tế.

## Artifact EDA

EDA đang được giữ riêng tại:

```text
artifacts/eda_20261017T155226873628Z_da4915/
```

Manifest của EDA này ghi `mode: sample`, gồm 25 sinh viên và 752 interaction. Vì vậy các tỷ lệ trong báo cáo EDA chỉ mô tả mẫu, không đại diện cho toàn bộ dữ liệu. Thống kê toàn bộ dùng trong modeling nằm trong `data_quality.json` của artifact mô hình chính thức.

## Giới hạn diễn giải

Kết quả không phải benchmark của bài báo MOOC-Radar hoặc NeuralCDM. Threshold 0,5, mastery target, confidence và điểm ưu tiên recommendation vẫn là heuristic. Hệ thống chưa có đánh giá can thiệp với người học/giảng viên, prerequisite được xác minh hoặc tài nguyên video có liên kết. Cần đánh giá thêm calibration, cold-start và hiệu quả học tập trước khi thay đổi model mặc định.
