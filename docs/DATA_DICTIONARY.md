# Từ điển dữ liệu MOOC-Radar cục bộ

## File dữ liệu gốc

Bốn file được chuyển nguyên trạng vào `data/raw/`. Dung lượng dưới đây là số byte thực tế của bản dữ liệu hiện có.

| File | Định dạng | Dung lượng (byte) | Vai trò |
| --- | --- | ---: | --- |
| `problem.json` | JSON Lines | 6.715.077 | Metadata bài tập, khóa học, concept và nhãn Bloom |
| `student-problem-fine.json` | JSON array gồm các `{"seq": [...]}` | 152.280.456 | Chuỗi trả lời ở cấp bài |
| `student-problem-middle.json` | JSON array cùng cấu trúc | 201.525.773 | Chuỗi có `skill_id` ở cấp exercise |
| `student-problem-coarse.json` | JSON array cùng cấu trúc | 197.158.175 | Chuỗi có `skill_id` ở cấp course |

Luồng chuẩn hóa và EDA chi tiết dùng event từ file `fine`. NCDM không đọc lại label từ file middle/coarse: granularity `middle` được tái tạo bằng `problem_id → exercise_id`, `coarse` bằng `problem_id → course_id`, và `fine` bằng `problem_concepts.csv`. Cách này giữ nguyên event IDs, label và split giữa các model. Notebook EDA chỉ lấy mẫu schema của hai file middle/coarse, không tự nhận đã quét toàn bộ chúng.

## Bảng và khóa liên kết

| Nguồn | Trường chính | Ý nghĩa và cách dùng |
| --- | --- | --- |
| `problem.json` | `problem_id` | Khóa nối với từng lượt trả lời; 9.383 bài metadata duy nhất |
| `problem.json` | `exercise_id`, `course_id` | 1.563 exercise và 73 khóa học |
| `problem.json` | `concepts` | Danh sách tên concept tiếng Trung; một bài có 1–6 concept |
| `problem.json` | `cognitive_dimension` | Mã Bloom dạng số 1–6 |
| `problem.json` | `detail` | Chuỗi biểu diễn Python dictionary gồm tiêu đề, nội dung, options, điểm tối đa và các trường khác |
| `student-problem-fine.json` | `log_id`, `user_id`, `problem_id` | ID sự kiện, sinh viên, bài tập |
| `student-problem-fine.json` | `is_correct`, `attempts`, `score`, `submit_time` | Kết quả, số lần thử, điểm gốc và thời gian nộp |

File fine có 14.224 chuỗi của 14.224 sinh viên, tổng cộng 898.933 lượt trả lời và 2.513 bài có tương tác. Vì thế số bài có tương tác nhỏ hơn số bài metadata. `score` thiếu ở 543.847 lượt, nhưng `is_correct` hiện diện trên toàn bộ file này. Hệ thống giữ `score` gốc; không tự chuyển điểm thành nhãn đúng/sai.

File fine không có concept hoặc Bloom trực tiếp trong sự kiện. Cần nối `problem_id` sang `problem.json`, rồi tạo `problem_concepts.csv` cho quan hệ nhiều-nhiều. ID concept được tạo ổn định bằng SHA-256 của **tên tiếng Trung gốc**; cùng một tên được xem là cùng concept giữa các khóa học. Đây là lựa chọn triển khai và cần xem lại nếu các khóa học dùng cùng tên với nghĩa khác. Bản metadata cục bộ có 9.985 tên concept riêng biệt. Bốn liên kết concept trùng trong danh sách gốc được nhận diện riêng; chúng không phải bốn lượt trả lời trùng.

Hai `detail` rỗng hoặc không phân tích được bằng `ast.literal_eval`; ID và các trường metadata cấp ngoài vẫn dùng được. Mã Bloom được giữ dạng 1–6 vì file cục bộ không có bảng ánh xạ tên mức nhận thức. Tỷ lệ đúng theo Bloom chỉ phản ánh kết quả trên câu hỏi mang nhãn đó, không xác nhận năng lực Bloom của sinh viên.

Không tìm thấy bảng tiêu đề khóa học, prerequisite, hoặc trường liên kết video tường minh trong bốn file cục bộ. Không suy ra video từ `context_id`. Số lượng của bản cục bộ khác mô tả tổng quan trong [repository MOOC-Radar](https://github.com/THU-KEG/MOOC-Radar); khi so sánh cần nêu rõ phiên bản và phạm vi file.

## Nguyên tắc đọc và kiểm tra

`src/moocradar/reading.py` đọc JSON array theo từng chuỗi sinh viên và JSON Lines theo từng dòng. Notebook [Data Quality & EDA](../notebooks/01_data_quality_eda.ipynb) kiểm tra thứ tự thời gian **gốc**, duplicate, missing, foreign key và độ bao phủ trước khi chuẩn hóa. Báo cáo trong `data/processed/<mode>/data_quality.json` được tạo sau bước chuẩn hóa; báo cáo EDA trong `artifacts/eda_<run_id>/data_quality_report.json` đánh giá raw data và có phạm vi mẫu/toàn bộ ghi rõ.
