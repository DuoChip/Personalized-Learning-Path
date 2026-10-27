# Demo: Hệ thống gợi ý lộ trình học cá nhân hóa (Streamlit)

Web demo cho assignment Modern MIS. App gọi trực tiếp `moocradar.pipeline.train()` và
`moocradar.inference.infer()`; không có logic mô hình viết lại riêng cho demo.

## Chạy nhanh (local)

```bash
pip install -r requirements.txt -r demo/requirements.txt
streamlit run demo/app.py
```

Lần mở đầu tiên, app tự chạy `train` (seed 42) trên bộ mẫu `demo/sample_data/processed`
và lưu run vào `demo/.cache/artifacts/` (đã nằm trong `.gitignore`). Mất vài giây.

## Dùng run trên dữ liệu đầy đủ

```bash
export MOOC_PROCESSED_DIR=data/processed/full
export MOOC_RUN_DIR=artifacts/<run_id>      # run có manifest.json "status": "complete"
streamlit run demo/app.py
```

## Dữ liệu mẫu

`demo/sample_data/processed` là bộ mẫu cùng cấu trúc MOOC-Radar (25 sinh viên, 616 lượt,
60 bài, 15 concept, 3 khóa), lấy từ `data/processed/smoke` trên Drive của nhóm. SHA-256 của file
nguồn ghi trong `data_quality.json`. Tên concept tiếng Việt trong `glossary_vi.csv` chỉ dùng để hiển thị.

Số liệu toàn bộ MOOC-Radar trên trang Tổng quan và Đánh giá được trích từ `docs/RESULTS.md`
(run `20261017T150803777229Z_009410`); metrics của bộ mẫu chỉ minh họa quy trình đánh giá.

## Các trang

| Trang | Nội dung |
| --- | --- |
| Tổng quan hệ thống | KPI dữ liệu đầy đủ, thách thức dữ liệu, luồng DSS, bản đồ năng lực cả lớp |
| Hồ sơ người học | Radar mastery, mastery kèm độ tin cậy, Bloom, dòng thời gian chia 70/15/15 |
| Lộ trình & mô phỏng | Gap có bằng chứng, Top-5 kèm tách điểm, nộp thử một câu trả lời và xem cập nhật |
| So sánh hai sinh viên | Cùng khóa, khác hồ sơ, khác lộ trình |
| Đánh giá mô hình | Smoothed concept so với BKT: ROC, hiệu chuẩn, ma trận nhầm lẫn |
| Kiến trúc hệ thống thông tin | 4 thành phần, quy trình vận hành, FR/NFR, giới hạn |

## Kịch bản trình bày (khoảng 7 phút)

1. **Tổng quan (1')**: bài toán, quy mô dữ liệu, 4 vấn đề dữ liệu và cách xử lý.
2. **Hồ sơ người học (1,5')**: chọn một sinh viên; chỉ ra cột độ tin cậy và concept "mới tiếp xúc".
3. **Lộ trình & mô phỏng (2')**: mở "Vì sao bài #1 đứng đầu?", rồi nộp bài #1 là *Sai* và *Đúng*
   để thấy mastery và thứ hạng thay đổi trong vài ms.
4. **So sánh (1')**: hai sinh viên cùng khóa nhận Top-5 khác nhau.
5. **Đánh giá (1')**: BKT có accuracy 80% nhưng TN = 0; vì sao chọn smoothed concept.
6. **Kiến trúc IS & giới hạn (0,5')**: nêu rõ những gì chưa triển khai.

Dự phòng: quay sẵn video màn hình bước 2–3 trong trường hợp máy chiếu/mạng có sự cố.
