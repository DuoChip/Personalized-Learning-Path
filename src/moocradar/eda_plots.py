# File: src/moocradar/eda_plots.py
# Purpose: Draw Vietnamese EDA figures and write a factual Vietnamese summary.

import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(path: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _label_bars(ax, values: list[int], total: int) -> None:
    for bar, value in zip(ax.patches, values):
        percentage = 100 * value / total if total else 0
        ax.annotate(
            f"{value:,}\n({percentage:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9,
        )


def make_eda_plots(report: dict, events: dict, concept_rows: list[dict],
                   course_rows: list[dict], monthly_rows: list[dict],
                   output_dir: Path) -> dict[str, str]:
    """Save up to eight report-ready figures with evidence-based captions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "figure.figsize": (9, 5),
    })
    notes = {}
    lengths = events["student_interactions"]
    if lengths:
        plt.figure()
        plt.hist(lengths, bins=min(40, max(5, len(set(lengths)))),
                 color="#31688e", edgecolor="white")
        _save(
            output_dir / "01_tuong_tac_sinh_vien.png",
            "Phân phối số lượt tương tác theo sinh viên",
            "Số lượt tương tác", "Số sinh viên",
        )
        notes["01_tuong_tac_sinh_vien.png"] = (
            f"Đã quan sát {len(lengths):,} sinh viên; trung vị có "
            f"{sorted(lengths)[len(lengths) // 2]:,} lượt tương tác và "
            f"mức cao nhất là {max(lengths):,}. Biểu đồ cho thấy độ dài "
            "lịch sử học tập trong phạm vi dữ liệu được quét."
        )
    labels = report["labels"]
    label_total = labels.get("0", 0) + labels.get("1", 0)
    if label_total:
        values = [labels.get("0", 0), labels.get("1", 0)]
        fig, ax = plt.subplots()
        ax.bar(["Sai", "Đúng"], values, color=["#d65f5f", "#3a9d74"])
        ax.set_ylim(0, max(values) * 1.18)
        _label_bars(ax, values, label_total)
        _save(
            output_dir / "02_dung_sai.png",
            "Phân phối nhãn trả lời đúng/sai",
            "Kết quả", "Số lượt trả lời",
        )
        notes["02_dung_sai.png"] = (
            f"Tỷ lệ trả lời đúng là {100 * values[1] / label_total:.1f}% "
            f"trên {label_total:,} lượt có nhãn hợp lệ. Khi đánh giá mô hình "
            "cần xem thêm ROC-AUC/Brier bên cạnh accuracy."
        )
    concept_counts = [row["interactions"] for row in concept_rows]
    if concept_counts:
        plt.figure()
        plt.hist(
            [math.log10(1 + count) for count in concept_counts],
            bins=35, color="#5e6caf", edgecolor="white",
        )
        _save(
            output_dir / "03_tan_suat_concept.png",
            "Tần suất tương tác theo concept",
            "log10(1 + số lượt tương tác)", "Số concept",
        )
        notes["03_tan_suat_concept.png"] = (
            f"Có {len(concept_counts):,} concept xuất hiện trong các lượt trả lời "
            "đã quét. Trục ngang dùng log để quan sát cả concept hiếm và phổ biến; "
            "một lượt làm bài có thể đóng góp cho nhiều concept."
        )
    minimum = 3 if report["mode"] == "sample" else 20
    high_error = sorted(
        (row for row in concept_rows if row["interactions"] >= minimum),
        key=lambda row: (-row["error_rate"], -row["interactions"], row["concept_id"]),
    )[:12]
    if high_error:
        ordered = list(reversed(high_error))
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(
            [row["concept_id"][2:10] for row in ordered],
            [row["error_rate"] * 100 for row in ordered],
            color="#cc6d5a",
        )
        ax.set_xlim(0, 100)
        _save(
            output_dir / "04_concept_ty_le_sai_cao.png",
            f"Concept có tỷ lệ sai cao (ít nhất {minimum} lượt)",
            "Tỷ lệ sai (%)", "Mã concept (rút gọn)",
        )
        notes["04_concept_ty_le_sai_cao.png"] = (
            f"Biểu đồ chỉ xét concept có ít nhất {minimum} lượt để giảm nhiễu "
            f"do cỡ mẫu quá nhỏ. Mức sai cao nhất trong nhóm này là "
            f"{100 * high_error[0]['error_rate']:.1f}%; cần xem số lượt "
            "cụ thể trong concept_statistics.csv trước khi kết luận."
        )
    bloom = report["bloom_events"]
    bloom_total = sum(bloom.values())
    if bloom_total:
        levels = sorted(bloom, key=int)
        level_numbers = [int(level) for level in levels]
        values = [bloom[level] for level in levels]
        fig, ax = plt.subplots()
        ax.bar(level_numbers, values, color="#6f95b5")
        ax.set_xticks(level_numbers)
        ax.set_ylim(0, max(values) * 1.19)
        _label_bars(ax, values, bloom_total)
        _save(
            output_dir / "05_bloom_phan_phoi.png",
            "Phân phối lượt trả lời theo nhãn Bloom",
            "Mã mức nhận thức Bloom", "Số lượt trả lời",
        )
        notes["05_bloom_phan_phoi.png"] = (
            f"Có {bloom_total:,} lượt trả lời liên kết với nhãn Bloom hợp lệ. "
            "Các mã 1–6 được giữ nguyên vì file cục bộ không cung cấp bảng "
            "ánh xạ tên mức nhận thức."
        )
        rates = [
            report["bloom_correct"].get(level, 0) / bloom[level] * 100
            for level in levels
        ]
        fig, ax = plt.subplots()
        bars = ax.bar(level_numbers, rates, color="#4b8f87")
        ax.set_xticks(level_numbers)
        ax.set_ylim(0, 110)
        for bar, rate in zip(bars, rates):
            ax.annotate(
                f"{rate:.1f}%", (bar.get_x() + bar.get_width() / 2,
                                bar.get_height()),
                ha="center", va="bottom",
            )
        _save(
            output_dir / "06_bloom_ty_le_dung.png",
            "Tỷ lệ trả lời đúng theo nhãn Bloom",
            "Mã mức nhận thức Bloom", "Tỷ lệ đúng (%)",
        )
        notes["06_bloom_ty_le_dung.png"] = (
            "Tỷ lệ đúng được tính từ nhãn câu hỏi và kết quả trả lời thực tế. "
            "Chỉ số này không tự chứng minh năng lực Bloom của từng sinh viên."
        )
    if monthly_rows:
        months = [
            datetime.strptime(row["month"], "%Y-%m")
            for row in monthly_rows
        ]
        plt.figure(figsize=(10, 5))
        plt.plot(
            months, [row["interactions"] for row in monthly_rows],
            color="#5775a7", marker="o", markersize=3,
        )
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.xticks(rotation=30)
        _save(
            output_dir / "07_tuong_tac_theo_thang.png",
            "Lượt tương tác theo tháng",
            "Tháng", "Số lượt tương tác",
        )
        notes["07_tuong_tac_theo_thang.png"] = (
            f"Dữ liệu có timestamp hợp lệ trải từ "
            f"{monthly_rows[0]['month']} đến {monthly_rows[-1]['month']}. "
            "Biểu đồ mô tả thời điểm ghi nhận tương tác, không đo tiến bộ học tập."
        )
    if course_rows:
        top = course_rows[:10]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(
            [row["course_id"] for row in reversed(top)],
            [row["interactions"] for row in reversed(top)],
            color="#8b83b3",
        )
        _save(
            output_dir / "08_khoa_hoc_nhieu_tuong_tac.png",
            "10 khóa học có nhiều lượt tương tác nhất",
            "Số lượt tương tác", "Mã khóa học",
        )
        notes["08_khoa_hoc_nhieu_tuong_tac.png"] = (
            f"Khóa học có nhiều tương tác nhất ghi nhận "
            f"{top[0]['interactions']:,} lượt trong phạm vi đã quét. "
            "Tần suất này không phản ánh trực tiếp chất lượng khóa học."
        )
    (output_dir / "README.md").write_text(
        "# Tệp: plots/eda/README.md\n"
        "# Mục đích: Chú thích biểu đồ EDA bằng tiếng Việt.\n\n"
        "# Biểu đồ EDA\n\n"
        + "\n".join(f"- `{name}`: {note}" for name, note in notes.items())
        + "\n",
        encoding="utf-8",
    )
    return notes


def write_eda_summary(report: dict, missing_rows: list[dict],
                      plot_notes: dict[str, str], path: Path) -> None:
    """Write a Vietnamese report using only computed observations."""
    counts = report["counts"]
    total = counts.get("interactions", 0)
    missing_score = next(
        (row for row in missing_rows
         if row["dataset"] == "interactions" and row["field"] == "score"),
        {"missing_count": 0, "missing_pct": 0},
    )
    issues = [
        f"- `score` thiếu ở {missing_score['missing_count']:,}/{total:,} "
        f"lượt ({missing_score['missing_pct']:.2f}%).",
        f"- Trùng `log_id`: {counts.get('duplicate_log_id', 0):,}; "
        f"bản ghi trùng hoàn toàn: {counts.get('duplicate_exact_record', 0):,}. "
        "Quan hệ một bài–nhiều concept không được tính là bản ghi trùng.",
        f"- Có {counts.get('out_of_order_sequences', 0):,} chuỗi "
        f"và {counts.get('out_of_order_pairs', 0):,} cặp thời gian "
        "không theo thứ tự gốc.",
        f"- Timestamp không hợp lệ: {counts.get('invalid_submit_time', 0):,}; "
        f"nhãn đúng/sai bất thường: {counts.get('invalid_is_correct', 0):,}.",
        f"- Giá trị `score` âm/không hợp lệ: "
        f"{counts.get('invalid_score', 0):,}; điểm lớn hơn mức tối đa ghi "
        f"trong metadata: {counts.get('score_above_problem_max', 0):,}. "
        "Cần kiểm tra quy ước giá trị đặc biệt trước khi chuyển đổi.",
        f"- Lượt không nối được với problem metadata: "
        f"{counts.get('unmatched_problem_events', 0):,}; "
        f"lượt thiếu liên kết concept: "
        f"{counts.get('events_without_concepts', 0):,}.",
        f"- Problem có nhiều concept: "
        f"{counts.get('multi_concept_problems', 0):,}; "
        f"độ bao phủ Bloom ở cấp lượt: "
        f"{(report['coverage']['bloom_event_pct'] or 0):.2f}%.",
    ]
    recommendations = [
        "Giữ `score` gốc và giá trị thiếu; sử dụng nhãn `is_correct` gốc "
        "cho bài toán dự đoán đúng/sai. Kiểm tra riêng các điểm âm như "
        "`-1.0` trước khi coi chúng là giá trị thiếu.",
        "Kiểm tra trùng theo `log_id` và nội dung sự kiện trước khi quyết định "
        "loại bản ghi; giữ bảng liên kết nhiều-nhiều `problem_concepts`.",
        "Sắp xếp ổn định từng chuỗi theo `submit_time` trước khi chia "
        "train/validation/test; không dùng nhãn tương lai để tạo feature.",
        "Gắn cờ lượt thiếu metadata/concept; không gán mastery bằng 0 cho "
        "concept chưa quan sát.",
        "Giữ mã Bloom 1–6 cho đến khi có bảng ánh xạ được xác minh; "
        "không suy diễn năng lực cá nhân chỉ từ nhãn bài tập.",
    ]
    if not report["video_fields"]:
        issues.append(
            "- Không thấy trường video có liên kết tường minh trong metadata "
            "cục bộ; độ bao phủ video được ghi nhận là 0% theo tiêu chí này."
        )
        recommendations.append(
            "Chỉ đề xuất video khi có bảng liên kết video–problem/concept "
            "được xác minh; hiện ưu tiên bài tập."
        )
    if report["mode"] == "sample":
        scope = (
            f"Chế độ mẫu: {report['max_students']} chuỗi sinh viên đầu tiên; "
            "problem metadata được đọc đầy đủ. Các tỷ lệ tương tác chỉ mô tả mẫu."
        )
    else:
        scope = "Chế độ toàn bộ: đã quét toàn bộ file tương tác fine cục bộ."
    lines = [
        "# Tệp: eda_summary.md",
        "# Mục đích: Tóm tắt đánh giá chất lượng dữ liệu và EDA từ dữ liệu thực.",
        "",
        "# Tổng hợp chất lượng dữ liệu và EDA",
        "",
        f"**Phạm vi:** {scope}",
        "",
        "## Tổng quan",
        "",
        f"- {counts.get('problem_records', 0):,} problem metadata; "
        f"{counts.get('student_sequences', 0):,} chuỗi sinh viên; "
        f"{total:,} lượt trả lời.",
        f"- {counts.get('files_found', 0):,} file dữ liệu gốc được tìm thấy.",
        f"- {counts.get('users', 0):,} sinh viên, "
        f"{counts.get('interacted_problems', 0):,} bài có tương tác, "
        f"{counts.get('courses', 0):,} khóa học và "
        f"{counts.get('concepts', 0):,} concept trong metadata.",
        f"- Độ dài chuỗi gốc: nhỏ nhất {report['sequence_length']['min']:,}, "
        f"trung vị {report['sequence_length']['median']:,}, "
        f"lớn nhất {report['sequence_length']['max']:,} lượt.",
        "",
        "## Vấn đề chất lượng",
        "",
        *issues,
        "",
        "## Phát hiện từ biểu đồ",
        "",
        *(f"- **{name}:** {note}" for name, note in plot_notes.items()),
        "",
        "## Ảnh hưởng đến modeling",
        "",
        "Thiếu `score` làm giảm khả năng dùng điểm số liên tục; nhãn "
        "`is_correct` vẫn có thể dùng khi hợp lệ. Mất cân bằng nhãn "
        "có thể khiến accuracy đánh lừa. Các chuỗi sai thứ tự thời gian "
        "có thể gây leakage nếu chia dữ liệu theo vị trí gốc. "
        "Quan hệ nhiều concept cần được giữ riêng để mỗi lượt trả lời "
        "chỉ đóng góp một dự đoán ở cấp câu hỏi.",
        "",
        "## Khuyến nghị preprocessing",
        "",
        *(f"{i}. {text}" for i, text in enumerate(recommendations, 1)),
        "",
        "## Giới hạn diễn giải",
        "",
        "Các thống kê được tính từ file cục bộ. Không có bằng chứng về "
        "prerequisite hoặc liên kết video rõ ràng; nhãn Bloom số chưa "
        "được ánh xạ sang tên mức độ. Notebook không sửa hoặc xóa dữ liệu gốc.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
