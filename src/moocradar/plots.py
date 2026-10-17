# File: src/moocradar/plots.py
# Purpose: Save labeled EDA charts from observed interactions.

from pathlib import Path
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(path: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def make_plots(stats: dict, output_dir: Path) -> None:
    """Generate plots only for distributions available in the local dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lengths = stats["student_lengths"]
    if lengths:
        plt.figure()
        plt.hist(lengths, bins=40)
        _save(output_dir / "student_interactions.png", "Số lượt tương tác theo sinh viên", "Số lượt", "Số sinh viên")
    attempts = list(stats["concept_attempts"].values())
    if attempts:
        plt.figure()
        plt.hist(attempts, bins=40)
        _save(output_dir / "concept_interactions.png", "Số lượt tương tác theo concept", "Số lượt", "Số concept")
    correct = stats["correct"]
    if correct:
        plt.figure()
        plt.bar(["Sai", "Đúng"], [correct[0], correct[1]])
        _save(output_dir / "correctness.png", "Phân phối kết quả làm bài", "Kết quả", "Số lượt")
    bloom = stats["bloom"]
    if bloom:
        levels = sorted(bloom, key=int)
        level_numbers = [int(level) for level in levels]
        plt.figure()
        plt.bar(level_numbers, [bloom[level][0] for level in levels])
        plt.xticks(level_numbers)
        _save(output_dir / "bloom_distribution.png", "Phân phối theo nhãn Bloom", "Mã Bloom", "Số lượt")
        plt.figure()
        plt.bar(level_numbers, [bloom[level][1] / bloom[level][0] for level in levels])
        plt.xticks(level_numbers)
        plt.ylim(0, 1)
        _save(output_dir / "bloom_accuracy.png", "Tỷ lệ đúng theo nhãn Bloom", "Mã Bloom", "Tỷ lệ đúng")
    daily = stats["daily"]
    if daily:
        days = sorted(daily)
        plt.figure(figsize=(9, 4))
        plt.plot([date.fromisoformat(day) for day in days], [daily[day] for day in days])
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(8))
        _save(output_dir / "daily_interactions.png", "Tương tác theo thời gian", "Ngày", "Số lượt")
    outcomes = stats["concept_outcomes"]
    high_error = sorted(
        ((1 - correct / total, cid) for cid, (total, correct) in outcomes.items() if total >= 20),
        reverse=True,
    )[:15]
    if high_error:
        plt.figure(figsize=(9, 5))
        plt.barh([cid for _, cid in reversed(high_error)],
                 [rate for rate, _ in reversed(high_error)])
        plt.xlim(0, 1)
        _save(output_dir / "high_error_concepts.png",
              "Concept có tỷ lệ sai cao (ít nhất 20 lượt)",
              "Tỷ lệ sai", "Mã concept")
    (output_dir / "README.md").write_text(
        "# Chú thích biểu đồ\n\n"
        "- `student_interactions.png`: số lượt trong lịch sử học tập của từng sinh viên.\n"
        "- `concept_interactions.png`: tần suất concept; một lượt có thể thuộc nhiều concept.\n"
        "- `correctness.png`: phân phối nhãn đúng/sai.\n"
        "- `bloom_accuracy.png`: tỷ lệ đúng theo nhãn bài, không xác nhận năng lực Bloom.\n"
        "- `bloom_distribution.png`: số lượt theo từng mã Bloom.\n"
        "- `high_error_concepts.png`: tỷ lệ sai cao trong nhóm có ít nhất 20 lượt.\n"
        "- `daily_interactions.png`: số lượt theo ngày khi có timestamp.\n",
        encoding="utf-8",
    )
