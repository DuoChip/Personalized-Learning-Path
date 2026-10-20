# File: src/moocradar/__main__.py
# Purpose: Command-line entry points for preparation, training, inference, and translation.

import argparse
import json
import logging
from pathlib import Path

from .inference import infer
from .ncdm_inference import infer_ncdm
from .ncdm_pipeline import train_ncdm
from .eda import run_eda
from .pipeline import train
from .preparation import prepare
from .translation import apply, extract


def _hidden_dims(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive hidden dimensions from the CLI."""
    try:
        dimensions = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("hidden dims must be comma-separated integers") from error
    if not dimensions or any(value < 1 for value in dimensions):
        raise argparse.ArgumentTypeError("hidden dims must be positive")
    return dimensions


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m moocradar")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Validate and normalize raw data")
    prep.add_argument("--data-root", type=Path, default=Path("data/raw"))
    prep.add_argument("--output-dir", type=Path, default=Path("data/processed/full"))
    prep.add_argument("--max-students", type=int)

    fit = sub.add_parser("train", help="Train and evaluate saved models")
    fit.add_argument("--processed-dir", type=Path, default=Path("data/processed/full"))
    fit.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--mastery-target", type=float, default=0.7)
    fit.add_argument("--top-n", type=int, default=5)
    fit.add_argument("--model", choices=("bkt", "ncdm"), default="bkt")
    fit.add_argument("--granularity", choices=("coarse", "middle", "fine"), default="middle")
    fit.add_argument("--batch-size", type=int, default=512)
    fit.add_argument("--learning-rate", type=float, default=0.002)
    fit.add_argument("--epochs", type=int, default=20)
    fit.add_argument("--patience", type=int, default=4)
    fit.add_argument("--hidden-dims", type=_hidden_dims, default=(256, 128))
    fit.add_argument("--dropout", type=float, default=0.5)
    fit.add_argument("--num-workers", type=int, default=0)
    fit.add_argument("--device", default="auto")
    fit.add_argument("--resume", type=Path)

    run = sub.add_parser("run", help="Prepare then train in one command")
    run.add_argument("--mode", choices=("smoke_test", "full_run"), default="smoke_test")
    run.add_argument("--data-root", type=Path, default=Path("data/raw"))
    run.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    run.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    run.add_argument("--smoke-students", type=int, default=25)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--model", choices=("bkt", "ncdm"), default="bkt")
    run.add_argument("--granularity", choices=("coarse", "middle", "fine"), default="middle")
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--learning-rate", type=float, default=0.002)
    run.add_argument("--epochs", type=int, default=20)
    run.add_argument("--patience", type=int, default=4)
    run.add_argument("--hidden-dims", type=_hidden_dims, default=(256, 128))
    run.add_argument("--dropout", type=float, default=0.5)
    run.add_argument("--num-workers", type=int, default=0)
    run.add_argument("--device", default="auto")

    eda = sub.add_parser("eda", help="Audit raw data and export Vietnamese EDA")
    eda.add_argument("--data-root", type=Path, default=Path("data/raw"))
    eda.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    eda.add_argument("--mode", choices=("sample", "full"), default="sample")
    eda.add_argument("--sample-students", type=int, default=25)

    demo = sub.add_parser("demo", help="Load a model and emit one updated learner profile")
    demo.add_argument("--run-dir", type=Path, required=True)
    demo.add_argument("--processed-dir", type=Path, required=True)
    demo.add_argument("--user-id", required=True)
    demo.add_argument("--new-problem-id")
    demo.add_argument("--new-is-correct", type=int, choices=(0, 1))
    demo.add_argument("--target-course-id")
    demo.add_argument("--top-n", type=int, default=5)
    demo.add_argument("--model", choices=("auto", "bkt", "ncdm"), default="auto")
    demo.add_argument("--device", default="cpu")

    translation = sub.add_parser("translation", help="Extract or apply offline translations")
    translation.add_argument("action", choices=("extract", "apply"))
    translation.add_argument("--problem-file", type=Path, default=Path("data/raw/problem.json"))
    translation.add_argument("--cache", type=Path, default=Path("data/translated/translation_cache.csv"))
    translation.add_argument("--output", type=Path, default=Path("data/translated/problem_text_en.csv"))
    translation.add_argument("--limit", type=int)
    translation.add_argument("--batch-size", type=int, default=1000)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "prepare":
        print(json.dumps(prepare(args.data_root, args.output_dir, args.max_students), indent=2))
    elif args.command == "eda":
        run_dir = run_eda(
            args.data_root, args.artifact_root,
            args.sample_students if args.mode == "sample" else None,
        )
        print(json.dumps({"run_dir": str(run_dir), "status": "complete"}))
    elif args.command == "train":
        if args.model == "ncdm":
            run_dir, metrics = train_ncdm(
                args.processed_dir, args.artifact_root, seed=args.seed,
                granularity=args.granularity, batch_size=args.batch_size,
                learning_rate=args.learning_rate, epochs=args.epochs,
                patience=args.patience, hidden_dims=args.hidden_dims,
                dropout=args.dropout, num_workers=args.num_workers, device=args.device,
                mastery_target=args.mastery_target, top_n=args.top_n,
                resume_path=args.resume,
            )
        else:
            run_dir, metrics = train(args.processed_dir, args.artifact_root,
                                     seed=args.seed, mastery_target=args.mastery_target,
                                     top_n=args.top_n)
        print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))
    elif args.command == "run":
        output = args.processed_root / ("smoke" if args.mode == "smoke_test" else "full")
        prepare(args.data_root, output, args.smoke_students if args.mode == "smoke_test" else None)
        if args.model == "ncdm":
            run_dir, metrics = train_ncdm(
                output, args.artifact_root, seed=args.seed,
                granularity=args.granularity, batch_size=args.batch_size,
                learning_rate=args.learning_rate, epochs=args.epochs,
                patience=args.patience, hidden_dims=args.hidden_dims,
                dropout=args.dropout, num_workers=args.num_workers, device=args.device,
            )
        else:
            run_dir, metrics = train(output, args.artifact_root, seed=args.seed)
        print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))
    elif args.command == "demo":
        model = args.model
        if model == "auto":
            model = "ncdm" if (args.run_dir / "models" / "best_ncdm.pt").exists() else "bkt"
        if model == "ncdm":
            if args.new_problem_id is not None or args.new_is_correct is not None:
                raise ValueError(
                    "Standard NCDM has no real-time state update; retrain periodically or use BKT"
                )
            result = infer_ncdm(
                args.user_id, args.processed_dir, args.run_dir,
                args.target_course_id, args.top_n, device=args.device,
            )
        else:
            result = infer(args.user_id, args.processed_dir, args.run_dir,
                           args.new_problem_id, args.new_is_correct,
                           args.target_course_id, args.top_n)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.action == "extract":
        count = extract(args.problem_file, args.cache, args.limit, args.batch_size)
        print(json.dumps({"unique_strings": count, "cache": str(args.cache)}))
    else:
        print(json.dumps(apply(args.problem_file, args.cache, args.output)))


if __name__ == "__main__":
    main()
