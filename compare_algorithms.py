"""同じデータ条件・seed・学習予算でI2Iアルゴリズムを順次比較する。"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_ALGORITHMS = ["regression", "i2i_rfr_x0", "resshift"]


def _run(command, dry_run=False):
    print(f"\n$ {shlex.join(command)}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def _git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _save_summary(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="同一条件で複数アルゴリズムを学習・評価する"
    )
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument(
        "--algorithms", nargs="+", default=DEFAULT_ALGORITHMS
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--budget_mode", choices=["steps", "minutes"], default="steps")
    parser.add_argument("--budget", type=float, default=100000)
    parser.add_argument(
        "--schedule_steps",
        type=int,
        default=100000,
        help="時間予算モードでLR scheduleに使う最大step数",
    )
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--max_lr", type=float, default=2e-4)
    parser.add_argument(
        "--mixed_precision_policy",
        choices=["float32", "mixed_float16", "mixed_bfloat16"],
        default="mixed_float16",
    )
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="全アルゴリズムへ共通で追加するOmegaConf override",
    )
    args = parser.parse_args()

    args.exp_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.exp_root / "comparison_summary.json"
    summary = {
        "git_revision": _git_revision(),
        "budget_mode": args.budget_mode,
        "budget": args.budget,
        "optimizer": args.optimizer,
        "max_lr": args.max_lr,
        "mixed_precision_policy": args.mixed_precision_policy,
        "runs": [],
    }

    for seed in args.seeds:
        for algorithm in args.algorithms:
            run_dir = args.exp_root / f"{algorithm}_seed{seed}"
            if args.budget_mode == "steps":
                num_train_steps = int(args.budget)
                max_train_minutes = 0.0
            else:
                num_train_steps = int(args.schedule_steps)
                max_train_minutes = float(args.budget)

            overrides = [
                f"exp_dir={run_dir}",
                f"algorithm.name={algorithm}",
                f"reproducibility.seed={seed}",
                f"num_train_steps={num_train_steps}",
                f"eval_every={args.eval_every}",
                f"max_train_minutes={max_train_minutes}",
                f"gpu={args.gpu}",
                f"optimizer.name={args.optimizer}",
                f"optimizer.{args.optimizer}.max_lr={args.max_lr}",
                f"mixed_precision_policy={args.mixed_precision_policy}",
                *args.overrides,
            ]
            train_command = [sys.executable, "main.py", "--overrides", *overrides]
            started_at = time.monotonic()
            train_returncode = _run(train_command, dry_run=args.dry_run)
            elapsed_seconds = time.monotonic() - started_at

            run_result = {
                "algorithm": algorithm,
                "seed": seed,
                "exp_dir": str(run_dir),
                "elapsed_seconds": elapsed_seconds,
                "train_returncode": train_returncode,
                "train_command": train_command,
            }

            checkpoint = run_dir / "checkpoints" / "model_best.keras"
            eval_json = run_dir / "evaluation.json"
            if (
                train_returncode == 0
                and not args.skip_eval
                and (args.dry_run or checkpoint.exists())
            ):
                eval_command = [
                    sys.executable,
                    "eval.py",
                    str(checkpoint),
                    "--gpu",
                    str(args.gpu),
                    "--seed",
                    str(seed),
                    "--output_json",
                    str(eval_json),
                ]
                run_result["eval_returncode"] = _run(
                    eval_command, dry_run=args.dry_run
                )
                run_result["eval_command"] = eval_command
                run_result["evaluation_json"] = str(eval_json)

            summary["runs"].append(run_result)
            _save_summary(summary_path, summary)

            if train_returncode != 0 and not args.dry_run:
                print(f"Training failed for {algorithm}, seed={seed}", file=sys.stderr)

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
