"""重新计算各 run 的 checkpoint 两两方向，只输出按优化器分组的汇总表。"""

import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path

import torch

from cs336_basics.checkpoint_angles import checkpoint_angles


def summarize(angles):
    n = len(angles)
    result = {}
    for label, start in [("all", 0), ("latter_half", n // 2), ("last5", max(0, n - 5))]:
        maximum = angles[start:, start:].max().item()
        result[label] = {
            "start_index": start,
            "count": n - start,
            "max_angle_deg": math.degrees(maximum),
            "max_direction_distance": 2 * math.sin(maximum / 2),
        }
    return result


def write_report(path, rows, skipped):
    with path.open("w", encoding="utf-8") as report:
        report.write("# Checkpoint parameter-direction summary\n\n")
        report.write("FP64; each maximum covers ALL checkpoint pairs within its range. "
                     "All = all available checkpoints; half = latter half by checkpoint count; "
                     "last5 = last five. Angles are in degrees; distances compare unit vectors.\n\n")
        for optimizer in ["adamw", "sgd"]:
            group = [row for row in rows if row["optimizer"] == optimizer]
            if not group:
                continue
            report.write(f"## {optimizer.upper()}\n\n")
            report.write("| Run (weight decay) | LR schedule | Learning rate | "
                         "all max_angle_deg | all max_direction_distance | "
                         "half max_angle_deg | half max_direction_distance | "
                         "last5 max_angle_deg | last5 max_direction_distance |\n")
            report.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
            for row in group:
                values = []
                for label in ["all", "latter_half", "last5"]:
                    stat = row["statistics"][label]
                    values.extend([f"{stat['max_angle_deg']:.8g}", f"{stat['max_direction_distance']:.8g}"])
                report.write(f"| {row['run']} (wd={row['weight_decay']:g}) | "
                             f"{row['lr_schedule']} | {row['learning_rate']} | " + " | ".join(values) + " |\n")
            report.write("\n")
        report.write(f"Computed runs: {len(rows)}; pair comparisons: "
                     f"{sum(row['pair_count'] for row in rows)}; skipped empty runs: {len(skipped)}.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.output_path.exists() or args.output_path.with_suffix(".json").exists():
        parser.error("output already exists; choose a new output_path")
    torch.set_num_threads(args.threads)
    runs = [(p, sorted(p.glob("step_*.pt"))) for p in sorted(args.runs_dir.glob("tinystories_*")) if p.is_dir()]
    rows, skipped, errors = [], [], []
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    for run, paths in runs:
        if not paths:
            skipped.append(run.name)
            continue
        print(f"ANALYZING {run.name}: {len(paths)} checkpoints", flush=True)
        try:
            config = json.loads((run / "args.json").read_text(encoding="utf-8"))
            optimizer = config.get("optimizer", "adamw")
            schedule = config.get("lr_schedule") or ("sgd" if optimizer == "sgd" else "cosine")
            maximum, minimum = config["max_lr"], config["min_lr"]
            if schedule == "constant":
                lr = f"{maximum:g}"
                schedule_label = "constant"
            elif schedule == "sgd":
                lr = f"{maximum:g}/sqrt(step+1)"
                schedule_label = "inverse-sqrt"
            else:
                lr = f"{maximum:g} -> {minimum:g}"
                schedule_label = f"warmup({config['warmup_iters']})+cosine"
            with io.StringIO() as details:
                angles = checkpoint_angles(paths, details)
                steps = [int(line.split()[1]) for line in details.getvalue().splitlines()
                         if line.startswith("CHECKPOINT ") and not line.startswith("CHECKPOINT step")]
            stats = summarize(angles)
            for value in stats.values():
                value.update(first_step=steps[value["start_index"]], last_step=steps[-1])
            row = {"run": run.name, "optimizer": optimizer, "weight_decay": config["weight_decay"],
                   "lr_schedule": schedule_label, "learning_rate": lr, "checkpoint_count": len(paths),
                   "pair_count": len(paths) * (len(paths) - 1) // 2, "statistics": stats,
                   "checkpoints": [str(p) for p in paths]}
            rows.append(row)
            write_report(args.output_path, rows, skipped)
            print(f"DONE {run.name}", flush=True)
        except Exception as error:
            errors.append({"run": run.name, "error": repr(error)})
            print(f"FAILED {run.name}: {error}", flush=True)
    write_report(args.output_path, rows, skipped)
    payload = {"created_at": created, "runs": rows, "skipped_empty": skipped, "errors": errors}
    args.output_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if errors:
        with args.output_path.open("a", encoding="utf-8") as report:
            report.write("\nErrors: " + json.dumps(errors) + "\n")
        raise SystemExit(1)
    print(f"COMPLETE {len(rows)} runs; saved {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
