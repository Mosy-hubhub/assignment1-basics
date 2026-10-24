"""离线比较每个 run 内 checkpoint 的参数方向，不比较优化器状态或梯度。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import torch


def _model_state(path):
    before = path.stat()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["model"]
    # 当前模型的 RoPE/causal mask buffer 均为 non-persistent，model 中只有参数。
    if not state or any(not torch.is_tensor(p) or not p.is_floating_point() for p in state.values()):
        raise ValueError(f"{path}: model must contain floating-point parameter tensors")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{path}: checkpoint changed while reading; rerun after saving finishes")
    return state, checkpoint["iteration"]


@torch.no_grad()
def checkpoint_angles(paths, report, chunk_size=262144):
    """按参数名拼接，用 FP64 Gram 矩阵计算所有两两夹角，返回弧度矩阵。"""
    first, _ = _model_state(paths[0])
    keys = sorted(first)
    shapes = {key: first[key].shape for key in keys}
    parameter_count = sum(first[key].numel() for key in keys)
    del first
    # 一个 run 一个矩阵，约 7 GB/41 个 22.7M 参数 checkpoint；不占 GPU。
    weights = torch.empty((len(paths), parameter_count), dtype=torch.float64)
    steps = []
    for row, path in enumerate(paths):
        state, step = _model_state(path)
        if sorted(state) != keys or any(state[key].shape != shapes[key] for key in keys):
            raise ValueError(f"{path}: parameter names or shapes differ within this run")
        offset = 0
        for key in keys:
            value = state[key].reshape(-1)
            weights[row, offset:offset + value.numel()].copy_(value)
            offset += value.numel()
        del state
        steps.append(step)
        print(f"LOADED step={step} file={path.name}", file=report, flush=True)
    if not torch.isfinite(weights).all():
        raise ValueError("nonfinite parameters; direction is undefined")
    norms = torch.linalg.vector_norm(weights, dim=1)
    if (norms == 0).any():
        raise ValueError("zero parameter norm; direction is undefined")
    weights.div_(norms[:, None])
    # 分块累加，限制矩阵乘法的工作集大小。
    cosine = torch.zeros((len(paths), len(paths)), dtype=torch.float64)
    for start in range(0, parameter_count, chunk_size):
        block = weights[:, start:start + chunk_size]
        cosine.addmm_(block, block.T)
    del weights
    cosine.clamp_(-1, 1)
    cosine.fill_diagonal_(1)
    angles = torch.acos(cosine)
    print("CHECKPOINT step weight_norm file", file=report)
    for step, norm, path in zip(steps, norms.tolist(), paths):
        print(f"CHECKPOINT {step} {norm:.12g} {path.name}", file=report)
    print("PAIR step_a step_b cosine angle_deg direction_distance", file=report)
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            angle = angles[i, j].item()
            # 两个单位参数向量的欧氏距离。
            distance = 2 * math.sin(angle / 2)
            print(f"PAIR {steps[i]} {steps[j]} {cosine[i, j].item():.15g} "
                  f"{math.degrees(angle):.12g} {distance:.12g}", file=report)
    print("ADJACENT step_a step_b angle_deg", file=report)
    for i in range(len(paths) - 1):
        print(f"ADJACENT {steps[i]} {steps[i+1]} {math.degrees(angles[i, i+1].item()):.12g}", file=report)
    # 采样后缀集合的最大两两距离，比只看相邻 checkpoint 更接近 Cauchy 检查。
    for start in sorted({0, len(paths) // 2, max(0, len(paths) - 5)}):
        maximum = angles[start:, start:].max().item()
        print(f"TAIL from_step={steps[start]} count={len(paths)-start} "
              f"max_angle_deg={math.degrees(maximum):.12g} "
              f"max_direction_distance={2*math.sin(maximum/2):.12g}", file=report)
    report.flush()
    return angles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--run_glob", default="tinystories_*")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    # 固定本次文件列表；正在训练的 run 后续新生成的 checkpoint 留给下次分析。
    runs = [(run, sorted(run.glob("step_*.pt")))
            for run in sorted(args.runs_dir.glob(args.run_glob)) if run.is_dir()]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("x", encoding="utf-8", buffering=1) as report:
        print(f"CREATED {datetime.now(timezone.utc).isoformat()}", file=report)
        print("Parameter directions within each run; FP64 statistics; angles in degrees.", file=report)
        print("Finite checkpoint samples are evidence, not a proof of convergence.", file=report)
        for run, paths in runs:
            print(f"\nRUN {run.name} checkpoint_count={len(paths)}", file=report, flush=True)
            if not paths:
                print("SKIP no checkpoints in this snapshot", file=report)
                continue
            print(f"ANALYZING {run.name}: {len(paths)} checkpoints", flush=True)
            try:
                checkpoint_angles(paths, report)
                print("STATUS completed", file=report, flush=True)
            except (OSError, RuntimeError, ValueError, KeyError, EOFError) as error:
                print(f"STATUS failed: {error}", file=report, flush=True)
                print(f"FAILED {run.name}: {error}", flush=True)
        print("\nREPORT_COMPLETE", file=report, flush=True)
    print(f"Saved {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
