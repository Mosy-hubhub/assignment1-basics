from __future__ import annotations

import json
from pathlib import Path
import wandb

from cs336_basics.loss import Cross_entropy_loss
from cs336_basics.utils import lr_cosine_schedule, lr_sgd_schedule, lr_constant_schedule, gradient_clipping, load_checkpoint, save_checkpoint, get_batch
from cs336_basics.optimizer import AdamW, SGD
from cs336_basics.layers import transformer_lm

import time
import numpy as np
import torch
import argparse


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # 1. 加载已经分词好的训练集和验证集
    train_data = np.memmap(
        args.train_path,
        dtype=np.uint32,
        mode="r",
    )
    val_data = np.memmap(
        args.val_path,
        dtype=np.uint32,
        mode="r",
    )

    # 2. 初始化模型，移动到训练设备
    model = transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rope=args.use_rope,
        use_norm=args.use_norm,
        norm_position=args.norm_position,
        use_swiglu=args.use_swiglu,
    )
    model = model.to(args.device)

    # 3. 初始化优化器
    if args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.max_lr)
    else:
        optimizer = AdamW(
            model.parameters(), lr=args.max_lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay, eps=args.eps
        )

    # 4. 可选：恢复 checkpoint
    start_step = 0
    if args.resume_path is not None:
        start_step = load_checkpoint(args.resume_path, model, optimizer)
    if start_step >= args.max_steps:
        print(f"checkpoint step={start_step}; max_steps={args.max_steps}, nothing to train")
        return

    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=vars(args),
        dir=str(out_dir.resolve()),
    ) as run:
        _train_steps(args, model, optimizer, train_data, val_data, start_step, out_dir, run)


class _DirectionMonitor:
    """记录参数与负梯度、当前梯度与历史方向 EMA 的余弦相似度。"""

    def __init__(self, beta):
        self.beta = beta
        self.ema = None

    @torch.no_grad()
    def update(self, model):
        parameters = [p for p in model.parameters() if p.requires_grad]
        weights = torch.cat([p.detach().flatten().float() for p in parameters])
        gradient = torch.cat([
            p.grad.detach().flatten().float() if p.grad is not None
            else torch.zeros_like(p, dtype=torch.float32).flatten()
            for p in parameters
        ])
        grad_norm = torch.linalg.vector_norm(gradient)
        weight_norm = torch.linalg.vector_norm(weights)
        # 零梯度的方向没有定义。
        if not torch.isfinite(grad_norm) or grad_norm == 0:
            return {}
        direction = gradient / grad_norm
        metrics = {"direction/grad_norm": grad_norm.item()}

        if torch.isfinite(weight_norm) and weight_norm > 0:
            metrics["alignment/negative_grad_cos"] = (-torch.dot(weights / weight_norm, direction)).clamp(-1, 1).item()
        # 与更新前的 EMA 比较，再加入当前单位梯度。
        if self.ema is not None:
            ema_norm = torch.linalg.vector_norm(self.ema)
            if ema_norm > 0:
                metrics["direction/grad_ema_cos"] = torch.dot(direction, self.ema / ema_norm).clamp(-1, 1).item()
            self.ema.mul_(self.beta).add_(direction, alpha=1 - self.beta)
        else:
            self.ema = direction.clone()
        return metrics


def _train_steps(args, model, optimizer, train_data, val_data, start_step, out_dir, run):

    start_time = time.perf_counter()
    model.train()
    # 恢复训练时，监测用的 EMA 重新积累。
    direction_monitor = _DirectionMonitor(args.direction_ema_beta) if args.track_direction else None

    # 5. 开始训练
    for step in range(start_step, args.max_steps):
        # 5.1 根据当前 step 计算学习率
        if args.lr_schedule == "constant":
            lr = lr_constant_schedule(it=step, learning_rate=args.max_lr)
        elif args.lr_schedule == "sgd":
            lr = lr_sgd_schedule(it=step, learning_rate=args.max_lr)
        else:
            lr = lr_cosine_schedule(
                it=step,
                max_learning_rate=args.max_lr,
                min_learning_rate=args.min_lr,
                warmup_iters=args.warmup_iters,
                cosine_cycle_iters=args.cosine_cycle_iters,
            )

        for group in optimizer.param_groups:
            group["lr"] = lr

        # 5.2 随机采样一个 batch
        x, y = get_batch(
            dataset=train_data, batch_size=args.batch_size, context_length=args.context_length, device=args.device
        )

        # 5.3 清空上一轮梯度
        optimizer.zero_grad()

        # 5.4 前向传播
        logits = model(x)  # [batch_size, context_length, vocab_size]

        # 5.5 计算所有位置的平均交叉熵
        loss = Cross_entropy_loss(logits.reshape(-1, args.vocab_size), y.reshape(-1))

        # 5.6 反向传播
        loss.backward()

        completed_steps = step + 1
        # 参数范数从第一步起每步记录，不受方向监测开关及起始步数限制。
        with torch.no_grad():
            weight_norm = torch.linalg.vector_norm(torch.stack([
                # 只提高统计的累加精度，避免大范数下微小变化被 FP32 舍入掩盖。
                torch.linalg.vector_norm(p.detach(), dtype=torch.float64)
                for p in model.parameters() if p.requires_grad
            ]))
        metrics = {"charts/weight_norm": weight_norm.item()}
        if direction_monitor is not None and completed_steps >= args.direction_start_step:
            # 对应本次 forward 的参数与梯度；在裁剪、优化器更新之前计算。
            metrics.update(direction_monitor.update(model))

        # 5.7 梯度裁剪
        gradient_clipping(model.parameters(), max_l2_norm=args.grad_clip)

        # 5.8 更新参数
        optimizer.step()

        # 已完成的训练步数
        completed_steps = step + 1

        # 6. 定期记录训练指标
        if completed_steps % args.log_interval == 0 or completed_steps == args.max_steps:
            elapsed = time.perf_counter() - start_time
            print(f"step={completed_steps}, loss={loss.item():.4f}, lr={lr:.6g}, time={elapsed:.1f}s")
            metrics.update(loss=loss.item(), lr=float(lr), elapsed_seconds=elapsed)

        # 7. 定期计算验证损失
        if completed_steps % args.eval_interval == 0 or completed_steps == args.max_steps:
            model.eval()

            with torch.no_grad():
                val_losses = []
                for _ in range(args.eval_batches):
                    x_eval, y_eval = get_batch(
                        dataset=val_data,
                        batch_size=args.batch_size,
                        context_length=args.context_length,
                        device=args.device,
                    )
                    logits = model(x_eval)
                    loss_eval = Cross_entropy_loss(logits.reshape(-1, args.vocab_size), y_eval.reshape(-1))
                    val_losses.append(loss_eval.item())
                loss_val = sum(val_losses) / len(val_losses)

            model.train()
            print(f"step={completed_steps}, loss_val={loss_val:.4f}")
            metrics.update(loss_val=loss_val, lr=float(lr), elapsed_seconds=time.perf_counter() - start_time)

        # 同一步的指标合并提交，避免 W&B 丢弃重复 step。
        if metrics:
            run.log(metrics, step=completed_steps)

        # 8. 定期保存 checkpoint
        if completed_steps >= args.save_start_step and completed_steps % args.save_interval == 0:
            save_checkpoint(model, optimizer, iteration=completed_steps, out=out_dir / f"step_{completed_steps:08d}.pt")

    if completed_steps >= args.save_start_step and completed_steps % args.save_interval != 0:
        save_checkpoint(model, optimizer, iteration=completed_steps, out=out_dir / f"step_{completed_steps:08d}.pt")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a Transformer LM")

    # 数据：已经分词后的 token ID 文件
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)

    # 模型参数：默认采用讲义中的 TinyStories 设置
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument(
        "--d_ff", type=int, default=None, help="默认：SwiGLU 约为 8/3*d_model（向下取 64 的倍数），SiLU 为 4*d_model"
    )
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--use_rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--norm_position", choices=["pre", "post"], default="pre")
    parser.add_argument("--use_swiglu", action=argparse.BooleanOptionalAction, default=True)

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    # lr schedule
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument(
        "--warmup_iters", "--warmup_steps", "--warmup_step", dest="warmup_iters", type=int, default=1000
    )
    parser.add_argument("--cosine_cycle_iters", type=int, default=None, help="默认等于 max_steps")

    parser.add_argument(
        "--optimizer", choices=["adamw", "sgd"], default="adamw",
        help="sgd 无动量和 weight decay；学习率由 lr_schedule 控制",
    )
    parser.add_argument(
        "--lr_schedule", choices=["constant", "sgd", "cosine"], default=None,
        help="默认：SGD 使用平方根衰减，AdamW 使用 warmup+cosine；constant 固定为 max_lr",
    )
    # 以下参数仅供 AdamW 使用；这些默认值只是调参起点。
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    # 学习率调度与梯度裁剪
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # 日志、验证与 checkpoint
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--save_start_step", type=int, default=0, help="达到此步数后才保存 checkpoint，包括最终 checkpoint")
    parser.add_argument("--track_direction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--direction_start_step", type=int, default=0, help="达到此步数后每步计算方向指标并开始积累 EMA")
    parser.add_argument("--direction_ema_beta", type=float, default=0.99, help="单位梯度方向 EMA 的衰减系数")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")

    args = parser.parse_args(argv)
    if args.lr_schedule is None:
        args.lr_schedule = "sgd" if args.optimizer == "sgd" else "cosine"
    if args.direction_start_step < 0 or args.save_start_step < 0:
        parser.error("direction_start_step and save_start_step must be nonnegative")
    if not 0 <= args.direction_ema_beta < 1:
        parser.error("direction_ema_beta must be in [0, 1)")
    if args.d_ff is None:
        args.d_ff = max(64, (8 * args.d_model // 3) // 64 * 64) if args.use_swiglu else 4 * args.d_model
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_steps
    for name in (
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "batch_size",
        "max_steps",
        "log_interval",
        "eval_interval",
        "eval_batches",
        "save_interval",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.d_model % args.num_heads != 0:
        parser.error("d_model must be divisible by num_heads")
    if args.use_rope and (args.d_model // args.num_heads) % 2 != 0:
        parser.error("RoPE requires an even head dimension")
    if args.lr_schedule == "cosine" and not 0 <= args.warmup_iters < args.cosine_cycle_iters:
        parser.error("require 0 <= warmup_iters < cosine_cycle_iters")
    if not 0 <= args.min_lr <= args.max_lr:
        parser.error("require 0 <= min_lr <= max_lr")
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1):
        parser.error("beta1 and beta2 must be in [0, 1)")
    if args.eps <= 0 or args.grad_clip <= 0 or args.weight_decay < 0:
        parser.error("eps and grad_clip must be positive; weight_decay must be nonnegative")
    return args


if __name__ == "__main__":
    train(parse_args())
