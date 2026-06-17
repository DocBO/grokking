from __future__ import annotations

from argparse import ArgumentParser, Namespace
import csv
from dataclasses import dataclass
from pathlib import Path
import time

import torch

from data import ALL_OPERATIONS
from training import (
    add_random_force,
    collapse_ensemble,
    current_learning_rate,
    evaluate,
    evaluate_ensemble,
    evaluate_ensemble_simple,
    get_device,
    get_ensemble_config,
    get_ensemble_simple_config,
    load_data,
    loss_scaled_temperature,
    resample_weak_trajectories,
    setup_ensemble,
    setup_ensemble_simple,
    setup_model,
    should_probe_ensemble,
    train_step,
)


TRACE_FIELDS = [
    "run_id",
    "seed",
    "repeat",
    "optimizer",
    "step",
    "wall_time_s",
    "model_updates",
    "probe_count",
    "collapse_count",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "temperature",
    "free_energy",
    "best_index",
    "reached_threshold",
]

SUMMARY_FIELDS = [
    "run_id",
    "seed",
    "repeat",
    "optimizer",
    "threshold",
    "reached_threshold",
    "time_to_threshold_step",
    "time_to_threshold_s",
    "model_updates_to_threshold",
    "probe_count_to_threshold",
    "collapse_count_to_threshold",
    "final_step",
    "final_wall_time_s",
    "final_val_acc",
    "final_val_loss",
]


@dataclass
class RunResult:
    reached_threshold: bool
    threshold_step: int | None
    threshold_time_s: float | None
    threshold_model_updates: int | None
    threshold_probe_count: int | None
    threshold_collapse_count: int | None
    final_step: int
    final_wall_time_s: float
    final_val_acc: float
    final_val_loss: float


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = args.output_dir / "time_step_trace.csv"
    summary_path = args.output_dir / "time_to_threshold_summary.csv"

    with trace_path.open("w", newline="") as trace_file, summary_path.open(
        "w", newline=""
    ) as summary_file:
        trace_writer = csv.DictWriter(trace_file, fieldnames=TRACE_FIELDS)
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        trace_writer.writeheader()
        summary_writer.writeheader()

        for seed in range(args.num_seeds):
            for repeat in range(args.repeats):
                run_seed = args.seed_offset + seed + repeat * args.seed_stride
                for optimizer_name in args.optimizers:
                    run_id = f"{optimizer_name}_seed{seed}_repeat{repeat}"
                    print(f"Running {run_id} with torch seed {run_seed}")
                    torch.manual_seed(run_seed)
                    result = run_one(
                        args,
                        optimizer_name,
                        run_id,
                        seed,
                        repeat,
                        run_seed,
                        trace_writer,
                    )
                    summary_writer.writerow(
                        {
                            "run_id": run_id,
                            "seed": seed,
                            "repeat": repeat,
                            "optimizer": optimizer_name,
                            "threshold": args.threshold,
                            "reached_threshold": result.reached_threshold,
                            "time_to_threshold_step": result.threshold_step,
                            "time_to_threshold_s": result.threshold_time_s,
                            "model_updates_to_threshold": result.threshold_model_updates,
                            "probe_count_to_threshold": result.threshold_probe_count,
                            "collapse_count_to_threshold": result.threshold_collapse_count,
                            "final_step": result.final_step,
                            "final_wall_time_s": result.final_wall_time_s,
                            "final_val_acc": result.final_val_acc,
                            "final_val_loss": result.final_val_loss,
                        }
                    )
                    summary_file.flush()
                    trace_file.flush()

    print(f"Wrote trace data to {trace_path}")
    print(f"Wrote summary data to {summary_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--operation", type=str, choices=ALL_OPERATIONS.keys(), default="x/y")
    parser.add_argument("--training_fraction", type=float, default=0.5)
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--num_steps", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--seed_stride", type=int, default=100_000)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=Path("stats_runs"))
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=["adamw", "ensemble-simple", "ensemble"],
        default=["adamw", "ensemble-simple", "ensemble"],
    )
    parser.add_argument("--ensemble_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1e-5)
    parser.add_argument("--min_temperature", type=float, default=1e-7)
    parser.add_argument("--max_temperature", type=float, default=1e-2)
    parser.add_argument("--probe_interval", type=int, default=10)
    parser.add_argument("--collapse_probes", type=int, default=10)
    parser.add_argument("--temperature_heating", type=float, default=2.0)
    parser.add_argument("--temperature_cooling", type=float, default=0.5)
    parser.add_argument("--stall_window", type=int, default=5)
    parser.add_argument("--resample_fraction", type=float, default=0.5)
    parser.add_argument("--init_perturb_scale", type=float, default=1e-3)
    parser.add_argument("--collapse_perturb_scale", type=float, default=1e-3)
    parser.add_argument("--resample_perturb_scale", type=float, default=1e-3)
    parser.add_argument("--force_scale", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    return parser.parse_args()


def run_one(
    args: Namespace,
    optimizer_name: str,
    run_id: str,
    seed: int,
    repeat: int,
    run_seed: int,
    trace_writer: csv.DictWriter,
) -> RunResult:
    device = get_device(args.device)
    train_inputs, train_labels, val_inputs, val_labels, batch_size = load_data(
        args.operation,
        args.prime,
        args.training_fraction,
        args.batch_size,
        device,
        run_seed,
    )

    if optimizer_name == "adamw":
        return run_adamw(
            args,
            run_id,
            seed,
            repeat,
            train_inputs,
            train_labels,
            val_inputs,
            val_labels,
            batch_size,
            device,
            trace_writer,
        )
    if optimizer_name == "ensemble-simple":
        return run_ensemble_simple(
            args,
            run_id,
            seed,
            repeat,
            train_inputs,
            train_labels,
            val_inputs,
            val_labels,
            batch_size,
            device,
            trace_writer,
        )
    return run_ensemble(
        args,
        run_id,
        seed,
        repeat,
        train_inputs,
        train_labels,
        val_inputs,
        val_labels,
        batch_size,
        device,
        trace_writer,
    )


def run_adamw(
    args: Namespace,
    run_id: str,
    seed: int,
    repeat: int,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    val_inputs: torch.Tensor,
    val_labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
    trace_writer: csv.DictWriter,
) -> RunResult:
    model, optimizer, scheduler, criterion = setup_model(
        args.num_layers,
        args.dim_model,
        args.num_heads,
        args.prime,
        args.learning_rate,
        args.weight_decay,
        device,
    )
    n_train = len(train_inputs)
    perm = torch.randperm(n_train, device=device)
    batch_idx = 0
    start_time = time.perf_counter()
    state = ThresholdState(args.threshold)
    final_val_loss = float("nan")
    final_val_acc = 0.0

    for step in range(args.num_steps):
        if batch_idx >= n_train:
            perm = torch.randperm(n_train, device=device)
            batch_idx = 0

        idx = perm[batch_idx : batch_idx + batch_size]
        batch_idx += batch_size
        train_step(model, train_inputs[idx], train_labels[idx], optimizer, criterion)
        scheduler.step()

        completed_steps = step + 1
        if should_eval(completed_steps, args.eval_interval):
            train_loss, train_acc = evaluate(model, train_inputs, train_labels, criterion)
            final_val_loss, final_val_acc = evaluate(
                model, val_inputs, val_labels, criterion
            )
            elapsed = time.perf_counter() - start_time
            model_updates = completed_steps
            state.observe(completed_steps, elapsed, model_updates, 0, 0, final_val_acc)
            trace_writer.writerow(
                trace_row(
                    run_id,
                    seed,
                    repeat,
                    "adamw",
                    completed_steps,
                    elapsed,
                    model_updates,
                    0,
                    0,
                    train_loss,
                    train_acc,
                    final_val_loss,
                    final_val_acc,
                    None,
                    None,
                    None,
                    args.threshold,
                )
            )

    return state.result(args.num_steps, time.perf_counter() - start_time, final_val_acc, final_val_loss)


def run_ensemble_simple(
    args: Namespace,
    run_id: str,
    seed: int,
    repeat: int,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    val_inputs: torch.Tensor,
    val_labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
    trace_writer: csv.DictWriter,
) -> RunResult:
    simple_cfg = get_ensemble_simple_config(args)
    models, optimizers, schedulers, criterion = setup_ensemble_simple(
        args.num_layers,
        args.dim_model,
        args.num_heads,
        args.prime,
        args.learning_rate,
        args.weight_decay,
        simple_cfg,
        device,
    )
    n_train = len(train_inputs)
    perms = [torch.randperm(n_train, device=device) for _ in range(simple_cfg.n_trajectories)]
    batch_idxs = [0 for _ in range(simple_cfg.n_trajectories)]
    temperature = simple_cfg.init_temperature
    best_free_energy = float("inf")
    no_improve_steps = 0
    probe_count = 0
    collapse_count = 0
    start_time = time.perf_counter()
    state = ThresholdState(args.threshold)
    final_val_loss = float("nan")
    final_val_acc = 0.0

    for step in range(args.num_steps):
        for particle_idx, model in enumerate(models):
            if batch_idxs[particle_idx] >= n_train:
                perms[particle_idx] = torch.randperm(n_train, device=device)
                batch_idxs[particle_idx] = 0

            start = batch_idxs[particle_idx]
            stop = start + batch_size
            idx = perms[particle_idx][start:stop]
            batch_idxs[particle_idx] = stop
            train_step(model, train_inputs[idx], train_labels[idx], optimizers[particle_idx], criterion)
            schedulers[particle_idx].step()
            add_random_force(
                model,
                current_learning_rate(optimizers[particle_idx]),
                temperature,
                simple_cfg.force_scale,
            )

        completed_steps = step + 1
        if should_eval(completed_steps, args.eval_interval):
            probe_count += 1
            metrics = evaluate_ensemble_simple(
                models,
                train_inputs,
                train_labels,
                val_inputs,
                val_labels,
                criterion,
                simple_cfg,
                temperature,
            )
            free_energies = torch.tensor([metric.free_energy for metric in metrics], device=device)
            best_idx = int(torch.argmin(free_energies).item())
            best = metrics[best_idx]
            final_val_loss = best.val_loss
            final_val_acc = best.val_acc

            if best.free_energy < best_free_energy:
                best_free_energy = best.free_energy
                no_improve_steps = 0
                temperature = max(simple_cfg.min_temperature, temperature * simple_cfg.cooling)
            else:
                no_improve_steps += 1

            if no_improve_steps >= simple_cfg.stall_window:
                temperature = min(simple_cfg.max_temperature, temperature * simple_cfg.heating)
                no_improve_steps = 0

            resampled = resample_weak_trajectories(
                models, optimizers, free_energies, best_idx, temperature, simple_cfg
            )
            if resampled:
                collapse_count += 1

            elapsed = time.perf_counter() - start_time
            model_updates = completed_steps * simple_cfg.n_trajectories
            state.observe(
                completed_steps,
                elapsed,
                model_updates,
                probe_count,
                collapse_count,
                final_val_acc,
            )
            trace_writer.writerow(
                trace_row(
                    run_id,
                    seed,
                    repeat,
                    "ensemble-simple",
                    completed_steps,
                    elapsed,
                    model_updates,
                    probe_count,
                    collapse_count,
                    best.train_loss,
                    best.train_acc,
                    best.val_loss,
                    best.val_acc,
                    temperature,
                    best.free_energy,
                    best_idx,
                    args.threshold,
                )
            )

    return state.result(args.num_steps, time.perf_counter() - start_time, final_val_acc, final_val_loss)


def run_ensemble(
    args: Namespace,
    run_id: str,
    seed: int,
    repeat: int,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    val_inputs: torch.Tensor,
    val_labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
    trace_writer: csv.DictWriter,
) -> RunResult:
    ensemble_cfg = get_ensemble_config(args)
    models, optimizers, schedulers, criterion = setup_ensemble(
        args.num_layers,
        args.dim_model,
        args.num_heads,
        args.prime,
        args.learning_rate,
        args.weight_decay,
        ensemble_cfg,
        device,
    )
    n_train = len(train_inputs)
    perms = [torch.randperm(n_train, device=device) for _ in range(ensemble_cfg.n_trajectories)]
    batch_idxs = [0 for _ in range(ensemble_cfg.n_trajectories)]
    temperature = ensemble_cfg.min_temperature
    probes_since_collapse = 0
    probe_count = 0
    collapse_count = 0
    start_time = time.perf_counter()
    state = ThresholdState(args.threshold)
    final_val_loss = float("nan")
    final_val_acc = 0.0

    for step in range(args.num_steps):
        for particle_idx, model in enumerate(models):
            if batch_idxs[particle_idx] >= n_train:
                perms[particle_idx] = torch.randperm(n_train, device=device)
                batch_idxs[particle_idx] = 0

            start = batch_idxs[particle_idx]
            stop = start + batch_size
            idx = perms[particle_idx][start:stop]
            batch_idxs[particle_idx] = stop
            train_step(model, train_inputs[idx], train_labels[idx], optimizers[particle_idx], criterion)
            schedulers[particle_idx].step()
            add_random_force(
                model,
                current_learning_rate(optimizers[particle_idx]),
                temperature,
                ensemble_cfg.force_scale,
            )

        completed_steps = step + 1
        if should_probe_ensemble(completed_steps, ensemble_cfg.probe_interval):
            probe_count += 1
            metrics = evaluate_ensemble(
                models,
                train_inputs,
                train_labels,
                val_inputs,
                val_labels,
                criterion,
                ensemble_cfg,
                temperature,
            )
            free_energies = torch.tensor([metric.free_energy for metric in metrics], device=device)
            best_idx = int(torch.argmin(free_energies).item())
            best = metrics[best_idx]
            final_val_loss = best.val_loss
            final_val_acc = best.val_acc
            temperature = loss_scaled_temperature(best.val_loss, ensemble_cfg)
            probes_since_collapse += 1

            collapsed = 0
            if probes_since_collapse >= ensemble_cfg.collapse_probes:
                collapsed = collapse_ensemble(
                    models, optimizers, best_idx, temperature, ensemble_cfg
                )
                probes_since_collapse = 0
                collapse_count += 1

            elapsed = time.perf_counter() - start_time
            model_updates = completed_steps * ensemble_cfg.n_trajectories
            state.observe(
                completed_steps,
                elapsed,
                model_updates,
                probe_count,
                collapse_count,
                final_val_acc,
            )
            trace_writer.writerow(
                trace_row(
                    run_id,
                    seed,
                    repeat,
                    "ensemble",
                    completed_steps,
                    elapsed,
                    model_updates,
                    probe_count,
                    collapse_count,
                    best.train_loss,
                    best.train_acc,
                    best.val_loss,
                    best.val_acc,
                    temperature,
                    best.free_energy,
                    best_idx,
                    args.threshold,
                )
            )
            _ = collapsed

    return state.result(args.num_steps, time.perf_counter() - start_time, final_val_acc, final_val_loss)


class ThresholdState:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.reached = False
        self.step: int | None = None
        self.wall_time_s: float | None = None
        self.model_updates: int | None = None
        self.probe_count: int | None = None
        self.collapse_count: int | None = None

    def observe(
        self,
        step: int,
        wall_time_s: float,
        model_updates: int,
        probe_count: int,
        collapse_count: int,
        val_acc: float,
    ) -> None:
        if self.reached or val_acc < self.threshold:
            return
        self.reached = True
        self.step = step
        self.wall_time_s = wall_time_s
        self.model_updates = model_updates
        self.probe_count = probe_count
        self.collapse_count = collapse_count

    def result(
        self,
        final_step: int,
        final_wall_time_s: float,
        final_val_acc: float,
        final_val_loss: float,
    ) -> RunResult:
        return RunResult(
            reached_threshold=self.reached,
            threshold_step=self.step,
            threshold_time_s=self.wall_time_s,
            threshold_model_updates=self.model_updates,
            threshold_probe_count=self.probe_count,
            threshold_collapse_count=self.collapse_count,
            final_step=final_step,
            final_wall_time_s=final_wall_time_s,
            final_val_acc=final_val_acc,
            final_val_loss=final_val_loss,
        )


def should_eval(step: int, eval_interval: int) -> bool:
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    return step == 1 or step % eval_interval == 0


def trace_row(
    run_id: str,
    seed: int,
    repeat: int,
    optimizer_name: str,
    step: int,
    wall_time_s: float,
    model_updates: int,
    probe_count: int,
    collapse_count: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    temperature: float | None,
    free_energy: float | None,
    best_idx: int | None,
    threshold: float,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "seed": seed,
        "repeat": repeat,
        "optimizer": optimizer_name,
        "step": step,
        "wall_time_s": wall_time_s,
        "model_updates": model_updates,
        "probe_count": probe_count,
        "collapse_count": collapse_count,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "temperature": temperature,
        "free_energy": free_energy,
        "best_index": best_idx,
        "reached_threshold": val_acc >= threshold,
    }


if __name__ == "__main__":
    main()
