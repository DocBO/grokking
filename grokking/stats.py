from __future__ import annotations

from argparse import ArgumentParser, Namespace
import csv
from dataclasses import dataclass
from pathlib import Path
import statistics
import time
from typing import cast

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
    loss_scaled_max_temperature,
    loss_scaled_temperature,
    resample_weak_trajectories,
    setup_ensemble,
    setup_ensemble_simple,
    setup_model,
    should_probe_ensemble,
    time_cool_max_temperature,
    train_step,
)


TRACE_FIELDS = [
    "run_id",
    "seed",
    "repeat",
    "optimizer",
    "training_fraction",
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
    "temperature_ceiling",
    "free_energy",
    "best_index",
    "reached_threshold",
]

SUMMARY_FIELDS = [
    "run_id",
    "seed",
    "repeat",
    "optimizer",
    "training_fraction",
    "max_steps",
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
    "best_val_acc",
    "best_val_loss",
]

AGGREGATE_FIELDS = [
    "training_fraction",
    "optimizer",
    "max_steps",
    "runs",
    "successes",
    "success_rate",
    "best_val_acc",
    "mean_best_val_acc",
    "median_best_val_acc",
    "mean_steps_to_transition",
    "median_steps_to_transition",
    "min_steps_to_transition",
    "max_steps_to_transition",
    "mean_time_to_transition_s",
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
    best_val_acc: float
    best_val_loss: float


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = args.output_dir / "time_step_trace.csv"
    summary_path = args.output_dir / "time_to_threshold_summary.csv"
    aggregate_path = args.output_dir / "sweep_summary.csv"
    summary_rows: list[dict[str, object]] = []

    with (
        trace_path.open("w", newline="") as trace_file,
        summary_path.open("w", newline="") as summary_file,
    ):
        trace_writer = csv.DictWriter(trace_file, fieldnames=TRACE_FIELDS)
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        trace_writer.writeheader()
        summary_writer.writeheader()

        for training_fraction in args.training_fractions:
            args.training_fraction = training_fraction
            for seed in range(args.num_seeds):
                for repeat in range(args.repeats):
                    run_seed = args.seed_offset + seed + repeat * args.seed_stride
                    for optimizer_name in args.optimizers:
                        fraction_label = f"{training_fraction:.2f}"
                        run_id = (
                            f"{optimizer_name}_fraction{fraction_label}_"
                            f"seed{seed}_repeat{repeat}"
                        )
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
                        row: dict[str, object] = {
                            "run_id": run_id,
                            "seed": seed,
                            "repeat": repeat,
                            "optimizer": optimizer_name,
                            "training_fraction": training_fraction,
                            "max_steps": max_steps_for_optimizer(
                                args, optimizer_name
                            ),
                            "threshold": args.threshold,
                            "reached_threshold": result.reached_threshold,
                            "time_to_threshold_step": result.threshold_step,
                            "time_to_threshold_s": result.threshold_time_s,
                            "model_updates_to_threshold": (
                                result.threshold_model_updates
                            ),
                            "probe_count_to_threshold": result.threshold_probe_count,
                            "collapse_count_to_threshold": (
                                result.threshold_collapse_count
                            ),
                            "final_step": result.final_step,
                            "final_wall_time_s": result.final_wall_time_s,
                            "final_val_acc": result.final_val_acc,
                            "final_val_loss": result.final_val_loss,
                            "best_val_acc": result.best_val_acc,
                            "best_val_loss": result.best_val_loss,
                        }
                        summary_writer.writerow(row)
                        summary_rows.append(row)
                        summary_file.flush()
                        trace_file.flush()

    write_aggregate_summary(aggregate_path, summary_rows)

    print(f"Wrote trace data to {trace_path}")
    print(f"Wrote summary data to {summary_path}")
    print(f"Wrote aggregate sweep data to {aggregate_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--operation", type=str, choices=ALL_OPERATIONS.keys(), default="x/y"
    )
    parser.add_argument("--training_fraction", type=float, default=0.5)
    parser.add_argument(
        "--training_fractions",
        nargs="+",
        type=float,
        default=None,
        help="training fractions to sweep; defaults to --training_fraction",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="first training fraction in a generated sweep",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="number of fractions in a generated sweep",
    )
    parser.add_argument(
        "--stepsize",
        type=float,
        default=0.01,
        help="increment between generated training fractions",
    )
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--num_steps", type=int, default=100_000)
    parser.add_argument(
        "--adamw_num_steps",
        type=int,
        default=None,
        help="AdamW-specific maximum; defaults to --num_steps",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--seed_stride", type=int, default=100_000)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument(
        "--strict_threshold",
        action="store_true",
        help="count success only when validation accuracy is greater than threshold",
    )
    parser.add_argument(
        "--stop_at_threshold",
        action="store_true",
        help="stop each run at its first successful validation evaluation",
    )
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=Path("stats_runs"))
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=["adamw", "ensemble-simple", "ensemble"],
        default=["adamw", "ensemble-simple", "ensemble"],
    )
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "ensemble-simple", "ensemble"],
        default=None,
        help="run only one optimizer; overrides --optimizers",
    )
    parser.add_argument("--ensemble_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1e-5)
    parser.add_argument(
        "--temperature_start",
        type=float,
        default=None,
        help=(
            "initial temperature for ensemble-simple; "
            "defaults to --temperature when omitted"
        ),
    )
    parser.add_argument("--min_temperature", type=float, default=1e-7)
    parser.add_argument("--max_temperature", type=float, default=1e-2)
    parser.add_argument("--probe_interval", type=int, default=10)
    parser.add_argument("--collapse_probes", type=int, default=10)
    parser.add_argument("--temperature_heating", type=float, default=2.0)
    parser.add_argument("--temperature_cooling", type=float, default=0.5)
    parser.add_argument("--time_cooling_rate", type=float, default=1e-4)
    parser.add_argument("--stall_window", type=int, default=5)
    parser.add_argument("--resample_fraction", type=float, default=0.5)
    parser.add_argument("--init_perturb_scale", type=float, default=1e-3)
    parser.add_argument("--collapse_perturb_scale", type=float, default=1e-3)
    parser.add_argument("--resample_perturb_scale", type=float, default=1e-3)
    parser.add_argument("--force_scale", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    args = parser.parse_args()
    if args.fraction is not None:
        fraction_steps = 1 if args.steps is None else args.steps
        if fraction_steps <= 0:
            parser.error("--steps must be positive")
        args.training_fractions = [
            round(args.fraction + index * args.stepsize, 10)
            for index in range(fraction_steps)
        ]
    elif args.steps is not None:
        parser.error("--steps requires --fraction")
    elif args.training_fractions is None:
        args.training_fractions = [args.training_fraction]
    if args.optimizer is not None:
        args.optimizers = [args.optimizer]
    if args.adamw_num_steps is not None and args.adamw_num_steps <= 0:
        parser.error("--adamw_num_steps must be positive")
    return args


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
    state = ThresholdState(args.threshold, args.strict_threshold)
    final_val_loss = float("nan")
    final_val_acc = 0.0
    final_step = 0

    for step in range(max_steps_for_optimizer(args, "adamw")):
        if batch_idx >= n_train:
            perm = torch.randperm(n_train, device=device)
            batch_idx = 0

        idx = perm[batch_idx : batch_idx + batch_size]
        batch_idx += batch_size
        train_step(model, train_inputs[idx], train_labels[idx], optimizer, criterion)
        scheduler.step()

        completed_steps = step + 1
        final_step = completed_steps
        if should_eval(completed_steps, args.eval_interval):
            train_loss, train_acc = evaluate(
                model, train_inputs, train_labels, criterion
            )
            final_val_loss, final_val_acc = evaluate(
                model, val_inputs, val_labels, criterion
            )
            elapsed = time.perf_counter() - start_time
            model_updates = completed_steps
            state.observe(completed_steps, elapsed, model_updates, 0, 0, final_val_acc)
            state.observe_loss(final_val_loss)
            trace_writer.writerow(
                trace_row(
                    run_id,
                    seed,
                    repeat,
                    "adamw",
                    args.training_fraction,
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
                    None,
                    args.threshold,
                    args.strict_threshold,
                )
            )
            if args.stop_at_threshold and state.reached:
                break

    return state.result(
        final_step, time.perf_counter() - start_time, final_val_acc, final_val_loss
    )


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
    perms = [
        torch.randperm(n_train, device=device) for _ in range(simple_cfg.n_trajectories)
    ]
    batch_idxs = [0 for _ in range(simple_cfg.n_trajectories)]
    temperature = simple_cfg.init_temperature
    time_temperature_ceiling = simple_cfg.max_temperature
    best_free_energy = float("inf")
    no_improve_steps = 0
    probe_count = 0
    collapse_count = 0
    start_time = time.perf_counter()
    state = ThresholdState(args.threshold, args.strict_threshold)
    final_val_loss = float("nan")
    final_val_acc = 0.0
    final_step = 0

    for step in range(args.num_steps):
        for particle_idx, model in enumerate(models):
            if batch_idxs[particle_idx] >= n_train:
                perms[particle_idx] = torch.randperm(n_train, device=device)
                batch_idxs[particle_idx] = 0

            start = batch_idxs[particle_idx]
            stop = start + batch_size
            idx = perms[particle_idx][start:stop]
            batch_idxs[particle_idx] = stop
            train_step(
                model,
                train_inputs[idx],
                train_labels[idx],
                optimizers[particle_idx],
                criterion,
            )
            schedulers[particle_idx].step()
            add_random_force(
                model,
                current_learning_rate(optimizers[particle_idx]),
                temperature,
                simple_cfg.force_scale,
            )

        time_temperature_ceiling = time_cool_max_temperature(
            time_temperature_ceiling,
            simple_cfg.time_cooling_rate,
            simple_cfg.min_temperature,
        )
        temperature = min(temperature, time_temperature_ceiling)
        completed_steps = step + 1
        final_step = completed_steps
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
            free_energies = torch.tensor(
                [metric.free_energy for metric in metrics], device=device
            )
            best_idx = int(torch.argmin(free_energies).item())
            best = metrics[best_idx]
            final_val_loss = best.val_loss
            final_val_acc = best.val_acc
            temperature_ceiling = min(
                time_temperature_ceiling,
                loss_scaled_max_temperature(best.val_loss, simple_cfg),
            )

            if best.free_energy < best_free_energy:
                best_free_energy = best.free_energy
                no_improve_steps = 0
                temperature = max(
                    simple_cfg.min_temperature, temperature * simple_cfg.cooling
                )
            else:
                no_improve_steps += 1

            if no_improve_steps >= simple_cfg.stall_window:
                temperature = min(temperature_ceiling, temperature * simple_cfg.heating)
                no_improve_steps = 0

            temperature = min(temperature, temperature_ceiling)
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
            state.observe_loss(final_val_loss)
            trace_writer.writerow(
                trace_row(
                    run_id,
                    seed,
                    repeat,
                    "ensemble-simple",
                    args.training_fraction,
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
                    temperature_ceiling,
                    best.free_energy,
                    best_idx,
                    args.threshold,
                    args.strict_threshold,
                )
            )
            if args.stop_at_threshold and state.reached:
                break

    return state.result(
        final_step, time.perf_counter() - start_time, final_val_acc, final_val_loss
    )


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
    perms = [
        torch.randperm(n_train, device=device)
        for _ in range(ensemble_cfg.n_trajectories)
    ]
    batch_idxs = [0 for _ in range(ensemble_cfg.n_trajectories)]
    temperature = ensemble_cfg.min_temperature
    probes_since_collapse = 0
    probe_count = 0
    collapse_count = 0
    start_time = time.perf_counter()
    state = ThresholdState(args.threshold, args.strict_threshold)
    final_val_loss = float("nan")
    final_val_acc = 0.0
    final_step = 0

    for step in range(args.num_steps):
        for particle_idx, model in enumerate(models):
            if batch_idxs[particle_idx] >= n_train:
                perms[particle_idx] = torch.randperm(n_train, device=device)
                batch_idxs[particle_idx] = 0

            start = batch_idxs[particle_idx]
            stop = start + batch_size
            idx = perms[particle_idx][start:stop]
            batch_idxs[particle_idx] = stop
            train_step(
                model,
                train_inputs[idx],
                train_labels[idx],
                optimizers[particle_idx],
                criterion,
            )
            schedulers[particle_idx].step()
            add_random_force(
                model,
                current_learning_rate(optimizers[particle_idx]),
                temperature,
                ensemble_cfg.force_scale,
            )

        completed_steps = step + 1
        final_step = completed_steps
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
            free_energies = torch.tensor(
                [metric.free_energy for metric in metrics], device=device
            )
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
            state.observe_loss(final_val_loss)
            trace_writer.writerow(
                trace_row(
                    run_id,
                    seed,
                    repeat,
                    "ensemble",
                    args.training_fraction,
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
                    None,
                    best.free_energy,
                    best_idx,
                    args.threshold,
                    args.strict_threshold,
                )
            )
            _ = collapsed
            if args.stop_at_threshold and state.reached:
                break

    return state.result(
        final_step, time.perf_counter() - start_time, final_val_acc, final_val_loss
    )


class ThresholdState:
    def __init__(self, threshold: float, strict: bool = False) -> None:
        self.threshold = threshold
        self.strict = strict
        self.reached = False
        self.step: int | None = None
        self.wall_time_s: float | None = None
        self.model_updates: int | None = None
        self.probe_count: int | None = None
        self.collapse_count: int | None = None
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")

    def observe(
        self,
        step: int,
        wall_time_s: float,
        model_updates: int,
        probe_count: int,
        collapse_count: int,
        val_acc: float,
    ) -> None:
        self.best_val_acc = max(self.best_val_acc, val_acc)
        if self.reached or not self.is_success(val_acc):
            return
        self.reached = True
        self.step = step
        self.wall_time_s = wall_time_s
        self.model_updates = model_updates
        self.probe_count = probe_count
        self.collapse_count = collapse_count

    def observe_loss(self, val_loss: float) -> None:
        self.best_val_loss = min(self.best_val_loss, val_loss)

    def is_success(self, val_acc: float) -> bool:
        if self.strict:
            return val_acc > self.threshold
        return val_acc >= self.threshold

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
            best_val_acc=self.best_val_acc,
            best_val_loss=self.best_val_loss,
        )


def should_eval(step: int, eval_interval: int) -> bool:
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    return step == 1 or step % eval_interval == 0


def max_steps_for_optimizer(args: Namespace, optimizer_name: str) -> int:
    if optimizer_name == "adamw" and args.adamw_num_steps is not None:
        return int(args.adamw_num_steps)
    return int(args.num_steps)


def write_aggregate_summary(
    output_path: Path, summary_rows: list[dict[str, object]]
) -> None:
    groups: dict[tuple[float, str], list[dict[str, object]]] = {}
    for row in summary_rows:
        key = (
            cast(float, row["training_fraction"]),
            cast(str, row["optimizer"]),
        )
        groups.setdefault(key, []).append(row)

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        for (training_fraction, optimizer_name), rows in sorted(groups.items()):
            successful_rows = [
                row for row in rows if bool(row["reached_threshold"])
            ]
            best_accuracies = [cast(float, row["best_val_acc"]) for row in rows]
            transition_steps = [
                cast(int, row["time_to_threshold_step"]) for row in successful_rows
            ]
            transition_times = [
                cast(float, row["time_to_threshold_s"]) for row in successful_rows
            ]
            writer.writerow(
                {
                    "training_fraction": training_fraction,
                    "optimizer": optimizer_name,
                    "max_steps": max(cast(int, row["max_steps"]) for row in rows),
                    "runs": len(rows),
                    "successes": len(successful_rows),
                    "success_rate": len(successful_rows) / len(rows),
                    "best_val_acc": max(best_accuracies),
                    "mean_best_val_acc": statistics.fmean(best_accuracies),
                    "median_best_val_acc": statistics.median(best_accuracies),
                    "mean_steps_to_transition": (
                        statistics.fmean(transition_steps) if transition_steps else ""
                    ),
                    "median_steps_to_transition": (
                        statistics.median(transition_steps) if transition_steps else ""
                    ),
                    "min_steps_to_transition": (
                        min(transition_steps) if transition_steps else ""
                    ),
                    "max_steps_to_transition": (
                        max(transition_steps) if transition_steps else ""
                    ),
                    "mean_time_to_transition_s": (
                        statistics.fmean(transition_times) if transition_times else ""
                    ),
                }
            )


def trace_row(
    run_id: str,
    seed: int,
    repeat: int,
    optimizer_name: str,
    training_fraction: float,
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
    temperature_ceiling: float | None,
    free_energy: float | None,
    best_idx: int | None,
    threshold: float,
    strict_threshold: bool,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "seed": seed,
        "repeat": repeat,
        "optimizer": optimizer_name,
        "training_fraction": training_fraction,
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
        "temperature_ceiling": temperature_ceiling,
        "free_energy": free_energy,
        "best_index": best_idx,
        "reached_threshold": (
            val_acc > threshold if strict_threshold else val_acc >= threshold
        ),
    }


if __name__ == "__main__":
    main()
