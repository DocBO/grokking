from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Sized
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data import get_data_loaders
from model import Transformer


def main(args: Namespace) -> None:
    wandb.init(project="grokking", config=vars(args))
    assert wandb.run is not None
    config = wandb.config
    device = get_device(config.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    train_inputs, train_labels, val_inputs, val_labels, batch_size = load_data(
        config.operation, config.prime, config.training_fraction, config.batch_size, device
    )

    optimizer_name = getattr(config, "optimizer", "adamw")
    if optimizer_name == "ensemble":
        ensemble_main(
            config,
            train_inputs,
            train_labels,
            val_inputs,
            val_labels,
            batch_size,
            device,
        )
        return

    model, optimizer, scheduler, criterion = setup_model(
        config.num_layers, config.dim_model, config.num_heads, config.prime,
        config.learning_rate, config.weight_decay, device,
    )

    n_train = len(train_inputs)
    perm = torch.randperm(n_train, device=device)
    batch_idx = 0

    for step in tqdm(range(config.num_steps)):
        if batch_idx >= n_train:
            perm = torch.randperm(n_train, device=device)
            batch_idx = 0

        idx = perm[batch_idx : batch_idx + batch_size]
        batch_idx += batch_size

        train_step(model, train_inputs[idx], train_labels[idx], optimizer, criterion)
        scheduler.step()

        if step in (1, 10) or step % 100 == 0:
            train_loss, train_acc = evaluate(model, train_inputs, train_labels, criterion)
            val_loss, val_acc = evaluate(model, val_inputs, val_labels, criterion)
            wandb.log(
                {
                    "training/loss": train_loss,
                    "training/accuracy": train_acc,
                    "validation/loss": val_loss,
                    "validation/accuracy": val_acc,
                },
                step=step,
            )


@dataclass
class EnsembleConfig:
    n_trajectories: int
    temperature_scale: float
    min_temperature: float
    max_temperature: float
    probe_interval: int
    collapse_probes: int
    init_perturb_scale: float
    collapse_perturb_scale: float
    force_scale: float
    entropy_weight: float


@dataclass
class ParticleMetrics:
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    spread: float
    free_energy: float


def ensemble_main(
    config: Any,
    train_inputs: Tensor,
    train_labels: Tensor,
    val_inputs: Tensor,
    val_labels: Tensor,
    batch_size: int,
    device: torch.device,
) -> None:
    ensemble_cfg = get_ensemble_config(config)
    models, optimizers, schedulers, criterion = setup_ensemble(
        config.num_layers,
        config.dim_model,
        config.num_heads,
        config.prime,
        config.learning_rate,
        config.weight_decay,
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
    best_idx = 0

    for step in tqdm(range(config.num_steps)):
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
                learning_rate=current_learning_rate(optimizers[particle_idx]),
                temperature=temperature,
                force_scale=ensemble_cfg.force_scale,
            )

        if should_probe_ensemble(step, ensemble_cfg.probe_interval):
            particle_metrics = evaluate_ensemble(
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
                [metrics.free_energy for metrics in particle_metrics],
                device=device,
            )
            best_idx = int(torch.argmin(free_energies).item())
            best_loss = particle_metrics[best_idx].val_loss
            temperature = loss_scaled_temperature(best_loss, ensemble_cfg)
            probes_since_collapse += 1

            collapsed = 0
            if probes_since_collapse >= ensemble_cfg.collapse_probes:
                collapsed = collapse_ensemble(
                    models,
                    optimizers,
                    best_idx,
                    temperature,
                    ensemble_cfg,
                )
                probes_since_collapse = 0

            log_ensemble_metrics(
                particle_metrics,
                best_idx,
                temperature,
                collapsed,
                probes_since_collapse,
                step,
            )


def get_ensemble_config(config: Any) -> EnsembleConfig:
    ensemble_cfg = EnsembleConfig(
        n_trajectories=getattr(config, "ensemble_size", 4),
        temperature_scale=getattr(config, "temperature", 1e-5),
        min_temperature=getattr(config, "min_temperature", 1e-7),
        max_temperature=getattr(config, "max_temperature", 1e-2),
        probe_interval=getattr(config, "probe_interval", 10),
        collapse_probes=getattr(config, "collapse_probes", 10),
        init_perturb_scale=getattr(config, "init_perturb_scale", 1e-3),
        collapse_perturb_scale=getattr(config, "collapse_perturb_scale", 1e-3),
        force_scale=getattr(config, "force_scale", 1.0),
        entropy_weight=getattr(config, "entropy_weight", 0.01),
    )
    validate_ensemble_config(ensemble_cfg)
    return ensemble_cfg


def validate_ensemble_config(ensemble_cfg: EnsembleConfig) -> None:
    if ensemble_cfg.n_trajectories < 1:
        raise ValueError("ensemble_size must be at least 1")
    if ensemble_cfg.temperature_scale < 0:
        raise ValueError("temperature must be non-negative")
    if ensemble_cfg.min_temperature <= 0:
        raise ValueError("min_temperature must be positive")
    if ensemble_cfg.max_temperature < ensemble_cfg.min_temperature:
        raise ValueError("max_temperature must be at least min_temperature")
    if ensemble_cfg.probe_interval <= 0:
        raise ValueError("probe_interval must be positive")
    if ensemble_cfg.collapse_probes <= 0:
        raise ValueError("collapse_probes must be positive")


def setup_ensemble(
    num_layers: int,
    dim_model: int,
    num_heads: int,
    prime: int,
    learning_rate: float,
    weight_decay: float,
    ensemble_cfg: EnsembleConfig,
    device: torch.device,
) -> tuple[
    list[Transformer],
    list[Optimizer],
    list[torch.optim.lr_scheduler.LRScheduler],
    torch.nn.CrossEntropyLoss,
]:
    models: list[Transformer] = []
    optimizers: list[Optimizer] = []
    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []

    base_model, _, _, criterion = setup_model(
        num_layers,
        dim_model,
        num_heads,
        prime,
        learning_rate,
        weight_decay,
        device,
    )
    base_state = {
        key: value.detach().clone()
        for key, value in base_model.state_dict().items()
    }

    for _ in range(ensemble_cfg.n_trajectories):
        model, optimizer, scheduler, _ = setup_model(
            num_layers,
            dim_model,
            num_heads,
            prime,
            learning_rate,
            weight_decay,
            device,
        )
        model.load_state_dict(base_state)
        perturb_parameters(model, ensemble_cfg.init_perturb_scale)
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    return models, optimizers, schedulers, criterion


def get_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_data(
    operation: str,
    prime: int,
    training_fraction: float,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    train_loader, val_loader = get_data_loaders(operation, prime, training_fraction, batch_size)
    train_dataset, val_dataset = train_loader.dataset, val_loader.dataset
    assert isinstance(train_dataset, Sized) and isinstance(val_dataset, Sized)
    train_inputs, train_labels = (t.to(device) for t in next(iter(DataLoader(train_dataset, batch_size=len(train_dataset)))))
    val_inputs, val_labels = (t.to(device) for t in next(iter(DataLoader(val_dataset, batch_size=len(val_dataset)))))
    print(f"train_inputs.device: {train_inputs.device}  shape: {train_inputs.shape}")
    actual_batch_size = min(batch_size, len(train_inputs) // 2)
    return train_inputs, train_labels, val_inputs, val_labels, actual_batch_size


def setup_model(
    num_layers: int,
    dim_model: int,
    num_heads: int,
    prime: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[Transformer, Optimizer, torch.optim.lr_scheduler.LRScheduler, torch.nn.CrossEntropyLoss]:
    model = Transformer(
        num_layers=num_layers,
        dim_model=dim_model,
        num_heads=num_heads,
        num_tokens=prime + 2,
        seq_len=4,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=10
    )

    criterion = torch.nn.CrossEntropyLoss()

    return model, optimizer, scheduler, criterion


def train_step(
    model: Transformer,
    inputs: Tensor,
    labels: Tensor,
    optimizer: Optimizer,
    criterion: torch.nn.CrossEntropyLoss,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)[-1, :, :]
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()


def current_learning_rate(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def should_probe_ensemble(step: int, probe_interval: int) -> bool:
    if probe_interval <= 0:
        raise ValueError("probe_interval must be positive")
    return step == 1 or step % probe_interval == 0


def loss_scaled_temperature(loss: float, ensemble_cfg: EnsembleConfig) -> float:
    temperature = ensemble_cfg.temperature_scale * max(loss, 0.0)
    return min(
        ensemble_cfg.max_temperature,
        max(ensemble_cfg.min_temperature, temperature),
    )


def add_random_force(
    model: Transformer,
    learning_rate: float,
    temperature: float,
    force_scale: float,
) -> None:
    if temperature <= 0 or force_scale <= 0:
        return

    noise_scale = force_scale * (2.0 * learning_rate * temperature) ** 0.5
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param) * noise_scale)


def perturb_parameters(model: Transformer, scale: float) -> None:
    if scale <= 0:
        return

    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param) * scale)


def flatten_parameters(model: Transformer) -> Tensor:
    return torch.cat([param.detach().flatten() for param in model.parameters()])


def particle_spreads(models: list[Transformer]) -> list[float]:
    if len(models) == 1:
        return [0.0]

    vectors = torch.stack([flatten_parameters(model) for model in models])
    distances = torch.cdist(vectors, vectors).pow(2)
    return [
        float((distances[i].sum() / (len(models) - 1)).item())
        for i in range(len(models))
    ]


def evaluate_ensemble(
    models: list[Transformer],
    train_inputs: Tensor,
    train_labels: Tensor,
    val_inputs: Tensor,
    val_labels: Tensor,
    criterion: torch.nn.CrossEntropyLoss,
    ensemble_cfg: EnsembleConfig,
    temperature: float,
) -> list[ParticleMetrics]:
    spreads = particle_spreads(models)
    metrics = []

    for model, spread in zip(models, spreads):
        train_loss, train_acc = evaluate(model, train_inputs, train_labels, criterion)
        val_loss, val_acc = evaluate(model, val_inputs, val_labels, criterion)
        free_energy = val_loss - (
            ensemble_cfg.entropy_weight
            * temperature
            * torch.log(torch.tensor(spread + 1e-12)).item()
        )

        metrics.append(
            ParticleMetrics(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                spread=spread,
                free_energy=free_energy,
            )
        )

    return metrics


def reset_optimizer_state(optimizer: Optimizer) -> None:
    optimizer.state.clear()


def copy_trajectory(
    source: Transformer,
    target: Transformer,
    perturb_scale: float,
) -> None:
    target.load_state_dict(source.state_dict())
    perturb_parameters(target, perturb_scale)


def collapse_ensemble(
    models: list[Transformer],
    optimizers: list[Optimizer],
    center_idx: int,
    temperature: float,
    ensemble_cfg: EnsembleConfig,
) -> int:
    if len(models) <= 1:
        return 0

    perturb_scale = (
        ensemble_cfg.collapse_perturb_scale
        * max(temperature, ensemble_cfg.min_temperature) ** 0.5
    )
    collapsed = 0

    for idx in range(len(models)):
        if idx == center_idx:
            continue
        copy_trajectory(models[center_idx], models[idx], perturb_scale)
        reset_optimizer_state(optimizers[idx])
        collapsed += 1

    return collapsed


def log_ensemble_metrics(
    particle_metrics: list[ParticleMetrics],
    best_idx: int,
    temperature: float,
    collapsed: int,
    probes_since_collapse: int,
    step: int,
) -> None:
    best = particle_metrics[best_idx]
    payload = {
        "training/loss": best.train_loss,
        "training/accuracy": best.train_acc,
        "validation/loss": best.val_loss,
        "validation/accuracy": best.val_acc,
        "ensemble/best_index": best_idx,
        "ensemble/best_free_energy": best.free_energy,
        "ensemble/temperature": temperature,
        "ensemble/collapsed": collapsed,
        "ensemble/probes_since_collapse": probes_since_collapse,
    }

    for idx, metrics in enumerate(particle_metrics):
        payload[f"ensemble/particle_{idx}/training_loss"] = metrics.train_loss
        payload[f"ensemble/particle_{idx}/training_accuracy"] = metrics.train_acc
        payload[f"ensemble/particle_{idx}/validation_loss"] = metrics.val_loss
        payload[f"ensemble/particle_{idx}/validation_accuracy"] = metrics.val_acc
        payload[f"ensemble/particle_{idx}/spread"] = metrics.spread
        payload[f"ensemble/particle_{idx}/free_energy"] = metrics.free_energy

    wandb.log(payload, step=step)


def evaluate(
    model: Transformer,
    val_inputs: Tensor,
    val_labels: Tensor,
    criterion: torch.nn.CrossEntropyLoss,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        output = model(val_inputs)[-1, :, :]
        loss = criterion(output, val_labels).item()
        acc = (torch.argmax(output, dim=1) == val_labels).float().mean().item()
    return loss, acc
