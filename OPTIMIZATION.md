# Ensemble Optimization Methods

This document records the two ensemble-based grokking optimization setups used
in this repository. The intent is to preserve enough implementation detail to
support a scientific publication methods section and to make the experimental
variants reproducible.

## Baseline Context

The baseline grokking loop trains a single Transformer on modular arithmetic
data. For each optimization step:

1. A minibatch is drawn from the full training tensor.
2. The model predicts the final sequence position.
3. Cross-entropy loss is backpropagated.
4. AdamW performs the parameter update.
5. A short linear learning-rate warmup scheduler is advanced.
6. Training and validation loss and accuracy are logged periodically.

The ensemble methods below keep the same model, data, objective, AdamW update,
and scheduler. They modify only the optimizer-level trajectory management after
the ordinary gradient update.

## Shared Ensemble Machinery

Both ensemble setups instantiate `K = --ensemble_size` independent Transformer
trajectories.

Initialization:

1. A single base Transformer is initialized.
2. Its state dict is cloned.
3. Each ensemble member loads the same base weights.
4. Each member receives an independent Gaussian parameter perturbation with
   scale `--init_perturb_scale`.
5. Each member has its own AdamW optimizer and scheduler.

Per training step:

1. Each ensemble member keeps its own shuffled permutation over the training
   examples.
2. Each member consumes one minibatch from its own permutation.
3. Each member performs the normal AdamW training step.
4. After the AdamW step, a random parameter force is added:

   ```text
   theta <- theta + force_scale * sqrt(2 * lr * T) * epsilon
   epsilon ~ Normal(0, I)
   ```

   where `T` is the current ensemble temperature and `lr` is the member's
   current optimizer learning rate.

The random-force term is a Langevin-style perturbation. It is intended to help
individual trajectories cross local barriers in the empirical loss landscape
while preserving the ordinary supervised learning objective.

## Free-Energy Score

Both setups use a free-energy-like score for selecting a trajectory:

```text
F_i = L_val,i - alpha * T * log(S_i + eps)
```

where:

- `L_val,i` is the validation cross-entropy loss of trajectory `i`.
- `alpha = --entropy_weight`.
- `T` is the current ensemble temperature.
- `S_i` is the mean squared distance from trajectory `i` to the other
  ensemble trajectories in flattened parameter space.
- `eps = 1e-12` avoids `log(0)`.

The score prefers low validation loss while giving a small reward to trajectories
that represent distinct parameter-space regions. This is a pragmatic proxy for
the energy-entropy tradeoff; it is not claimed to be a thermodynamic free energy
unless separately justified.

## Setup 1: Intermediate Adaptive Resampling

This was the first integration of the barrier-crossing idea into the grokking
loop.

### Objective

The goal was to combine the original grokking training process with an ensemble
of noisy trajectories. The ensemble was periodically evaluated and weak
trajectories were resampled around the best current trajectory.

### Technical Steps

1. Added an optional optimizer mode selected with:

   ```bash
   --optimizer ensemble
   ```

2. Added ensemble hyperparameters:

   ```text
   --ensemble_size
   --temperature
   --min_temperature
   --max_temperature
   --temperature_heating
   --temperature_cooling
   --time_cooling_rate
   --stall_window
   --resample_fraction
   --init_perturb_scale
   --resample_perturb_scale
   --force_scale
   --selection_metric
   --entropy_weight
   ```

3. Kept `K` separate models, optimizers, schedulers, minibatch permutations,
   and batch cursors.

4. Applied the Langevin-style random parameter force after every AdamW update:

   ```text
   theta_i <- theta_i + force_scale * sqrt(2 * lr_i * T) * epsilon_i
   ```

5. Applied time-dependent exponential decay to the maximum temperature after
   every training step:

   ```text
   T_max,time <- max(
       min_temperature,
       T_max,time * exp(-time_cooling_rate)
   )
   T <- min(T, T_max,time)
   ```

   The default `--time_cooling_rate 1e-4` lowers the available heating range
   continuously in optimization time. It does not reduce the current
   temperature while that temperature is already below the moving ceiling.
   A value of `0` disables this mechanism.

6. Evaluated all trajectories on the full train and validation sets at logging
   points.

7. Computed a selection score per trajectory. The supported modes were:

   ```text
   validation_loss
   training_loss
   free_energy
   ```

8. Selected the trajectory with the minimum score.

9. Adapted temperature according to score progress:

   - If the selected score improved over the best previously observed score,
     temperature was cooled:

     ```text
     T <- max(min_temperature, T * temperature_cooling)
     ```

   - If no improvement was observed for `--stall_window` evaluation events,
     temperature was heated:

     ```text
     T_ceiling(L_val) = clamp(
         max_temperature * min(max(L_val, 0), 1),
         min_temperature,
         max_temperature
     )
     T <- min(T_ceiling(L_val), T * temperature_heating)
     ```

   At evaluation, the effective ceiling is the smaller of the time-dependent
   and loss-dependent ceilings. Validation losses at or above `1` leave the
   time-dependent ceiling unchanged. Below `1`, the loss ceiling decreases
   linearly with the selected trajectory's current validation loss. Neither
   ceiling falls below `--min_temperature`.

10. Resampled the weakest fraction of trajectories:

   ```text
   n_resample = floor(resample_fraction * K)
   ```

   The worst-scoring trajectories were replaced by copies of the current best
   trajectory plus Gaussian parameter noise:

   ```text
   theta_j <- theta_best + resample_perturb_scale * sqrt(max(T, min_temperature)) * epsilon_j
   ```

11. Cleared the AdamW moment state for each copied trajectory so stale optimizer
    statistics did not carry over to the new location.

12. Logged aggregate best-trajectory metrics and per-particle train loss,
    train accuracy, validation loss, validation accuracy, spread, and score.
    The aggregate metrics include both `ensemble_simple/temperature` and
    `ensemble_simple/temperature_ceiling`.

### Interpretation

This setup treated the ensemble as a population of candidate learning
trajectories. The heating rule increased exploration when progress stalled, and
resampling concentrated compute near currently promising solutions. It was
useful as an initial translation of barrier crossing, but it introduced several
coupled mechanisms: stall detection, heating, cooling, partial resampling, and
choice of selection metric. Those couplings made experimental interpretation
harder.

## Setup 2: Final Probe-Collapse Method

The current method simplifies the ensemble dynamics and makes the control flow
more publication-friendly.

### Objective

The goal is to decouple probing from collapse and make the ensemble collapse
rule deterministic: after a fixed number of probes, collapse all trajectories
around the least-free-energy state.

### Technical Steps

1. Retained the same `--optimizer ensemble` mode, shared initialization, AdamW
   update, and random force.

2. Replaced adaptive stall-based probing with fixed-interval probing:

   ```text
   probe every M training steps
   M = --probe_interval
   default M = 10
   ```

3. Maintained a probe counter:

   ```text
   probes_since_collapse += 1
   ```

4. At each probe, evaluated every trajectory on the full train and validation
   tensors.

5. Computed the free-energy score for every trajectory:

   ```text
   F_i = L_val,i - entropy_weight * T * log(S_i + 1e-12)
   ```

6. Selected the collapse center as:

   ```text
   center = argmin_i F_i
   ```

   This rule is unconditional: the least-free-energy trajectory is always the
   center at the time of collapse.

7. Updated temperature from the current best validation loss:

   ```text
   T = clamp(temperature * L_val,center, min_temperature, max_temperature)
   ```

   Thus lower current loss implies lower temperature, but exploration never
   fully vanishes because:

   ```text
   min_temperature > 0
   ```

8. Collapsed the ensemble immediately after `N` probes:

   ```text
   N = --collapse_probes
   default N = 10
   ```

9. During collapse, copied the center trajectory to every non-center trajectory:

   ```text
   theta_j <- theta_center + collapse_perturb_scale * sqrt(max(T, min_temperature)) * epsilon_j
   ```

10. Left the center trajectory unchanged.

11. Cleared AdamW moment state for copied trajectories.

12. Reset the probe counter after collapse:

    ```text
    probes_since_collapse = 0
    ```

13. Logged:

    ```text
    training/loss
    training/accuracy
    validation/loss
    validation/accuracy
    ensemble/best_index
    ensemble/best_free_energy
    ensemble/temperature
    ensemble/collapsed
    ensemble/probes_since_collapse
    ensemble/particle_i/training_loss
    ensemble/particle_i/training_accuracy
    ensemble/particle_i/validation_loss
    ensemble/particle_i/validation_accuracy
    ensemble/particle_i/spread
    ensemble/particle_i/free_energy
    ```

### Current CLI Parameters

The current final method exposes:

```text
--optimizer ensemble
--ensemble_size
--temperature
--min_temperature
--max_temperature
--probe_interval
--collapse_probes
--init_perturb_scale
--collapse_perturb_scale
--force_scale
--entropy_weight
```

Recommended starting command:

```bash
uv run python grokking/cli.py \
  --optimizer ensemble \
  --ensemble_size 4 \
  --probe_interval 10 \
  --collapse_probes 10
```

For the harder task:

```bash
uv run python grokking/cli.py \
  --operation permuted_quadratic \
  --optimizer ensemble \
  --ensemble_size 4 \
  --training_fraction 0.3 \
  --probe_interval 10 \
  --collapse_probes 10
```

### Interpretation

The final method separates three roles:

1. Continuous exploration through parameter noise at every training step.
2. Measurement through fixed-interval ensemble probing.
3. Selection through deterministic collapse after a fixed number of probes.

This makes the method easier to ablate. For publication, useful ablations are:

1. Single-model AdamW baseline.
2. Ensemble without random force.
3. Ensemble with random force but no collapse.
4. Ensemble with fixed probe-collapse and no entropy term.
5. Ensemble with fixed probe-collapse and free-energy selection.
6. Sensitivity to `--probe_interval`, `--collapse_probes`, `--temperature`,
   `--min_temperature`, and `--entropy_weight`.

## Harder Generalization Task

A harder task, `permuted_quadratic`, was added to stress generalization:

```text
q(x, y) = x^2 + xy + 3y^2 + 5x + 7y mod (p - 1)
label = g^q mod p
```

where `g` is a primitive root modulo the prime `p`.

The polynomial is compact in exponent space, but the primitive-root map
scrambles local label smoothness in token space. This makes the task more
difficult than modular addition, subtraction, or division under random train
and validation splits.

## Publication Notes

The free-energy score and Langevin-style random force should be described as
optimization heuristics inspired by barrier crossing in rugged landscapes. The
implementation does not sample from a calibrated posterior or known stationary
distribution. Claims should therefore be empirical unless additional theory is
developed.

For reproducibility, report:

1. Prime `p`.
2. Operation/task.
3. Training fraction and split seed, if controlled.
4. Model depth, width, and attention heads.
5. Batch size.
6. AdamW learning rate and weight decay.
7. Number of optimization steps.
8. Ensemble size.
9. Temperature scale, minimum temperature, and maximum temperature.
10. Probe interval and collapse-probe count.
11. Perturbation scales and force scale.
12. Entropy weight.
13. Number of independent random seeds.

## Statistics Loop

The repository includes `grokking/stats.py` to prepare aggregate time-to-target
data for publication. The default run is intentionally large:

```bash
uv run python grokking/stats.py
```

Default benchmark design:

1. `20` seed indices.
2. `3` repeats per seed index.
3. All optimizer routes:
   - `adamw`
   - `ensemble-simple`
   - `ensemble`
4. Validation target:
   - `90%` validation accuracy.

The script writes two CSV files:

```text
stats_runs/time_step_trace.csv
stats_runs/time_to_threshold_summary.csv
```

`time_step_trace.csv` stores the per-evaluation time-step data:

```text
run_id
seed
repeat
optimizer
step
wall_time_s
model_updates
probe_count
collapse_count
train_loss
train_acc
val_loss
val_acc
temperature
free_energy
best_index
reached_threshold
```

`time_to_threshold_summary.csv` stores the first observed time to 90%
validation accuracy:

```text
time_to_threshold_step
time_to_threshold_s
model_updates_to_threshold
probe_count_to_threshold
collapse_count_to_threshold
final_val_acc
final_val_loss
```

The `model_updates` column is the approximate computation-step proxy. For the
single-model baseline, one training step equals one model update. For ensemble
methods, one training step equals `ensemble_size` model updates. This does not
fully price full-dataset evaluation passes, but it gives a simple compute proxy
that is stable across runs and easy to report alongside wall-clock time.
