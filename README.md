# Grokking

An implementation of the OpenAI 'Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets' paper in PyTorch.

<img src="figures/Figure_1_left_accuracy.png" height="200"> <img src="figures/Figure_1_left_loss.png" height="200">

## Installation

* Clone the repo and cd into it:
    ```bash
    git clone https://github.com/danielmamay/grokking.git
    cd grokking
    ```
* Install dependencies:
    ```bash
    uv sync
    ```

## Usage

The project uses [Weights & Biases](https://wandb.ai/site) to keep track of experiments. Run `uv run wandb login` to use the online dashboard, or `uv run wandb offline` to store the data on your local machine.

* To run a single experiment using the [CLI](grokking/cli.py):
    ```bash
    uv run python grokking/cli.py
    ```

* To run the ensemble temperature loop:
    ```bash
    uv run python grokking/cli.py --optimizer ensemble --ensemble_size 4
    ```
    This keeps several nearby Transformer trajectories, applies a
    loss-scaled random parameter force after each AdamW step, probes the
    ensemble every `--probe_interval` steps, and collapses all trajectories
    around the least-free-energy state every `--collapse_probes` probes.

* To run the earlier adaptive resampling ensemble:
    ```bash
    uv run python grokking/cli.py --optimizer ensemble-simple --ensemble_size 4
    ```
    `ensemble-simple` keeps the intermediate method available for comparison:
    it heats after stalled free-energy progress, cools after improvement, and
    resamples weak trajectories around the current best trajectory.

* To collect time-to-90% validation statistics:
    ```bash
    uv run python grokking/stats.py
    ```
    By default this runs 20 seeds, 3 repeats, and all optimizers
    (`adamw`, `ensemble-simple`, `ensemble`). It writes
    `stats_runs/time_step_trace.csv` and
    `stats_runs/time_to_threshold_summary.csv`.

* To run a harder generalization task:
    ```bash
    uv run python grokking/cli.py --operation permuted_quadratic --training_fraction 0.3
    ```
    `permuted_quadratic` maps a quadratic expression through a primitive-root
    permutation, so the underlying rule is compact but nearby labels look much
    less smooth than in the default modular division task.

* To run a grid search using W&B Sweeps:
    ```bash
    uv run wandb sweep sweep.yaml
    uv run wandb agent {entity}/grokking/{sweep_id}
    ```

## Development

* Type checking:
    ```bash
    uv run pyright grokking/
    ```
* Linting:
    ```bash
    uv run ruff check grokking/
    ```


## References

Code:

* [openai/grok](https://github.com/openai/grok)

Paper:

* [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)
