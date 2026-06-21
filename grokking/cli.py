from argparse import ArgumentParser, Namespace, SUPPRESS

from data import ALL_OPERATIONS
from training import main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--operation", type=str, choices=ALL_OPERATIONS.keys(), default="x/y"
    )
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "ensemble-simple", "ensemble"],
        default="adamw",
    )
    parser.add_argument(
        "--otimizer",
        dest="optimizer",
        type=str,
        choices=["adamw", "ensemble-simple", "ensemble"],
        help=SUPPRESS,
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
    args: Namespace = parser.parse_args()

    main(args)
