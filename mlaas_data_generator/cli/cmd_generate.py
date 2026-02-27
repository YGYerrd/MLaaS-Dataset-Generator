# cli/cmd_generate.py
from __future__ import annotations
import json
import argparse
from ..config import CONFIG
from ..path_resolver import resolve_output_path
from ..federated.orchestrator import FederatedDataGenerator
from .utils import parse_dataset_args, resolve_hidden_layers
from .profiles import DATASET_CHOICES

def _handle(args: argparse.Namespace) -> None:
    config = CONFIG.copy()
    config["num_clients"] = args.clients
    config["distribution_type"] = args.strategy
    config["dataset"] = args.dataset
    config["model_type"] = args.model_type or config.get("model_type", "CNN")

    ds_args = parse_dataset_args(args.dataset_arg)
    if ds_args: config["dataset_args"] = ds_args
    if args.distribution_bins is not None: config["distribution_bins"] = int(args.distribution_bins)
    if args.rounds is not None:      config["num_rounds"] = int(args.rounds)
    if args.batch_size is not None:  config["batch_size"] = int(args.batch_size)
    if args.learning_rate is not None: config["learning_rate"] = float(args.learning_rate)
    if args.epochs_per_round is not None: config["local_epochs"] = int(args.epochs_per_round)
    if args.sample_size is not None: config["sample_size"] = int(args.sample_size)
    if args.sample_frac is not None: config["sample_frac"] = float(args.sample_frac)
    if args.hidden_layers:           config["hidden_layers"] = resolve_hidden_layers(args.hidden_layers, config.get("hidden_layers",[128]))
    if args.activation is not None:  config["activation"] = args.activation
    if args.dropout is not None:     config["dropout"] = float(args.dropout)
    if args.weight_decay is not None: config["weight_decay"] = float(args.weight_decay)
    if args.optimizer is not None:   config["optimizer"] = args.optimizer
    if args.client_dropout_rate is not None: config["client_dropout_rate"] = float(args.client_dropout_rate)
    if args.model_type is not None:  config["model_type"] = args.model_type
    if args.task_type is not None:   config["task_type"] = args.task_type
    if args.seed is not None:        config["seed"] = int(args.seed)
    if args.no_save_weights:         config["save_weights"] = False

    if args.distribution_param is not None:
        if args.strategy in {"shard", "label_per_client"}:
            config["distribution_param"] = int(args.distribution_param)
        else:
            config["distribution_param"] = float(args.distribution_param)

    if args.strategy == "custom":
        if not args.distribution:
            raise SystemExit("--distribution is required when --strategy custom")
        with open(args.distribution, "r") as f:
            config["custom_distributions"] = json.load(f)

    gen = FederatedDataGenerator(
        config=config,
        dataset=args.dataset,
        task_type=config.get("task_type"),
        model_type=config.get("model_type"),
    )
    df = gen.run()
    out_path = resolve_output_path(args.output, kind="run")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} client records to {out_path}")

def register_generate(subparsers):
    p = subparsers.add_parser("generate", help="Run federated training and write results")
    p.add_argument("--rounds", type=int, default=None, help="Number of global rounds")
    p.add_argument("--clients", type=int, default=CONFIG["num_clients"], help="Number of clients")
    p.add_argument("--client-dropout-rate", type=float, default=None, help="Per-round dropout prob [0–1]")
    p.add_argument("--model-type", type=str, default=None, choices=["CNN","MLP","LogReg","ResNet","Custom", "hf", "hf_finetune"], help="Model family")
    p.add_argument("--task-type", type=str, default=None, choices=["classification","regression","clustering"], help="Task type")
    p.add_argument("--output", type=str, default="clients.csv", help="Output CSV")
    p.add_argument("--dataset", type=str, default="mnist", choices=DATASET_CHOICES, help="Dataset")
    p.add_argument("--dataset-arg", action="append", default=[], metavar="KEY=VALUE", help="Dataset loader args")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--strategy", type=str, default="iid",
                   choices=["iid","quantity_skew","dirichlet","shard","label_per_client","custom"],
                   help="Split strategy")
    p.add_argument("--no-save-weights", action="store_true", help="Disable weight JSONs")
    p.add_argument("--distribution-param", type=float, default=None, help="Param for split strategy")
    p.add_argument("--sample-size", type=int, default=None, help="Pre-split sample size")
    p.add_argument("--sample-frac", type=float, default=None, help="Pre-split sample frac")
    p.add_argument("--distribution", type=str, default=None, help="JSON path for custom strategy")
    p.add_argument("--distribution-bins", type=int, default=None, help="Bins for regression target summary")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size")
    p.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    p.add_argument("--epochs-per-round", type=int, default=None, help="Local epochs per round")
    p.add_argument("--hidden-layers", type=str, default=None, help="Comma-separated hidden sizes, e.g. '128,64'")
    p.add_argument("--activation", type=str, default=None,
                   choices=["relu","tanh","sigmoid","gelu","elu","selu","softplus","linear"])
    p.add_argument("--dropout", type=float, default=None, help="Dropout prob [0–0.9]")
    p.add_argument("--weight-decay", type=float, default=None, help="L2 coeff")
    p.add_argument("--optimizer", type=str, default=None, choices=["adam","adamw","sgd","rmsprop","adagrad"])
    p.set_defaults(_handler=_handle)
