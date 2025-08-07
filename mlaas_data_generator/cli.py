"""Command line interface for the MLaaS data generator."""

from __future__ import annotations

import argparse

from .config import CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MLaaS client data")
    parser.add_argument("--clients", type=int, default=CONFIG["num_clients"], help="number of clients")
    parser.add_argument(
        "--output", type=str, default="generated_clients.csv", help="output CSV file"
    )
    args = parser.parse_args()

    # Import heavy modules lazily so that `--help` works without TensorFlow installed
    from .federated import FederatedDataGenerator

    config = CONFIG.copy()
    config["num_clients"] = args.clients

    generator = FederatedDataGenerator(config)
    df = generator.run()
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} client records to {args.output}")


if __name__ == "__main__":
    main()