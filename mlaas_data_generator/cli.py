"""Command line interface for the MLaaS data generator."""

from __future__ import annotations

import argparse

from .config import CONFIG
from .data_utils import prepare_client_distributions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MLaaS client data")
    parser.add_argument(
        "--clients", 
        type=int, 
        default=CONFIG["num_clients"], 
        help="Number of clients"
        )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="generated_clients.csv", 
        help="Output CSV file"
    )

    parser.add_argument(
        "--dataset", 
        type=str,
        default= "mnist",
        choices=["fashion_mnist", "mnist"],
        help="Dataset to use"
    )

    parser.add_argument(
        "--distribution",
        type=str,
        default=None,
        help="Path to JSON file with custom client data distribution"
    )

    args = parser.parse_args()

    from .federated import FederatedDataGenerator
    import json

    config = CONFIG.copy()
    config["num_clients"] = args.clients
    
    client_distributions = None
    if args.distribution:
        with open(args.distribution, "r") as f:
            custom_distributions = json.load(f)
            client_distributions = prepare_client_distributions(custom_distributions, config["num_clients"])

    generator = FederatedDataGenerator(config, dataset=args.dataset, client_distributions=client_distributions)
    df = generator.run()
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} client records to {args.output}")


if __name__ == "__main__":
    main()