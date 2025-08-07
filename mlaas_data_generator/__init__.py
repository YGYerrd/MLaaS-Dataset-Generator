"""MLaaS data generator package."""

# The FederatedDataGenerator is intentionally imported lazily to keep optional
# dependencies light for consumers that only need metadata (e.g. ``--help``).

__all__ = ["FederatedDataGenerator"]


def __getattr__(name):
    if name == "FederatedDataGenerator":
        from .federated import FederatedDataGenerator
        return FederatedDataGenerator
    raise AttributeError(name)