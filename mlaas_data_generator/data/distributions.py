import numpy as np

def get_data_distribution(
    y,
    num_classes: int | None,
    bins: int | None = None,
    value_range: tuple[float, float] | None = None,
):
    """Return the target distribution for a client dataset.

    For classification tasks ``num_classes`` should be provided and the return
    value is a mapping of class index to count. For regression tasks
    ``num_classes`` can be ``None`` and a histogram with ``bins`` buckets will
    be produced over ``value_range``.
    """

    if num_classes is None:
        if bins is None:
            bins = 10
        if value_range is not None:
            hist, _ = np.histogram(y, bins=bins, range=value_range)
        else:
            hist, _ = np.histogram(y, bins=bins)
        return {
            f"bin_{i}": int(hist[i])
            for i in range(len(hist))
        }

    distribution = {i: 0 for i in range(num_classes)}
    for label in y:
        distribution[int(label)] += 1
    return distribution


def _generate_regular_distribution(num_clients: int, start_client: int = 1, num_labels: int = 10, samples_per_label: int = 100):
    regular_distributions = {}
    for i in range(start_client, num_clients + 1):
        regular_distributions[f"client_{i}"] = {
            label: samples_per_label for label in range(num_labels)
        }
    return regular_distributions


def prepare_client_distributions(custom_distributions: dict | None, num_clients: int):
    """Validate and extend custom distributions to match num_clients.

    If fewer distributions are provided than num_clients, regular distributions
    are generated for the remaining clients. If more are provided, the extra
    distributions are discarded. A warning is printed in both cases.
    """
    if custom_distributions is None:
        return None

    custom_distributions = {
        client: {int(label): count for label, count in dist.items()}
        for client, dist in custom_distributions.items()
    }

    num_custom = len(custom_distributions)
    if num_custom != num_clients:
        print(
            f"Warning: Provided distributions for {num_custom} clients, "
            f"but {num_clients} clients expected."
        )
        if num_custom < num_clients:
            start = num_custom + 1
            regular = _generate_regular_distribution(num_clients, start)
            custom_distributions.update(regular)
        else:
            allowed = sorted(custom_distributions.keys())[:num_clients]
            custom_distributions = {k: custom_distributions[k] for k in allowed}

    return custom_distributions