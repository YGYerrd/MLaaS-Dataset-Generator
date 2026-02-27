def load_huggingface_source(**kwargs):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError(
            "Hugging Face dataset loading requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from e

    dataset_name = kwargs.get("dataset_name")
    if not dataset_name:
        raise ValueError("HF source requires dataset_name=<repo_id> in dataset_args.")

    dataset_config = kwargs.get("dataset_config", None)
    train_split = kwargs.get("train_split", "train")
    requested_test_split = kwargs.get("test_split", "test")

    max_samples = kwargs.get("max_samples", None)
    seed = int(kwargs.get("seed", 42))
    max_length = int(kwargs.get("max_length", 128))

    task_type = kwargs.get("task", "classification")
    modality = kwargs.get("modality", "text")
    hf_task = kwargs.get("hf_task", "sequence_classification")

    ds_train = load_dataset(dataset_name, dataset_config, split=train_split)

    def _try_load_split(split_name):
        return load_dataset(dataset_name, dataset_config, split=split_name)

    def _is_unlabelled(ds):
        label_column = kwargs.get("label_column", "label")
        try:
            ys = ds[label_column]
        except Exception:
            return True
        if ys is None or len(ys) == 0:
            return True
        try:
            return all(int(v) == -1 for v in ys)
        except Exception:
            return False

    ds_test = None
    chosen_test_split = None
    for candidate in [requested_test_split, "validation", "val", "dev"]:
        try:
            tmp = _try_load_split(candidate)
        except Exception:
            continue
        if _is_unlabelled(tmp):
            continue
        ds_test = tmp
        chosen_test_split = candidate
        break

    if ds_test is None:
        test_size = float(kwargs.get("test_size", 0.2))
        split = ds_train.train_test_split(test_size=test_size, seed=seed, shuffle=True)
        ds_train = split["train"]
        ds_test = split["test"]
        chosen_test_split = "train_test_split"

    if max_samples:
        n = int(max_samples)
        ds_train = ds_train.select(range(min(n, len(ds_train))))
        ds_test = ds_test.select(range(min(max(1, n // 5), len(ds_test))))

    dataset_args = dict(kwargs)
    dataset_args.pop("preprocessors", None)

    meta = {
        "dataset_family": "hf",
        "hf_id": dataset_name,
        "hf_subset": dataset_config,
        "train_split": train_split,
        "test_split": chosen_test_split,
        "seed": seed,
        "max_length": max_length,
        "task_type": task_type,
        "modality": modality,
        "hf_task": hf_task,
        "dataset_args": dataset_args,
    }

    return (ds_train, None), (ds_test, None), meta