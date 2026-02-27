import numpy as np

def preprocess_hf_text_sequence(train, test, meta, *, hf_model_id, text_column="text", label_column="label"):
    ds_train, _ = train
    ds_test, _ = test

    cols = set(ds_train.column_names)
    if text_column not in cols:
        raise ValueError(f"Missing text_column '{text_column}' in dataset '{meta.get('hf_id')}'")
    if label_column not in cols:
        raise ValueError(f"Missing label_column '{label_column}' in dataset '{meta.get('hf_id')}'")

    try:
        from datasets import ClassLabel
    except Exception:
        ClassLabel = None

    label_feat = ds_train.features.get(label_column)
    if ClassLabel is not None and isinstance(label_feat, ClassLabel):
        num_classes = int(label_feat.num_classes)
        y_train = np.asarray(ds_train[label_column], dtype="int32")
        y_test = np.asarray(ds_test[label_column], dtype="int32")
        label_mapping = {str(i): i for i in range(num_classes)}
    else:
        uniq = {}
        y_train_list = []
        for v in ds_train[label_column]:
            if v not in uniq:
                uniq[v] = len(uniq)
            y_train_list.append(uniq[v])

        y_test_list = []
        for v in ds_test[label_column]:
            if v not in uniq:
                raise ValueError(f"Unseen label in test split: {v!r}")
            y_test_list.append(uniq[v])

        y_train = np.asarray(y_train_list, dtype="int32")
        y_test = np.asarray(y_test_list, dtype="int32")
        num_classes = int(len(uniq))
        label_mapping = {str(k): int(v) for k, v in uniq.items()}

    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise ImportError(
            "HF text preprocessing requires 'transformers'. Install with: pip install transformers"
        ) from e

    max_length = int(meta.get("max_length", 128))
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)

    texts_train = list(ds_train[text_column])
    texts_test = list(ds_test[text_column])

    enc_train = tokenizer(
        texts_train,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np",
    )
    enc_test = tokenizer(
        texts_test,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np",
    )

    X_train = {
        "input_ids": enc_train["input_ids"].astype("int32"),
        "attention_mask": enc_train["attention_mask"].astype("int32"),
    }
    X_test = {
        "input_ids": enc_test["input_ids"].astype("int32"),
        "attention_mask": enc_test["attention_mask"].astype("int32"),
    }

    meta2 = dict(meta)
    meta2.update({
        "input_shape": (max_length,),
        "num_classes": num_classes,
        "label_mapping": label_mapping,
        "text_column": text_column,
        "label_column": label_column,
        "hf_model_id": hf_model_id,
        "x_format": "dict",
        "x_keys": ["input_ids", "attention_mask"],
        "label_granularity": "sequence",
        "hf_task": "sequence_classification",
        "modality": "text",
    })

    return (X_train, y_train), (X_test, y_test), meta2