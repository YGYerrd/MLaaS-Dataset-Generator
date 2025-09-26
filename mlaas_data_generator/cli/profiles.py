DATASET_CHOICES = [
    "fashion_mnist",
    "mnist",
    "cifar10",
    "digits",
    "iris",
    "wine",
    "california_housing",
]


def infer_dataset_profile(dataset: str) -> dict:
    ds = dataset.lower()
    if ds in {"mnist", "fashion_mnist", "cifar10"}:
        return {
            "task": "classification",
            "ask_split": False,            
            "split_key": None,             
            "ask_scaler": False,           
            "default_model": "CNN",
            "allow_label_splits": True,    
            "allow_quantity_skew": True,
            "allow_custom": True,          
            "ask_target_scaler": False,    
        }

    if ds in {"iris", "wine", "digits"}:
        return {
            "task": "classification",          
            "ask_split": True,             
            "split_key": "test_size",
            "ask_scaler": True,            
            "default_model": "MLP",
            "allow_label_splits": True,    
            "allow_quantity_skew": True,
            "allow_custom": True,
            "ask_target_scaler": False,
        }

    if ds == "california_housing":
        return {
            "task": "regression",
            "ask_split": True,             
            "split_key": "test_size",
            "ask_scaler": True,            
            "default_model": "MLP",
            "allow_label_splits": False,   
            "allow_quantity_skew": True,   
            "allow_custom": False,         
            "ask_target_scaler": True,     
        }

    return {
        "task": "classification",
        "ask_split": False,
        "split_key": None,
        "ask_scaler": False,
        "default_model": "MLP",
        "allow_label_splits": True,
        "allow_quantity_skew": True,
        "allow_custom": True,
        "ask_target_scaler": False,
    }