import math
from typing import Callable, List, Dict, Any

DEFAULT_TOTAL_LAYERS = 23  # Standard YOLOv8m layer count (indices 0..22)


def get_freeze_layers(strategy: str, total_layers: int = DEFAULT_TOTAL_LAYERS) -> List[int]:
    """
    Returns a list of layer indices to freeze initially based on strategy.

    Args:
        strategy (str): 'full_finetune', 'frozen_backbone', or 'gradual_unfreeze'.
        total_layers (int): Total number of sequential layers in the network.

    Returns:
        List[int]: Indices of layers to freeze initially.
    """
    if strategy == "full_finetune":
        return []
    elif strategy == "frozen_backbone":
        # Freeze all but the last 3 layers (backbone stays frozen, neck+head trainable)
        return list(range(0, max(0, total_layers - 3)))
    elif strategy == "gradual_unfreeze":
        # Never freeze the detection head; freeze only up to (not including) the last 2 layers
        return list(range(0, max(0, total_layers - 2)))
    else:
        print(f"[Warning] Unknown finetune strategy '{strategy}'. Defaulting to full_finetune (no freeze).")
        return []


def _get_layer_index(name: str) -> int:
    """Extract layer index from parameter name string (e.g. 'model.10.conv.weight' -> 10)."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "model" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
        elif part.isdigit():
            return int(part)
    return -1


def make_unfreeze_callback(unfreeze_schedule: List[int], total_layers: int = DEFAULT_TOTAL_LAYERS) -> Callable:
    """
    Creates an Ultralytics 'on_train_epoch_start' callback that progressively unfreezes
    network layers starting from deepest/last layers moving toward input layers.

    Args:
        unfreeze_schedule (List[int]): Milestone epoch numbers for unfreezing stages (e.g. [10, 20, 30]).
        total_layers (int): Total number of layers in the model architecture.

    Returns:
        Callable: Ultralytics callback function.
    """
    num_milestones = len(unfreeze_schedule)
    chunk_size = math.ceil(total_layers / max(1, num_milestones))

    def on_train_epoch_start(trainer):
        epoch = getattr(trainer, "epoch", 0)
        if epoch in unfreeze_schedule:
            stage_idx = unfreeze_schedule.index(epoch)
            high_layer = total_layers - stage_idx * chunk_size - 1
            low_layer = max(0, total_layers - (stage_idx + 1) * chunk_size)
            unfrozen_layer_indices = list(range(low_layer, high_layer + 1))

            unfrozen_param_names = []
            model = getattr(trainer, "model", None)
            if model is not None and hasattr(model, "named_parameters"):
                for name, param in model.named_parameters():
                    idx = _get_layer_index(name)
                    if idx in unfrozen_layer_indices:
                        param.requires_grad = True
                        unfrozen_param_names.append(name)

            print(
                f"[Unfreeze Schedule] Epoch {epoch}: unfreezing layer group {stage_idx + 1}/{num_milestones} "
                f"(layers {low_layer}..{high_layer}), now trainable: {unfrozen_param_names}"
            )

    return on_train_epoch_start


def apply_finetune_strategy(model: Any, config: Dict[str, Any], total_layers: int = DEFAULT_TOTAL_LAYERS) -> List[int]:
    """
    Configures finetune strategy and callbacks on model.

    Args:
        model: Ultralytics YOLO or YOLOv8WithCIFR instance.
        config (dict): Loaded experiment config dict.
        total_layers (int): Total layer count for architecture.

    Returns:
        List[int]: Layer indices to freeze initially (passed as 'freeze' arg to train).
    """
    train_cfg = config.get("train", {})
    strategy = train_cfg.get("finetune_strategy", "full_finetune")
    unfreeze_schedule = train_cfg.get("unfreeze_schedule", [])

    freeze_layers = get_freeze_layers(strategy, total_layers=total_layers)

    if strategy == "gradual_unfreeze" and unfreeze_schedule:
        callback = make_unfreeze_callback(unfreeze_schedule, total_layers=total_layers)
        if hasattr(model, "add_callback"):
            model.add_callback("on_train_epoch_start", callback)
            print(f"[Finetune Strategy] Registered gradual unfreeze callback at epochs {unfreeze_schedule}.")
        elif hasattr(model, "yolo") and hasattr(model.yolo, "add_callback"):
            model.yolo.add_callback("on_train_epoch_start", callback)
            print(f"[Finetune Strategy] Registered gradual unfreeze callback on underlying YOLO model at epochs {unfreeze_schedule}.")
        else:
            print("[Warning] Could not register callback: model instance has no add_callback method.")

    return freeze_layers


if __name__ == "__main__":
    print("--- Testing Unfreeze Schedule ---")
    dummy_config = {
        "train": {
            "finetune_strategy": "gradual_unfreeze",
            "unfreeze_schedule": [10, 20, 30]
        }
    }

    # 1. Test get_freeze_layers with default total_layers=23
    freeze_list = get_freeze_layers("gradual_unfreeze", total_layers=23)
    print(f"Initial freeze layers for gradual_unfreeze (total_layers=23): {freeze_list}")
    assert freeze_list == list(range(21)), f"Expected layers 0..20 frozen initially, got {freeze_list}"
    assert 21 not in freeze_list and 22 not in freeze_list, "Head layers (21, 22) must not be frozen!"

    # 2. Test callback creation and simulation
    callback = make_unfreeze_callback(unfreeze_schedule=[10, 20, 30], total_layers=23)

    # Mock parameter object
    class DummyParam:
        def __init__(self, name):
            self.name = name
            self.requires_grad = False

    # Mock model
    class DummyModel:
        def __init__(self):
            self._params = {
                f"model.{i}.conv.weight": DummyParam(f"model.{i}.conv.weight")
                for i in range(23)
            }

        def named_parameters(self):
            return self._params.items()

    # Mock trainer
    class DummyTrainer:
        def __init__(self, epoch):
            self.epoch = epoch
            self.model = DummyModel()

    # Simulate epoch 10
    trainer_e10 = DummyTrainer(epoch=10)
    print("\nSimulating callback at epoch 10:")
    callback(trainer_e10)

    # Verify parameters unfrozen at epoch 10
    unfrozen_at_e10 = [name for name, p in trainer_e10.model.named_parameters() if p.requires_grad]
    print(f"Unfrozen parameters at epoch 10: {unfrozen_at_e10}")
    assert len(unfrozen_at_e10) > 0, "No parameters were unfrozen at epoch 10!"

    print("\nUnfreeze schedule module test completed successfully!")
