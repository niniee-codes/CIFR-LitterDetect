import argparse
import sys
import tempfile
from pathlib import Path
import yaml
from ultralytics import YOLO

# Support running from root or src directory
try:
    from src.config_loader import load_config
    from src.models.yolov8_cifr import YOLOv8WithCIFR
    from src.engine.unfreeze_schedule import apply_finetune_strategy
    from src.utils.wandb_logger import init_wandb, get_ultralytics_wandb_callback, finish_wandb
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config_loader import load_config
    from src.models.yolov8_cifr import YOLOv8WithCIFR
    from src.engine.unfreeze_schedule import apply_finetune_strategy
    from src.utils.wandb_logger import init_wandb, get_ultralytics_wandb_callback, finish_wandb


def fix_dataset_yaml_path(yaml_path: Path) -> Path:
    """
    Reads dataset YAML, sets its 'path' field to the absolute path of its parent
    directory with forward slashes, and writes it to a temporary file.

    Args:
        yaml_path (Path): Original dataset YAML path.

    Returns:
        Path: Path to temporary modified YAML file.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    parent_abs_path = yaml_path.resolve().parent.as_posix()
    data["path"] = parent_abs_path

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(data, temp_file)
    temp_file.close()

    return Path(temp_file.name)


def main():
    parser = argparse.ArgumentParser(description="Main training entrypoint for CIFR-LitterDetect.")
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file (e.g. configs/m4_fml_cifr.yaml)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (epochs=1, batch=2, CPU-friendly)"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Initialize Weights & Biases run
    wandb_run = init_wandb(config)

    # Resolve model base name
    model_base = config["model"]["base"]
    model_variant = model_base if model_base.endswith(".pt") else f"{model_base}.pt"

    use_cifr = config["model"].get("use_cifr", False)
    weighting_scheme = config["model"].get("cifr_weighting", "inv_sqrt_freq")
    run_name = config.get("run_name", "unnamed_run")

    # Resolve dataset YAML path
    target_key = config["data"].get("target")
    if target_key and target_key in config["data"]:
        data_yaml_path = config["data"][target_key]
    else:
        data_yaml_path = config["data"].get(
            "eightclass_yaml_path",
            config["data"].get("eightclass_yaml", "data/eightclass/datasets/RFT.yaml")
        )

    data_path = Path(data_yaml_path)

    epochs = 1 if args.debug else config["train"].get("epochs", 100)
    batch_size = 2 if args.debug else config["train"].get("batch_size", 16)
    device = "cpu" if args.debug else config["train"].get("device", 0)
    img_size = config["model"].get("img_size", 640)

    # Print summary before training starts
    print("\n==================================================")
    print("           TRAINING CONFIGURATION SUMMARY         ")
    print("==================================================")
    print(f"  Run Name      : {run_name}")
    print(f"  Model Base    : {model_variant}")
    print(f"  Use CIFR      : {use_cifr}")
    if use_cifr:
        print(f"  Weight Scheme : {weighting_scheme}")
    print(f"  Dataset Path  : {data_path}")
    print(f"  Epochs        : {epochs} {'(Debug Mode)' if args.debug else ''}")
    print(f"  Batch Size    : {batch_size}")
    print(f"  Image Size    : {img_size}")
    print(f"  Device        : {device}")
    print("==================================================\n")

    if not data_path.exists():
        print(f"[ERROR] Dataset configuration file does not exist: {data_path}")
        print("Please check your dataset path configuration.")
        finish_wandb(wandb_run)
        sys.exit(1)

    # Dynamically fix dataset yaml 'path' field to absolute parent path at runtime
    temp_data_path = fix_dataset_yaml_path(data_path)

    # Instantiate model & start training
    try:
        if use_cifr:
            weights_json = "results/tables/class_weights.json"
            model = YOLOv8WithCIFR(
                model_variant=model_variant,
                class_weights_json_path=weights_json,
                weighting_scheme=weighting_scheme,
            )
            print("Successfully initialized YOLOv8WithCIFR model.")
            try:
                total_layers = len(model.model.model)
            except Exception:
                total_layers = 23
            freeze_layers = apply_finetune_strategy(model, config, total_layers=total_layers)

            # Register W&B logging callback
            wandb_callbacks = get_ultralytics_wandb_callback(wandb_run)
            if "on_fit_epoch_end" in wandb_callbacks and hasattr(model.yolo, "add_callback"):
                model.yolo.add_callback("on_fit_epoch_end", wandb_callbacks["on_fit_epoch_end"])

            model.train_yolo(
                data=str(temp_data_path),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                device=device,
                project="runs/detect",
                name=run_name,
                freeze=freeze_layers,
            )
        else:
            model = YOLO(model_variant)
            print("Successfully initialized plain YOLO model.")
            try:
                total_layers = len(model.model.model)
            except Exception:
                total_layers = 23
            freeze_layers = apply_finetune_strategy(model, config, total_layers=total_layers)

            # Register W&B logging callback
            wandb_callbacks = get_ultralytics_wandb_callback(wandb_run)
            if "on_fit_epoch_end" in wandb_callbacks and hasattr(model, "add_callback"):
                model.add_callback("on_fit_epoch_end", wandb_callbacks["on_fit_epoch_end"])

            model.train(
                data=str(temp_data_path),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                device=device,
                project="runs/detect",
                name=run_name,
                freeze=freeze_layers,
            )

    except FileNotFoundError as e:
        print(f"\n[ERROR] Training failed due to missing file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An error occurred during training: {e}")
        sys.exit(1)
    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
