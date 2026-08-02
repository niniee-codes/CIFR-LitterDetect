from typing import Dict, Any, Callable, Optional


def init_wandb(config: Dict[str, Any]) -> Any:
    """
    Initializes a Weights & Biases run from an experiment configuration dict.

    Args:
        config (dict): Loaded experiment config dict.

    Returns:
        wandb_run or None: Active W&B run object if successful, else None.
    """
    try:
        import wandb

        logging_cfg = config.get("logging", {})
        wandb_project = logging_cfg.get("wandb_project", config.get("project", "floating-litter-cifr"))
        wandb_entity = logging_cfg.get("wandb_entity", None)
        run_name = config.get("run_name", "unnamed_run")
        tags = config.get("wandb", {}).get("tags", [])

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=config,
            tags=tags,
            reinit=True
        )
        print(f"[W&B] Successfully initialized Weights & Biases run: {run_name}")
        return run
    except Exception as e:
        print(f"[Warning] Failed to initialize Weights & Biases logging ({e}). Continuing training without W&B.")
        return None


def get_ultralytics_wandb_callback(wandb_run: Any) -> Dict[str, Callable]:
    """
    Returns a dictionary of Ultralytics-compatible callbacks that log training and
    validation metrics to W&B after each epoch.

    Args:
        wandb_run: Active W&B run object or None.

    Returns:
        dict: Ultralytics callback mapping (e.g. {'on_fit_epoch_end': callback_fn}).
    """
    def on_train_epoch_end(trainer):
        if wandb_run is None:
            return
        metrics_to_log = {}
        if hasattr(trainer, "metrics") and isinstance(trainer.metrics, dict):
            for key, value in trainer.metrics.items():
                try:
                    metrics_to_log[f"epoch/{key}"] = float(value)
                except (TypeError, ValueError):
                    continue
        if hasattr(trainer, "epoch"):
            metrics_to_log["epoch/epoch_num"] = trainer.epoch
        if metrics_to_log:
            wandb_run.log(metrics_to_log)

    return {
        "on_fit_epoch_end": on_train_epoch_end,
        "on_train_epoch_end": on_train_epoch_end,
    }


def finish_wandb(wandb_run: Any) -> None:
    """
    Safely finishes an active Weights & Biases run.

    Args:
        wandb_run: Active W&B run object or None.
    """
    if wandb_run is not None:
        try:
            wandb_run.finish()
            print("[W&B] Finished Weights & Biases run.")
        except Exception as e:
            print(f"[Warning] Error finishing W&B run: {e}")
