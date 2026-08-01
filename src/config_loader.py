import yaml
import os

def load_config(path):
    """
    Load a YAML config file. If it has a 'defaults' key pointing to a base 
    config, merge that base config with this file's overrides (deep merge 
    for nested dicts: data, model, train, logging, wandb).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "defaults" in cfg:
        config_dir = os.path.dirname(path)
        base_path = os.path.join(config_dir, cfg["defaults"])
        with open(base_path) as f:
            base = yaml.safe_load(f)

        merged = dict(base)
        for key, value in cfg.items():
            if key == "defaults":
                continue
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    return cfg


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = load_config(sys.argv[1])
        print(yaml.dump(result, default_flow_style=False))
