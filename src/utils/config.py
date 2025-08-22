import os
import yaml
from typing import Dict, Any

CONFIG_PATH = os.path.expanduser("/home/jonathanpittman/Documents/gnn-gui/config/default.yaml")

_DEFAULT_CONFIG: Dict[str, Any] = {
    "workspace_dir": "/home/jonathanpittman/Documents/gnn-gui",
    "jobs_dir": "/home/jonathanpittman/Documents/gnn-gui/jobs",
    "logs_dir": "/home/jonathanpittman/Documents/gnn-gui/logs",
    "conversion": {"script_path": "", "default_args": "--inputs {inputs} --output {output_dir}"},
    "training": {"script_path": "", "default_args": "--data {dataset_dir} --model {model}"},
    "slurm": {
        "use_slurm_by_default": False,
        "account": "",
        "partition": "",
        "qos": "",
        "gpus": 1,
        "cpus": 4,
        "mem": "16G",
        "time": "02:00:00",
        "additional": "",
        "env_activation": "",
    },
}


def ensure_dirs(config: Dict[str, Any]) -> None:
    for key in ["workspace_dir", "jobs_dir", "logs_dir"]:
        path = config.get(key)
        if path and not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)


def load_config() -> Dict[str, Any]:
    if not os.path.isfile(CONFIG_PATH):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        save_config(_DEFAULT_CONFIG)
        return dict(_DEFAULT_CONFIG)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Merge defaults
    def merge(d, default):
        for k, v in default.items():
            if isinstance(v, dict):
                d[k] = merge(d.get(k, {}) or {}, v)
            else:
                d.setdefault(k, v)
        return d
    data = merge(data, _DEFAULT_CONFIG)
    ensure_dirs(data)
    return data


def save_config(config: Dict[str, Any]) -> None:
    ensure_dirs(config)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

