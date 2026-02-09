"""Configuration management."""

import json
from pathlib import Path

DEFAULT_CONFIG = {
    # Cellpose parameters
    "cellpose_flow_threshold": 0.4,
    "cellpose_cellprob_threshold": 0.0,
    # Erosion parameters
    "erosion_pixels": 5,
    # Inclusion detection parameters
    "kernel_size": 7,
    "tophat_threshold": 30,
    "min_inclusion_area": 7,
    "max_inclusion_area": 50,
    "edge_buffer": 5,
    # Droplet filtering
    "min_droplet_diameter": 80,
    "max_droplet_diameter": 200,
    # Conversion factor
    "px_to_um": 1.14,
    # Cache settings
    "cache": {
        "enabled": True,
        "max_frames": 100,
        "strategy": "lru",
    },
    # Project settings defaults
    "settings": {
        "dilution": 500,
        "poisson": True,
        "count": 6.5e5,
        "inclusions": True,
    },
}


def load_config(config_path=None):
    """Load configuration from JSON file or use defaults.

    Args:
        config_path: Path to config.json. If None, looks in current directory.

    Returns:
        dict: Configuration dictionary with defaults merged.
    """
    config = DEFAULT_CONFIG.copy()

    if config_path is None:
        # Look for config.json in current directory or project root
        search_paths = [
            Path.cwd() / "config.json",
            Path(__file__).parent.parent.parent / "config.json",
        ]
        for path in search_paths:
            if path.exists():
                config_path = path
                break

    if config_path and Path(config_path).exists():
        print(f"Loading config from: {config_path}")
        with open(config_path, "r") as f:
            loaded_config = json.load(f)
            # Deep merge for nested dicts like 'cache'
            for key, value in loaded_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
    else:
        print("Using default configuration (no config.json found)")

    return config
