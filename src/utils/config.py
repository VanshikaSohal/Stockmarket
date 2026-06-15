"""
Configuration loader utility.
"""

import os
import yaml


def load_config(config_path=None):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str, optional): Path to the config file. Defaults to
            'config.yaml' in the project root (two levels up from this file).

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config.yaml"
        )
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
