import argparse
import os
from typing import Any, Dict, Optional

import yaml


def str2bool(value: str) -> bool:
    """argparse type for booleans: ``bool("False")`` would be True."""
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got '{value}'")


class ConfigParser:
    """
    Parser for YAML configuration files with command-line overrides.

    Example usage:
        config_parser = ConfigParser()
        config = config_parser.parse(config_path="config/custom_config.yaml")
        print(config["model"])

    """

    def __init__(self, default_config_path: str = "config/defaults.yaml"):
        """
        Initialize the config parser with optional default configuration path.

        Args:
            default_config_path: Path to the default configuration file
        """
        self.config = {}
        self.default_config_path = default_config_path

    def parse(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse configuration from YAML files and command-line arguments.

        Args:
            config_path: Path to specific configuration file (overrides defaults)

        Returns:
            Dictionary containing the merged configuration
        """
        # Load default configuration
        if os.path.exists(self.default_config_path):
            with open(self.default_config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}

        # Load specific configuration (if provided)
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                specific_config = yaml.safe_load(f) or {}
                # Merge with defaults (specific config takes precedence)
                self._update_nested_dict(self.config, specific_config)

        # Parse command-line arguments to override config values
        self._parse_command_line_args()

        return self.config

    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Recursively update a nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d

    def _parse_command_line_args(self) -> None:
        """Parse command-line arguments and update configuration."""
        parser = argparse.ArgumentParser(description="Override configuration parameters")

        # Add argument for configuration file
        parser.add_argument('--config', type=str, help='Path to configuration file')

        # Add arguments for each top-level config key to allow overrides
        for key, value in self.config.items():
            if isinstance(value, bool):
                parser.add_argument(f'--{key}', type=str2bool, help=f'Override {key}')
            elif isinstance(value, (int, float, str)):
                parser.add_argument(f'--{key}', type=type(value), help=f'Override {key}')

        args = parser.parse_args()

        # If config file specified via command line, load it
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                cmd_config = yaml.safe_load(f) or {}
                self._update_nested_dict(self.config, cmd_config)

        # Update config with command-line argument values
        for key, value in vars(args).items():
            if key != 'config' and value is not None and key in self.config:
                self.config[key] = value
