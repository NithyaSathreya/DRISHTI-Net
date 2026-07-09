"""
config.py
"""

import yaml


class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)


def load_config(filename):
    with open(filename, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(cfg)