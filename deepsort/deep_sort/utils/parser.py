import yaml
from easydict import EasyDict as edict

def read_from_file(config_path):
    with open(config_path, 'r') as f:
        cfg = edict(yaml.safe_load(f))
    return cfg
