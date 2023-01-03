import json
import os
from dotmap import DotMap
import pathlib


def get_config_from_json():
    json_path = "configs\config.json"
    with open(json_path, 'r') as config_file:
        config_dict = json.load(config_file)
    config = DotMap(config_dict)
    return config


def process_config(generation=False):
    config = get_config_from_json()
    if not generation:
        exp_path = str(pathlib.Path().resolve()) + "\\experiments"
        if not os.path.isdir(exp_path):
            os.makedirs(exp_path)
        config.callbacks.checkpoint_dir_val_acc = exp_path + "/max_val_accuracy"
        config.callbacks.checkpoint_dir_acc = exp_path + "/max_accuracy"
    return config
