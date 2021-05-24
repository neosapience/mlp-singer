import json
import os
import random
import warnings
from argparse import Namespace

import numpy as np
import torch

warnings.simplefilter(action="ignore", category=DeprecationWarning)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AttrDict(Namespace):
    def __init__(self, dictionary: dict):
        for key, value in dictionary.items():
            value = AttrDict(value) if isinstance(value, dict) else value
            setattr(self, key, value)

    def __setattr__(self, key, value):
        value = AttrDict(value) if isinstance(value, dict) else value
        super().__setattr__(key, value)

    def to_dict(self):
        return vars(self)


def load_checkpoint_config(checkpoint_path: str) -> AttrDict:
    root, _ = os.path.split(checkpoint_path)
    config_path = os.path.join(root, "config.json")
    return load_config(config_path)


def load_config(config_path: str) -> AttrDict:
    with open(config_path) as f:
        return AttrDict(json.load(f))


def load_trainer(checkpoint_path):
    # avoid circular import
    from trainer import Trainer

    config = load_checkpoint_config(checkpoint_path)
    trainer = Trainer(config)
    trainer.load_checkpoint(checkpoint_path)
    return trainer


def load_model(checkpoint_path, eval=False):
    config = load_checkpoint_config(checkpoint_path)
    model = init_model(config.model)
    try:
        checkpoint = torch.load(checkpoint_path)
    except RuntimeError:
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    if eval:
        model.eval()
    return model


def init_model(model_config):
    from model import MLPSinger

    model = MLPSinger(model_config)
    return model


def to_device(xs, device):
    moved_xs = []
    for x in xs:
        if isinstance(x, torch.Tensor):
            moved_xs.append(x.to(device))
        else:
            moved_xs.append(x)
    return moved_xs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopMonitor:
    def __init__(self, patience, mode="min"):
        mode = mode.lower()
        assert mode.lower() in {
            "min",
            "max",
        }, f"Expected `mode` to be one of 'min' or 'max', but got {mode} instead"
        self.log = []
        self.mode = mode
        self.patience = patience

    def check(self, metric):
        if not self.log:
            self.log.append(metric)
            return False
        flag = metric > self.log[-1]
        if flag == (self.mode == "min"):
            self.log.append(metric)
        else:
            self.log = []
        return len(self.log) > self.patience


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
