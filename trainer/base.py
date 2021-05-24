import os
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter

from data import make_loaders
from utils import EarlyStopMonitor, make_directory

warnings.simplefilter(action="ignore", category=FutureWarning)

# torch.autograd.set_detect_anomaly(True)


class BaseTrainer:
    def __init__(self, config):
        self.epoch = 0
        self.iteration = 0
        self.config = config
        self.monitor = EarlyStopMonitor(config.patience)
        self.device = torch.device(config.device)
        self.train_loader, self.valid_loader = make_loaders(config.data)
        self.num_train_iters = len(self.train_loader)
        self.num_valid_iters = len(self.valid_loader)
        log_path = os.path.join("logs", config.name)
        make_directory(log_path)
        self.writer = SummaryWriter(log_dir=log_path)
        save_path = os.path.join("checkpoints", config.name)
        make_directory(save_path)
        self.save_path = save_path
        self.save_keys = [
            "epoch",
            "iteration",
            "model",
            "optimizer",
            "scheduler",
            "monitor",
        ]

    def register_save_keys(self, *args):
        self.save_keys += args

    def _train_epoch(self):
        raise NotImplementedError

    @torch.no_grad()
    def _valid_epoch(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def save_checkpoint(self, title=""):
        assert (
            len(self.save_keys) > 2
        ), "Make sure to call `.register_save_keys()` to register attributes to save"
        if not title:
            title = self.epoch
        path = os.path.join(self.save_path, f"{title}.pt")
        save_dict = {}
        for key in self.save_keys:
            try:
                save_dict[key] = getattr(self, key).state_dict()
            except AttributeError:
                save_dict[key] = getattr(self, key)
        torch.save(save_dict, path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        for key, value in checkpoint.items():
            try:
                self.__dict__[key].load_state_dict(value)
            except AttributeError:
                self.__dict__[key] = value

    def log(self, dictionary):
        for key, value in dictionary.items():
            self.writer.add_scalar(key, value, self.epoch)
