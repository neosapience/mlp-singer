import torch
from torch.nn import functional as F
from tqdm import tqdm

from utils import count_parameters, init_model, to_device

from .base import BaseTrainer
from .scheduler import NoamScheduler


class Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.model = init_model(config.model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = NoamScheduler(self.optimizer, config.warmup_steps)
        tqdm.write(f"model parameters: {count_parameters(self.model)}")

    def _train_epoch(self):
        total_loss = 0
        self.model.train()

        for batch in tqdm(self.train_loader, leave=False):
            self.iteration += 1
            pitch, text, mel = to_device(batch, self.device)

            self.optimizer.zero_grad()
            pred = self.model(pitch, text)
            loss = F.l1_loss(pred, mel)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.scheduler.step()

        loss_dict = {"train/loss": total_loss / self.num_train_iters}
        return loss_dict

    @torch.no_grad()
    def _valid_epoch(self):
        total_loss = 0
        self.model.eval()

        for batch in tqdm(self.valid_loader, leave=False):
            pitch, text, mel = to_device(batch, self.device)

            pred = self.model(pitch, text)
            loss = F.l1_loss(pred, mel)
            total_loss += loss.item()

        loss_dict = {"valid/loss": total_loss / self.num_valid_iters}
        return loss_dict

    def train(self):
        best_loss = float("inf")
        num_epochs = self.config.num_epochs

        for _ in range(num_epochs):
            self.epoch += 1
            train_dict = self._train_epoch()
            valid_dict = self._valid_epoch()
            valid_loss = valid_dict["valid/loss"]
            if self.epoch % self.config.log_interval == 0:
                self.log(train_dict)
                self.log(valid_dict)
                self.log({"lr": self.scheduler.get_last_lr()[0]})
            if valid_loss < best_loss:
                self.save_checkpoint("best")
                best_loss = valid_loss
                continue
            elif self.epoch % self.config.save_interval == 0:
                self.save_checkpoint()
            if self.monitor.check(valid_loss):
                break
        if valid_loss != best_loss:
            self.save_checkpoint("last")
