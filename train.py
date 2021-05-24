import argparse
import os
import shutil

from trainer import Trainer
from utils import load_config, load_trainer, set_seed


def main(args):
    checkpoint_path = args.checkpoint_path
    if checkpoint_path:
        trainer = load_trainer(checkpoint_path)
    else:
        config = load_config(args.config_path)
        trainer = Trainer(config)
        shutil.copy(
            args.config_path, os.path.join(trainer.save_path, "config.json"),
        )
    try:
        trainer.train()
    except KeyboardInterrupt:
        if input("Save checkpoint? [y/n] ") == "y":
            trainer.save_checkpoint("last")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join("configs", "model.json"),
        help="path to config json file",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, help="path to checkpoint file",
    )
    args = parser.parse_args()
    set_seed()
    main(args)
