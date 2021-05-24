import sys

sys.path.append("..")

import argparse
import os

import torch
from tqdm import tqdm

from data.preprocess import Preprocessor
from utils import load_config


def main(args):
    config = load_config(args.config_path)
    preprocessor = Preprocessor(config)
    bin_path = args.bin_path
    if not os.path.isdir(bin_path):
        os.mkdir(bin_path)
        for mode in ("train", "valid"):
            os.mkdir(os.path.join(bin_path, mode))
    data_path = os.path.join("data", "raw")
    midi_root = os.path.join(data_path, "mid")
    text_root = os.path.join(data_path, "txt")
    wav_root = os.path.join(data_path, "wav")
    midi_list = os.listdir(midi_root)
    midi_list.sort()
    for midi_file in tqdm(midi_list):
        file_name, _ = os.path.splitext(midi_file)
        midi_path = os.path.join(midi_root, midi_file)
        text_path = os.path.join(text_root, f"{file_name}.txt")
        wav_path = os.path.join(wav_root, f"{file_name}.wav")
        if file_name[:-1] == args.valid_song:
            mode = "valid"
        elif file_name[:-1] == args.test_song:
            continue
        else:
            mode = "train"
        result = preprocessor(midi_path, text_path, wav_path)
        torch.save(result, os.path.join(bin_path, mode, f"{file_name}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_song", type=str, default="")
    parser.add_argument("--test_song", type=str, default="")
    parser.add_argument("--bin_path", type=str, default=os.path.join("data", "bin"))
    parser.add_argument(
        "--config_path", type=str, default=os.path.join("configs", "preprocess.json")
    )
    args = parser.parse_args()
    main(args)
