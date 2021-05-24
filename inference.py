import sys

sys.path.append("hifi-gan")

import argparse
import os
import subprocess

import numpy as np
import torch

from data.preprocess import Preprocessor
from utils import load_config, load_model, make_directory, set_seed


@torch.no_grad()
def main(args):
    make_directory(args.save_path)
    model = load_model(args.checkpoint_path, eval=True)
    preprocessor_config = load_config(args.preprocessor_config_path)
    preprocessor = Preprocessor(preprocessor_config)
    song = args.song
    data_path = args.data_path
    notes, phonemes = preprocessor.prepare_inference(
        os.path.join(data_path, "mid", f"{song}.mid"),
        os.path.join(data_path, "txt", f"{song}.txt"),
    )
    chunk_size = model.seq_len
    preds = []
    total_len = len(notes)
    notes = notes.unsqueeze(0)
    phonemes = phonemes.unsqueeze(0)
    num_chunks, remainder = divmod(total_len, chunk_size)
    for j in range(num_chunks):
        start = j * chunk_size
        end = start + chunk_size
        pred = model(notes[:, start:end], phonemes[:, start:end],)
        preds.append(pred)
    preds = torch.cat(tuple(preds), dim=1)
    if remainder:
        last = model(notes[:, -chunk_size:], phonemes[:, -chunk_size:])
        preds = torch.cat((preds[:, : -(chunk_size - remainder)], last), dim=1)
    assert preds.size(1) == total_len
    preds = preds.transpose(1, 2)
    np.save(os.path.join(args.save_path, f"{song}.npy"), preds.numpy())
    subprocess.call(
        f"cd hifi-gan; python inference_e2e.py --checkpoint_file {args.hifi_gan} --output_dir ../samples",
        shell=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join("hifi-gan", "test_mel_files"),
        help="path to save synthesized .wav file",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="", help="path to checkpoint file",
    )
    parser.add_argument(
        "--hifi_gan",
        type=str,
        default="g_02500000",
        help="path to hifi gan generator checkpoint file",
    )
    parser.add_argument(
        "--preprocessor_config_path",
        type=str,
        default=os.path.join("configs", "preprocess.json"),
        help="path to preprocessor config file",
    )
    parser.add_argument("--data_path", type=str, default=os.path.join("data", "raw"))
    parser.add_argument(
        "--song", type=str, default="little_star", help="song to infer on"
    )
    args = parser.parse_args()
    set_seed()
    main(args)
