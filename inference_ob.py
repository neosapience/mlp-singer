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
    device = args.device
    assert device in {"cpu", "cuda"}, "device must be one of `cpu` or `cuda`"
    save_path = args.save_path
    make_directory(save_path)
    make_directory(args.mel_path)
    model = load_model(args.checkpoint_path, eval=True)
    preprocessor_config = load_config(args.preprocessor_config_path)
    preprocessor = Preprocessor(preprocessor_config)
    mel_dim = preprocessor_config.n_mel_channels
    song = args.song
    data_path = args.data_path
    notes, phonemes = preprocessor.prepare_inference(
        os.path.join(data_path, "mid", f"{song}.mid"),
        os.path.join(data_path, "txt", f"{song}.txt"),
    )
    chunk_size = model.seq_len
    preds = []
    total_len = len(notes)
    notes = notes.to(device)
    phonemes = phonemes.to(device)
    total_len = len(notes) - chunk_size
    frames_to_drop = args.num_overlap
    step = chunk_size - 2 * frames_to_drop
    remainder = total_len % step
    pad_size = 0
    if remainder:
        pad_size = step - remainder
        padding = torch.zeros(pad_size, dtype=int).to(device)
        phonemes = torch.cat((phonemes, padding))
        notes = torch.cat((notes, padding))
    notes = notes.unfold(size=chunk_size, step=step, dimension=0)
    phonemes = phonemes.unfold(size=chunk_size, step=step, dimension=0)
    preds = model(notes, phonemes)
    mel_dim = preds.size(-1)
    first = preds[0][:-frames_to_drop]
    last = preds[-1][frames_to_drop:-pad_size]
    preds = preds[1:-1, frames_to_drop:-frames_to_drop]
    preds = preds.reshape(-1, mel_dim)
    preds = torch.cat((first, preds, last))
    preds = preds.transpose(0, 1).unsqueeze(0)
    np.save(os.path.join(args.mel_path, f"{song}.npy"), preds.numpy())
    subprocess.call(
        f"cd hifi-gan; python inference_e2e.py --checkpoint_file {args.hifi_gan} --output_dir ../{save_path}",
        shell=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="device to use")
    parser.add_argument(
        "--num_overlap", type=int, default=30, help="number of overlapping frames"
    )
    parser.add_argument(
        "--mel_path",
        type=str,
        default=os.path.join("hifi-gan", "test_mel_files"),
        help="path to save synthesized mel-spectrograms",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="samples",
        help="path to save synthesized .wav file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/default/model.pt",
        help="path to checkpoint file",
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
