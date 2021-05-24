import torch
from torch import nn

from .modules import MLPMixer


class MLPSinger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        d_model = config.d_pitch + config.d_phone
        self.text_embed = nn.Embedding(config.num_phone, config.d_phone)
        self.pitch_embed = nn.Embedding(config.num_pitch, config.d_pitch)
        self.embed = nn.Linear(d_model, d_model)
        self.decoder = MLPMixer(
            d_model=d_model,
            seq_len=config.seq_len,
            expansion_factor=config.expansion_factor,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.proj = nn.Linear(d_model, config.mel_dim)

    def forward(self, pitch, phonemes):
        pitch_embedding = self.pitch_embed(pitch)
        text_embedding = self.text_embed(phonemes)
        x = torch.cat((text_embedding, pitch_embedding), dim=-1)
        x = self.embed(x)
        x = self.decoder(x)
        out = self.proj(x)
        return out
