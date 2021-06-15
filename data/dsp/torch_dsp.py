import librosa
import torch
import torchaudio
from torchaudio.transforms import Resample, Spectrogram


def load(path, sample_rate=22050):
    waveform, source_rate = torchaudio.load(path)
    if len(waveform) > 1:
        waveform = waveform.mean(dim=0)
    if source_rate != sample_rate:
        resample = Resample(source_rate, sample_rate)
        waveform = resample(waveform)
    return waveform


class SignalProcessor:
    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        f_min=0,
        f_max=8000,
        n_mels=80,
        a_min=1e-5,
    ):
        self.spect_fn = Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=1,
        )
        self.mel_basis = torch.from_numpy(
            librosa.filters.mel(sample_rate, n_fft, n_mels, f_min, f_max)
        ).float()
        self.a_min = a_min

    def spectrogram(self, wav):
        return self.spect_fn(wav)

    def mel_spectrogram(self, wav):
        lin = self.spect_fn(wav)
        return torch.matmul(self.mel_basis, lin)

    def log_mel_spectrogram(self, wav):
        mel = self.mel_spectrogram(wav)
        return torch.log(torch.clamp(mel, min=self.a_min))
