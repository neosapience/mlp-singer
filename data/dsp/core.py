import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read

from .audio_processing import dynamic_range_compression, dynamic_range_decompression
from .stft import STFT


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        sampling_rate=22050,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        max_wav_value=32768.0,
    ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.frame_len = hop_length / sampling_rate * 1000
        self.max_wav_value = max_wav_value

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (1, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, _ = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        return self.lin2mel(magnitudes)

    def lin2mel(self, lin):
        if not torch.is_tensor(lin):
            lin = torch.from_numpy(lin)
        mel_output = torch.matmul(self.mel_basis, lin)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output.squeeze(0).transpose(0, 1)

    def load_wav(self, path, normalize=True):
        source_rate, audio = read(path)
        audio = torch.FloatTensor(audio.astype(np.float32))
        if audio.ndim > 1 and len(audio) > 1:
            audio = audio.mean(-1)
        if source_rate != self.sampling_rate:
            resample = torchaudio.transforms.Resample(source_rate, self.sampling_rate)
            audio = resample(audio)
        if normalize:
            audio = audio / self.max_wav_value
        audio = audio.unsqueeze(0)
        return audio
