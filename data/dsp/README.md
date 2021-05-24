# Digital Signal Processing

Utility modules to produce mel-spectrograms and normalize audio.

## Acknowledgements

The files included in this subdirectory were directly adopted from [NVIDIA Tacotron 2](https://github.com/NVIDIA/tacotron2) with minimum modifications. We also realize that the Tacotron 2 code was released at a point in time when `torchaudio`did not support STFT or other signal processing functions. For convenience, we have also included a `torch_dsp.py` script that contains utility functions that can replicate preprocessed output using `torchaudio` built-ins.