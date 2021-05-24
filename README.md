# MLP Singer

Official implementation of MLP Singer: An All-MLP Architecture for Parallel Korean Singing Voice Synthesis.

## Introduction

We present MLP Singer, an all-MLP architecture for parallel Korean singing voice synthesis. This work is directly inspired by recent works that use multi-layer perceptrons to replace computationally costly self-attention layers in transformers. We extend discriminative MLP architectures proposed in the computer vision literature to build a generative model that outputs mel-spectrograms given lyrics text and a corresponding quantized pitch sequence as discrete MIDI notes. To the best of our knowledge, this is the first work that uses an entirely MLP-based architecture for voice synthesis. The proposed system serves as a strong baseline, achieving comparable performance to auto-regressive and GAN-based SVS systems with fraction of their number of parameters and orders of magnitude quicker training and inference speed.

## Quickstart

Install project requirements.

```
pip install -r requirements.txt
```

To generate audio files, run 

```
python inference.py --checkpoint_path PATH/TO/CHECKPOINT.pt
```

## Dataset

We used the [Children Song Dataset](https://github.com/emotiontts/emotiontts_open_db/tree/master/Dataset/CSD), a to-be open-source singing voice dataset comprised of 100 annotated Korean and English children songs sung by a single professional singer. We used only the Korean subset of the dataset to train the model.

You can train the model on any custom dataset of your choice, as long as it includes lyrics text, midi transcriptions, and monophonic a capella audio file triplets. These files should be titled identically, and should also be placed in specific directory locations as shown below.

```
├── data
│   └── raw
│       ├── mid
│       ├── txt
│       └── wav
```

The directory names correspond to file extensions. We have included a sample as reference.

## Preprocessing

Once you have prepared the dataset, run 

```
python -m data.serialize
```

from the root directory. This will create `data/bin` that contains binary files used for training. This repository already contains example binary files created from the sample in `data/raw`. 

## Training

To train the model, run

```
python train.py
```

This will read the default configuration file located in `configs/model.json` to initialize the model. Alternatively, you could create a new configuration file and train the model via

```
python train.py --config_path PATH/TO/CONFIG.json
```

Running this command will create a folder under the `checkpoints` directory according to the `name` field specified in the configuration file.

You can also continue training from a checkpoint. For example, to resume training from the provided pretrained model checkpoint, run

```
python train.py --checkpoint_path PATH/TO/CHECKPOINT.pt
```

Unless explicitly specified via a `--config_path` flag, the script will read `config.json` in the checkpoint directory. In both cases, model checkpoints will be saved regularly according to the interval defined in the configuration file. 

## Inference

MLP Singer produces mel-spectrograms, which are then fed into a neural vocoder to generate raw waveforms. We use [HiFi-GAN](https://github.com/jik876/hifi-gan) as the vocoder backend, but you could also plug other vocoders like [WaveGlow](https://github.com/NVIDIA/waveglow).

```
python inference.py --checkpoint_path PATH/TO/CHECKPOINT.pt
```

This will create `.wav` samples in the `samples` directory, and save mel-spectrogram files as `.npy` files in `hifi-gan/test_mel_dirs`. 

You can also specify any song you want to perform inference on, as long as the song is present in `data/raw`. 

```
python inference.py --checkpoint_path PATH/TO/CHECKPOINT.pt --song little_star
```

The argument to the `--song` flag should match the title of the song as it is saved in `data/raw`.  

## Acknowledgements

This implementation was inspired by the following repositories.

* [Tacotron2](https://github.com/NVIDIA/tacotron2)
* [BEGANSing](https://github.com/SoonbeomChoi/BEGANSing)
* [pytorch-saltnet](https://github.com/tugstugi/pytorch-saltnet)
* [hifi-gan](https://github.com/jik876/hifi-gan)


## Citations

```bibtex
@inproceedings{choi2020children,
  title={Children’s Song Dataset for Singing Voice Research},
  author={Choi, Soonbeom and Kim, Wonil and Park, Saebyul and Yong, Sangeon and Nam, Juhan},
  booktitle={The 21th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2020},
  organization={International Society for Music Information Retrieval}
}
```

```bibtex
@misc{cho2017kog2p,
  title = {Korean Grapheme-to-Phoneme Analyzer (KoG2P)},
  author = {Yejin Cho},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/scarletcho/KoG2P}}
}
```

```bibtex
@misc{tolstikhin2021mlpmixer,
      title={MLP-Mixer: An all-MLP Architecture for Vision}, 
      author={Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Andreas Steiner and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
      year={2021},
      eprint={2105.01601},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@inproceedings{NEURIPS2020_c5d73680,
 author = {Kong, Jungil and Kim, Jaehyeon and Bae, Jaekyoung},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {17022--17033},
 publisher = {Curran Associates, Inc.},
 title = {HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis},
 volume = {33},
 year = {2020}
}
```