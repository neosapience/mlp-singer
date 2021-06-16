# MLP Singer

Official implementation of [MLP Singer: Towards Rapid Parallel Korean Singing Voice Synthesis](https://arxiv.org/abs/2106.07886). Audio samples are available on our [demo page](https://mlpsinger.github.io).

## Abstract

> Recent developments in deep learning have significantly improved the quality of synthesized singing voice audio. However, prominent neural singing voice synthesis systems suffer from slow inference speed due to their autoregressive design. Inspired by MLP-Mixer, a novel architecture introduced in the vision literature for attention-free image classification, we propose MLP Singer, a parallel Korean singing voice synthesis system. To the best of our knowledge, this is the first work that uses an entirely MLP-based architecture for voice synthesis. Listening tests demonstrate that MLP Singer outperforms a larger autoregressive GAN-based system, both in terms of audio quality and synthesis speed. In particular, MLP Singer achieves a real-time factor of up to 200 and 3400 on CPUs and GPUs respectively, enabling order of magnitude faster generation on both environments.

## Citation

Please cite this work as follows.

```bibtex
@misc{tae2021mlp,
      title={MLP Singer: Towards Rapid Parallel Korean Singing Voice Synthesis}, 
      author={Jaesung Tae and Hyeongju Kim and Younggun Lee},
      year={2021},
}
```

## Quickstart

1. Clone the repository including the git submodule.

   ```bash
   git clone --recurse-submodules https://github.com/neosapience/mlp-singer.git
   ```

2.  Install package requirements.

   ```bash
   cd mlp-singer
   pip install -r requirements.txt
   ```

3. To generate audio files with the trained model checkpoint, [download](https://drive.google.com/drive/folders/1YuOoV3lO2-Hhn1F2HJ2aQ4S0LC1JdKLd) the HiFi-GAN checkpoint along with its configuration file and place them in `hifi-gan`. 

4. Run inference using the following command. Generated audio samples are saved in the `samples` directory by default.

   ```bash
   python inference.py --checkpoint_path checkpoints/default/model.pt
   ```

## Dataset

We used the [Children Song Dataset](https://github.com/emotiontts/emotiontts_open_db/tree/master/Dataset/CSD), an open-source singing voice dataset comprised of 100 annotated Korean and English children songs sung by a single professional singer. We used only the Korean subset of the dataset to train the model.

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

This will read the default configuration file located in `configs/model.json` to initialize the model. Alternatively, you can also create a new configuration and train the model via

```
python train.py --config_path PATH/TO/CONFIG.json
```

Running this command will create a folder under the `checkpoints` directory according to the `name` field specified in the configuration file.

You can also continue training from a checkpoint. For example, to resume training from the provided pretrained model checkpoint, run

```
python train.py --checkpoint_path /checkpoints/default/model.pt
```

Unless a `--config_path` flag is explicitly provided, the script will read `config.json` in the checkpoint directory. In both cases, model checkpoints will be saved regularly according to the interval defined in the configuration file. 

## Inference

MLP Singer produces mel-spectrograms, which are then fed into a neural vocoder to generate raw waveforms. This repository uses [HiFi-GAN](https://github.com/jik876/hifi-gan) as the vocoder backend, but you can also plug other vocoders like [WaveGlow](https://github.com/NVIDIA/waveglow). To generate samples, run

```
python inference.py --checkpoint_path PATH/TO/CHECKPOINT.pt --song little_star
```

This will create `.wav` samples in the `samples` directory, and save mel-spectrogram files as `.npy` files in `hifi-gan/test_mel_dirs`. 

You can also specify any song you want to perform inference on, as long as the song is present in `data/raw`. The argument to the `--song` flag should match the title of the song as it is saved in `data/raw`.  

## Note

For demo and internal experiments, we used a variant of HiFi-GAN that used different mel-spectrogram configurations. As such, the provided checkpoint for MLP Singer is different from the one referred to in the paper. Moreover, the vocoder used in the demo was further fine-tuned on the Children's Song Dataset.

## Acknowledgements

This implementation was inspired by the following repositories.

* [Tacotron2](https://github.com/NVIDIA/tacotron2)
* [BEGANSing](https://github.com/SoonbeomChoi/BEGANSing)

## License

Released under the [MIT License](./LICENSE).
