import warnings

import mido
import torch

from .dsp import TacotronSTFT
from .g2p import encode

warnings.simplefilter(action="ignore", category=UserWarning)


class Preprocessor:
    def __init__(self, config):
        self.stft = TacotronSTFT(
            config.n_fft,
            config.hop_length,
            config.win_length,
            config.n_mel_channels,
            config.sampling_rate,
            config.mel_fmin,
            config.mel_fmax,
            config.max_wav_value,
        )
        self.frame_len = self.stft.frame_len
        self.consonant_emphasis = config.consonant_emphasis
        self.min_note = config.min_note

    def __call__(self, midi_path, text_path, wav_path):
        mel = self.preprocess_audio(wav_path)
        total_length = len(mel)
        note_sequence = get_note_sequence(midi_path, self.frame_len)
        text = get_phonemes(text_path)
        notes, phonemes = align(
            text, note_sequence, total_length, self.consonant_emphasis, self.min_note
        )
        return notes, phonemes, mel

    def preprocess_audio(self, wav_path: str):
        wav = self.stft.load_wav(wav_path)
        mel = self.stft.mel_spectrogram(wav)
        return mel

    def prepare_inference(self, midi_path, text_path):
        midi_file = mido.MidiFile(midi_path)
        note_sequence = get_note_sequence(midi_path, self.frame_len)
        total_length = round(1000 * midi_file.length / self.frame_len)
        text = get_phonemes(text_path)
        notes, phonemes = align(
            text, note_sequence, total_length, self.consonant_emphasis, self.min_note
        )
        return notes, phonemes


def get_phonemes(text_path, split_newline=False):
    with open(text_path) as f:
        text = f.read().splitlines()
    phonemes = []
    for line in text:
        line = line.strip()
        if line:
            encoded = encode(line)
            if split_newline:
                phonemes.append((encoded, line))
            else:
                phonemes.extend(encoded)
    return phonemes


def align(
    phonemes: list,
    note_sequence: list,
    total_length: int,
    consonant_emphasis: int,
    min_note: int,
):
    """align phonemes with notes and expand to mel-spectrogram length"""
    assert len(phonemes) == len(note_sequence), print("midi lyrics mismatch")
    expanded_text = torch.zeros(total_length)
    expanded_notes = torch.zeros(total_length)
    for (phoneme, event) in zip(phonemes, note_sequence):
        # print(phoneme, event[0])
        note, original_start, original_end = event
        note -= min_note
        assert note > 0
        expanded_notes[original_start:original_end] = note
        durations = get_phoneme_duration(
            phoneme, original_end - original_start, consonant_emphasis
        )
        pointer = 0
        start = original_start
        for idx in (phoneme.onset, phoneme.nucleus, phoneme.coda):
            if idx is not None:
                end = start + durations[pointer]
                expanded_text[start:end] = idx
                start = end
                pointer += 1
        assert pointer == len(durations)
        assert original_end == end == original_start + sum(durations)
    expanded_text = expanded_text.to(int)
    expanded_notes = expanded_notes.to(int)
    return expanded_notes, expanded_text


def get_phoneme_duration(phone, note_duration, length_c):
    # from https://github.com/SoonbeomChoi/BEGANSing/tree/master/preprocess.py#L24
    duration = []
    if note_duration < phone.num():
        length_c = 0
    elif note_duration <= phone.num() * length_c:
        length_c = max(note_duration // phone.num() - 1, 1)
    if phone.onset is not None:
        duration.append(length_c)
    if phone.nucleus is not None:
        length_v = note_duration - (phone.num() - 1) * length_c
        duration.append(length_v)
    if phone.coda is not None:
        duration.append(length_c)
    return duration


def get_note_sequence(
    midi_path: str, frame_len: float, count_frames: bool = True
) -> list:
    """generate a list containing midi events of form (note, start, end)"""
    pointer = 0
    time_keeper = {}
    note_sequence = []
    midi_file = mido.MidiFile(midi_path)
    track = find_track(midi_file.tracks)
    tempo = get_tempo(track)
    unit = tick2milisecond(tempo, midi_file.ticks_per_beat)
    for message in track:
        pointer += message.time
        event = message.type
        if event == "note_on" and message.velocity != 0:
            time_keeper[message.note] = pointer
        elif event == "note_off" or (event == "note_on" and message.velocity == 0):
            note = message.note
            start = time_keeper[note] * unit
            end = pointer * unit
            if count_frames:
                start = round(start / frame_len)
                end = round(end / frame_len)
            note_sequence.append((note, start, end))
            del time_keeper[note]
    return note_sequence


def find_track(tracks):
    max_idx = 0
    max_num_messages = 0
    for i, track in enumerate(tracks):
        num_messages = len(track)
        if num_messages > max_num_messages:
            max_num_messages = num_messages
            max_idx = i
    return tracks[max_idx]


def get_tempo(track) -> int:
    """find track tempo, in microseconds per beat"""
    for message in track:
        if message.type == "set_tempo":
            return message.tempo
    return 500000


def tick2milisecond(tempo: int, ticks_per_beat: int) -> float:
    """calculate how many miliseconds are in one tick"""
    return tempo / (1000 * ticks_per_beat)
