#!/usr/bin/env python3
# coding: utf-8

import json
import random
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from data_handling.text_transform import text_preprocess


def _audio_key(name):
    return unicodedata.normalize("NFKC", Path(str(name)).name).strip().casefold()


class AcousticFactorStore:
    def __init__(self, path, expected_factor_names):
        with np.load(path, allow_pickle=False) as archive:
            self.values = archive["factor_values"].astype(np.float32)
            self.offsets = archive["offsets"].astype(np.int64)
            self.audio_files = archive["audio_files"].astype(str)
            self.factor_names = tuple(archive["factor_names"].astype(str))
            self.window_ms = int(archive["window_ms"])
            self.hop_ms = int(archive["hop_ms"])

        expected_factor_names = tuple(expected_factor_names)
        if self.factor_names != expected_factor_names:
            raise ValueError(
                f"Factor order mismatch: {self.factor_names} != "
                f"{expected_factor_names}"
            )
        if self.values.shape[0] != len(self.factor_names):
            raise ValueError("factor_values does not match factor_names.")
        if len(self.offsets) != len(self.audio_files) + 1:
            raise ValueError("Invalid acoustic-factor offsets.")

        self.index_by_name = {
            _audio_key(name): index for index, name in enumerate(self.audio_files)
        }
        if len(self.index_by_name) != len(self.audio_files):
            raise ValueError("Duplicate normalized audio names in factor archive.")
        self.mean = None
        self.std = None

    def _slice(self, audio_name):
        try:
            index = self.index_by_name[_audio_key(audio_name)]
        except KeyError as error:
            raise KeyError(f"Missing acoustic factors for {audio_name}.") from error
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.values[:, start:end]

    def fit_normalization(self, audio_names):
        unique_names = sorted({_audio_key(name) for name in audio_names})
        factor_sum = np.zeros(len(self.factor_names), dtype=np.float64)
        factor_square_sum = np.zeros_like(factor_sum)
        frame_count = 0
        for name in unique_names:
            values = self._slice(name).astype(np.float64)
            factor_sum += values.sum(axis=1)
            factor_square_sum += np.square(values).sum(axis=1)
            frame_count += values.shape[1]
        if frame_count == 0:
            raise ValueError("Cannot fit factor normalization without frames.")
        mean = factor_sum / frame_count
        variance = np.maximum(factor_square_sum / frame_count - mean**2, 0.0)
        std = np.sqrt(variance)
        std[std < 1e-6] = 1.0
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        return self.mean.copy(), self.std.copy()

    def get(self, audio_name):
        if self.mean is None or self.std is None:
            raise RuntimeError("Factor normalization has not been fitted.")
        values = self._slice(audio_name)
        return (values - self.mean[:, None]) / self.std[:, None]


class AudioCaptionDataset(Dataset):
    def __init__(
        self,
        audio_config: dict,
        dataset: str = "AudioCaps",
        split: str = "train",
        factor_store=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.sr = audio_config["sr"]
        json_path = f"data/{dataset}/json_files/{split}.json"
        self.max_length = (
            audio_config["max_length"] * self.sr
            if audio_config["max_length"] != 0
            else 0
        )
        self.factor_store = factor_store

        with open(json_path, "r") as stream:
            json_obj = json.load(stream)["data"]

        if self.dataset == "AudioCaps" and split == "train":
            self.num_captions_per_audio = 1
            self.captions = [item["caption"] for item in json_obj]
            self.wav_paths = [item["audio"] for item in json_obj]
        elif split == "train":
            self.num_captions_per_audio = 5
            self.captions = [
                item[f"caption_{index}"]
                for item in json_obj
                for index in range(1, 6)
            ]
            self.wav_paths = [
                item["audio"]
                for item in json_obj
                for _ in range(1, 6)
            ]
        else:
            self.num_captions_per_audio = 5
            self.captions = [
                [item[f"caption_{index}"] for index in range(1, 6)]
                for item in json_obj
            ]
            self.wav_paths = [item["audio"] for item in json_obj]

    def __len__(self):
        return len(self.wav_paths)

    def _prepare_caption(self, caption):
        if self.dataset == "OnomaCap":
            return caption.strip()
        return text_preprocess(caption)

    def __getitem__(self, index):
        audio_idx = index if self.split in ["val", "test"] else (
            index // self.num_captions_per_audio
        )
        audio_name = self.wav_paths[index].split("/")[-1]
        wav_path = self.wav_paths[index]

        waveform, _ = librosa.load(wav_path, sr=self.sr, mono=True)
        if self.max_length != 0 and waveform.shape[-1] > self.max_length:
            max_start = waveform.shape[-1] - self.max_length
            start = random.randint(0, max_start)
            waveform = waveform[start:start + self.max_length]

        captions = self.captions[index]
        if isinstance(captions, list):
            caption = [self._prepare_caption(item) for item in captions]
        else:
            caption = self._prepare_caption(captions)
        output = (torch.from_numpy(waveform), caption, audio_name, audio_idx)
        if self.factor_store is None:
            return output
        factors = torch.from_numpy(
            np.ascontiguousarray(self.factor_store.get(audio_name))
        )
        return (*output, factors)
