#!/usr/bin/env python3
# coding: utf-8

import json
import random

import librosa
import torch
from torch.utils.data import Dataset

from data_handling.text_transform import text_preprocess


class AudioCaptionDataset(Dataset):
    def __init__(
        self,
        audio_config: dict,
        dataset: str = "AudioCaps",
        split: str = "train",
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
        return torch.from_numpy(waveform), caption, audio_name, audio_idx
