#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler

from data_handling.caption_dataset import AcousticFactorStore, AudioCaptionDataset


class AudioCaptionDataModule:
    def __init__(self, config: dict, dataset: str):
        super().__init__()
        audio_config = config["audio_args"]
        factor_config = config.get("factor_args", {})
        factor_store = None

        train_set = AudioCaptionDataset(audio_config, dataset, split="train")
        if factor_config.get("enabled", False):
            factor_store = AcousticFactorStore(
                factor_config["path"],
                factor_config["names"],
            )
            if factor_store.window_ms != int(factor_config["window_ms"]):
                raise ValueError("Factor window_ms does not match the archive.")
            if factor_store.hop_ms != int(factor_config["hop_ms"]):
                raise ValueError("Factor hop_ms does not match the archive.")
            mean, std = factor_store.fit_normalization(train_set.wav_paths)
            factor_config["normalization_mean"] = mean.tolist()
            factor_config["normalization_std"] = std.tolist()
            train_set.factor_store = factor_store

        self.train_set = train_set
        self.val_set = AudioCaptionDataset(
            audio_config,
            dataset,
            split="val",
            factor_store=factor_store,
        )
        self.test_set = AudioCaptionDataset(
            audio_config,
            dataset,
            split="test",
            factor_store=factor_store,
        )
        self.batch_size = config["data_args"]["batch_size"]
        self.num_workers = config["data_args"]["num_workers"]

    def _get_sampler(self, dataset, shuffle, is_distributed, num_tasks, global_rank):
        if not is_distributed:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=shuffle,
        )

    def train_dataloader(
        self,
        is_distributed=False,
        num_tasks=0,
        global_rank=0,
    ):
        sampler = self._get_sampler(
            self.train_set,
            True,
            is_distributed,
            num_tasks,
            global_rank,
        )
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )


def collate_fn(batch):
    has_factors = len(batch[0]) == 5
    max_audio_length = max(item[0].shape[-1] for item in batch)
    max_factor_length = (
        max(item[4].shape[-1] for item in batch) if has_factors else 0
    )
    wav_list = []
    text_list = []
    audio_name_list = []
    audio_idx_list = []
    factor_list = []
    factor_mask_list = []

    for item in batch:
        waveform, text, audio_name, audio_idx = item[:4]
        waveform = F.pad(
            waveform,
            [0, max_audio_length - waveform.shape[-1]],
            "constant",
            0.0,
        )
        wav_list.append(waveform)
        text_list.append(text)
        audio_name_list.append(audio_name)
        audio_idx_list.append(audio_idx)

        if has_factors:
            factors = item[4]
            factor_length = factors.shape[-1]
            factor_list.append(
                F.pad(
                    factors,
                    [0, max_factor_length - factor_length],
                    "constant",
                    0.0,
                )
            )
            mask = torch.zeros(max_factor_length, dtype=torch.bool)
            mask[:factor_length] = True
            factor_mask_list.append(mask)

    output = (
        torch.stack(wav_list),
        text_list,
        audio_name_list,
        Tensor(audio_idx_list).long(),
    )
    if not has_factors:
        return output
    return (
        *output,
        torch.stack(factor_list),
        torch.stack(factor_mask_list),
    )
