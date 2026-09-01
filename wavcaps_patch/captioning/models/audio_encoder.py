#!/usr/bin/env python3
# coding: utf-8
"""WavCaps audio encoder with strict frozen-encoder semantics."""

import torch
import yaml
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from models.audio_encoder_config import AudioEncoderConfig
from models.cnns import Cnn10, Cnn14, ResNet38
from models.htsat import HTSAT_Swin_Transformer


class AudioEncoderModel(PreTrainedModel):
    config_class = AudioEncoderConfig

    def __init__(self, config):
        super().__init__(config)

        if config.model_arch == "cnn":
            if config.model_name == "ResNet38":
                self.audio_enc = ResNet38(config)
            elif config.model_name == "Cnn14":
                self.audio_enc = Cnn14(config)
            elif config.model_name == "Cnn10":
                self.audio_enc = Cnn10(config)

            if config.pretrained:
                pretrained_cnn = torch.load(
                    f"pretrained_models/audio_encoder/{config.model_name}.pth"
                )["model"]
                updated = self.audio_enc.state_dict().copy()
                trained_names = [
                    name
                    for name in pretrained_cnn
                    if not (
                        "fc" in name
                        or name.startswith("spec")
                        or name.startswith("logmel")
                    )
                ]
                for name in trained_names:
                    updated[name] = pretrained_cnn[name]
                self.audio_enc.load_state_dict(updated)
            self.audio_width = 2048
        elif config.model_arch == "transformer":
            self.audio_enc = HTSAT_Swin_Transformer(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=527,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                config=config,
            )
            if config.pretrained:
                checkpoint = torch.load(
                    "pretrained_models/audio_encoder/HTSAT.ckpt",
                    map_location="cpu",
                )["state_dict"]
                for key in list(checkpoint):
                    if key.startswith("sed_model") and (
                        "spectrogram_extractor" not in key
                        and "logmel_extractor" not in key
                    ):
                        checkpoint[key[10:]] = checkpoint.pop(key)
                self.audio_enc.load_state_dict(checkpoint, strict=False)
            self.audio_width = 768
        else:
            raise NotImplementedError("No such audio encoder network.")

        self.freeze_audio_encoder = bool(config.freeze)
        if self.freeze_audio_encoder:
            for parameter in self.audio_enc.parameters():
                parameter.requires_grad = False
            self.audio_enc.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_audio_encoder:
            self.audio_enc.eval()
        return self

    def forward(
        self,
        input_ids,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        audio_embeds = self.audio_enc(input_ids)
        if not return_dict:
            return (audio_embeds,)
        return BaseModelOutput(audio_embeds, None, None)


if __name__ == "__main__":
    import os

    os.chdir("../")
    with open("settings/settings.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    config = AudioEncoderConfig(
        **config["audio_encoder_args"],
        audio_args=config["audio_args"],
    )
    print(AudioEncoderModel(config))
