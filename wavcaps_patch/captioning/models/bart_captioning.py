#!/usr/bin/env python3
# coding: utf-8

"""HTSAT-BART captioning model with canonical-Jamo targets."""

import copy
import math

import torch
import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartDecoderLayer

from models.audio_encoder import AudioEncoderModel
from models.audio_encoder_config import AudioEncoderConfig
from tools.jamo_preprocessing import (
    CHOSEONG,
    JAMO_VOCAB,
    JONGSEONG,
    JUNGSEONG,
    jamo_to_hangul_caption,
)


class DualCrossAttentionBartDecoderLayer(BartDecoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.factor_attn = copy.deepcopy(self.encoder_attn)
        self.factor_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.audio_token_count = None
        self.last_factor_attention = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=True,
    ):
        if encoder_hidden_states is None or self.audio_token_count is None:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        audio_memory = encoder_hidden_states[:, :self.audio_token_count]
        factor_memory = encoder_hidden_states[:, self.audio_token_count:]
        audio_mask = (
            encoder_attention_mask[..., :self.audio_token_count]
            if encoder_attention_mask is not None
            else None
        )
        factor_mask = (
            encoder_attention_mask[..., self.audio_token_count:]
            if encoder_attention_mask is not None
            else None
        )
        outputs = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=audio_memory,
            encoder_attention_mask=audio_mask,
            layer_head_mask=layer_head_mask,
            cross_attn_layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = outputs[0]
        residual = hidden_states
        factor_context, factor_attention, _ = self.factor_attn(
            hidden_states=hidden_states,
            key_value_states=factor_memory,
            attention_mask=factor_mask,
            layer_head_mask=None,
            output_attentions=output_attentions,
        )
        factor_context = nn.functional.dropout(
            factor_context,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = self.factor_attn_layer_norm(residual + factor_context)
        self.last_factor_attention = factor_attention
        return (hidden_states,) + outputs[1:]


class BartCaptionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        encoder_config = AudioEncoderConfig(
            **config["audio_encoder_args"],
            audio_args=config["audio_args"],
        )
        self.encoder = AudioEncoderModel(encoder_config)

        decoder_name = config["text_decoder_args"]["name"]
        decoder_pretrained = config["text_decoder_args"]["pretrained"]
        self.tokenizer = BartTokenizer.from_pretrained(decoder_name)
        if decoder_pretrained:
            self.decoder = BartForConditionalGeneration.from_pretrained(decoder_name)
        else:
            bart_config = BartConfig.from_pretrained(decoder_name)
            self.decoder = BartForConditionalGeneration(bart_config)

        added_count = self.tokenizer.add_tokens(list(JAMO_VOCAB))
        if added_count != len(JAMO_VOCAB):
            raise RuntimeError(
                f"Expected to add {len(JAMO_VOCAB)} Jamo tokens, added {added_count}."
            )
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.jamo_to_id = {
            token: self.tokenizer.convert_tokens_to_ids(token)
            for token in JAMO_VOCAB
        }
        if len(set(self.jamo_to_id.values())) != len(JAMO_VOCAB):
            raise RuntimeError("Canonical Jamo tokens do not have unique token IDs.")
        self.id_to_jamo = {token_id: token for token, token_id in self.jamo_to_id.items()}
        self.choseong_ids = frozenset(self.jamo_to_id[token] for token in CHOSEONG)
        self.jungseong_ids = frozenset(self.jamo_to_id[token] for token in JUNGSEONG)
        self.jongseong_ids = frozenset(self.jamo_to_id[token] for token in JONGSEONG)

        self.max_text_length = int(config["text_decoder_args"].get("max_length", 80))
        self._active_generation_max_length = self.max_text_length
        self.enc_to_dec_proj = nn.Linear(
            encoder_config.hidden_size,
            self.decoder.config.hidden_size,
        )
        factor_config = config.get("factor_args", {})
        self.factor_enabled = bool(factor_config.get("enabled", False))
        self.factor_names = tuple(factor_config.get("names", ()))
        self.factor_only_training = False
        self.effective_audio_modality_dropout = 0.0
        if self.factor_enabled:
            if not self.factor_names:
                raise ValueError("factor_args.names must not be empty.")
            hidden_size = self.decoder.config.hidden_size
            self.factor_value_projection = nn.Linear(1, hidden_size)
            self.factor_embedding = nn.Embedding(
                len(self.factor_names),
                hidden_size,
            )
            self.factor_layer_norm = nn.LayerNorm(hidden_size)
            self.factor_dropout = nn.Dropout(
                float(factor_config.get("dropout", 0.1))
            )
            self.audio_modality_dropout = float(
                factor_config.get("audio_modality_dropout", 0.0)
            )
            if not 0.0 <= self.audio_modality_dropout < 1.0:
                raise ValueError("audio_modality_dropout must be in [0, 1).")
            self.factor_only_epochs = int(factor_config.get("factor_only_epochs", 0))
            self.audio_dropout_transition_epochs = int(
                factor_config.get("audio_dropout_transition_epochs", 0)
            )
            if self.factor_only_epochs < 0:
                raise ValueError("factor_only_epochs must be non-negative.")
            if self.audio_dropout_transition_epochs < 0:
                raise ValueError(
                    "audio_dropout_transition_epochs must be non-negative."
                )
            self.effective_audio_modality_dropout = self.audio_modality_dropout
            decoder_layers = self.decoder.model.decoder.layers
            for index, layer in enumerate(decoder_layers):
                dual_layer = DualCrossAttentionBartDecoderLayer(self.decoder.config)
                dual_layer.load_state_dict(layer.state_dict(), strict=False)
                dual_layer.factor_attn.load_state_dict(layer.encoder_attn.state_dict())
                dual_layer.factor_attn_layer_norm.load_state_dict(
                    layer.encoder_attn_layer_norm.state_dict()
                )
                decoder_layers[index] = dual_layer
            init_std = float(self.decoder.config.init_std)
            nn.init.normal_(
                self.factor_value_projection.weight,
                mean=0.0,
                std=init_std,
            )
            nn.init.zeros_(self.factor_value_projection.bias)
            nn.init.normal_(
                self.factor_embedding.weight,
                mean=0.0,
                std=init_std,
            )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    def initialize_factor_attention_from_audio(self):
        if not self.factor_enabled:
            return
        for layer in self.decoder.model.decoder.layers:
            layer.factor_attn.load_state_dict(layer.encoder_attn.state_dict())
            layer.factor_attn_layer_norm.load_state_dict(
                layer.encoder_attn_layer_norm.state_dict()
            )

    @staticmethod
    def _is_factor_parameter(name):
        return (
            name.startswith("factor_")
            or ".factor_attn." in name
            or ".factor_attn_layer_norm." in name
        )

    def set_training_epoch(self, epoch):
        if not self.factor_enabled:
            return {
                "phase": "joint",
                "factor_only": False,
                "audio_modality_dropout": 0.0,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in self.parameters()
                ),
            }
        if epoch < 1:
            raise ValueError("Training epoch must be at least 1.")

        self.factor_only_training = epoch <= self.factor_only_epochs
        if self.factor_only_training:
            phase = "factor_only"
            effective_dropout = 1.0
        else:
            transition_step = epoch - self.factor_only_epochs
            if transition_step <= self.audio_dropout_transition_epochs:
                phase = "audio_dropout_transition"
                progress = transition_step / self.audio_dropout_transition_epochs
                effective_dropout = 1.0 - progress * (
                    1.0 - self.audio_modality_dropout
                )
            else:
                phase = "joint"
                effective_dropout = self.audio_modality_dropout

        self.effective_audio_modality_dropout = effective_dropout
        for name, parameter in self.named_parameters():
            parameter.requires_grad = (
                self._is_factor_parameter(name)
                if self.factor_only_training
                else True
            )
        return {
            "phase": phase,
            "factor_only": self.factor_only_training,
            "audio_modality_dropout": effective_dropout,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }

    @property
    def device(self):
        return next(self.parameters()).device

    def shift_tokens_right(self, input_ids, pad_token_id, decoder_start_token_id):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id
        if pad_token_id is None:
            raise ValueError("decoder pad_token_id must be defined.")
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def forward_encoder(self, audios):
        outputs = self.encoder(audios)
        return self.enc_to_dec_proj(outputs.last_hidden_state)

    def _sinusoidal_time_encoding(self, length, hidden_size, device, dtype):
        position = torch.arange(length, device=device, dtype=torch.float32)
        frequency = torch.exp(
            torch.arange(0, hidden_size, 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / hidden_size)
        )
        encoding = torch.zeros(
            length,
            hidden_size,
            device=device,
            dtype=torch.float32,
        )
        encoding[:, 0::2] = torch.sin(position[:, None] * frequency[None, :])
        encoding[:, 1::2] = torch.cos(position[:, None] * frequency[None, :])
        return encoding.to(dtype=dtype)

    def _encode_factor_tokens(self, factors, factor_mask):
        if not self.factor_enabled:
            raise RuntimeError("Acoustic factor conditioning is disabled.")
        if factors is None or factor_mask is None:
            raise ValueError("factors and factor_mask are required.")
        if factors.ndim != 3:
            raise ValueError(f"Expected factors [B, F, T], got {factors.shape}.")
        batch_size, factor_count, time_length = factors.shape
        if factor_count != len(self.factor_names):
            raise ValueError(
                f"Expected {len(self.factor_names)} factors, got {factor_count}."
            )
        if factor_mask.shape != (batch_size, time_length):
            raise ValueError(
                f"Expected factor_mask {(batch_size, time_length)}, "
                f"got {factor_mask.shape}."
            )

        values = factors.transpose(1, 2).unsqueeze(-1)
        value_embedding = self.factor_value_projection(values)
        factor_ids = torch.arange(factor_count, device=factors.device)
        factor_embedding = self.factor_embedding(factor_ids)[None, None, :, :]
        time_embedding = self._sinusoidal_time_encoding(
            time_length,
            value_embedding.size(-1),
            factors.device,
            value_embedding.dtype,
        )[None, :, None, :]
        tokens = self.factor_layer_norm(
            value_embedding + factor_embedding + time_embedding
        )
        tokens = self.factor_dropout(tokens)
        flat_tokens = tokens.reshape(batch_size, time_length * factor_count, -1)
        flat_mask = (
            factor_mask.bool()
            .unsqueeze(-1)
            .expand(batch_size, time_length, factor_count)
            .reshape(batch_size, time_length * factor_count)
        )
        return flat_tokens, flat_mask

    def encode_memory(self, audios, factors=None, factor_mask=None):
        audio_embeds = self.forward_encoder(audios)
        audio_memory = self.decoder.model.encoder(
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=audio_embeds,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )["last_hidden_state"]
        batch_size, audio_token_count, _ = audio_memory.shape
        if self.factor_enabled and self.training:
            if self.factor_only_training:
                audio_memory = torch.zeros_like(audio_memory)
            elif self.effective_audio_modality_dropout > 0.0:
                drop_audio = torch.rand(
                    batch_size,
                    1,
                    1,
                    device=audio_memory.device,
                ) < self.effective_audio_modality_dropout
                audio_memory = audio_memory.masked_fill(drop_audio, 0.0)
        audio_mask = torch.ones(
            batch_size,
            audio_token_count,
            dtype=torch.bool,
            device=audio_memory.device,
        )

        if self.factor_enabled:
            factor_tokens, flat_factor_mask = self._encode_factor_tokens(
                factors,
                factor_mask,
            )
            memory = torch.cat([audio_memory, factor_tokens], dim=1)
            memory_mask = torch.cat([audio_mask, flat_factor_mask], dim=1)
            factor_time_length = factors.shape[-1]
            for layer in self.decoder.model.decoder.layers:
                layer.audio_token_count = audio_token_count
        else:
            memory = audio_memory
            memory_mask = audio_mask
            factor_time_length = 0

        return (
            BaseModelOutput(last_hidden_state=memory),
            memory_mask,
            {
                "audio_token_count": audio_token_count,
                "factor_time_length": factor_time_length,
            },
        )

    def encode_jamo_batch(self, captions):
        encoded = []
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        for caption in captions:
            tokens = caption.split()
            if not tokens:
                raise ValueError("Jamo caption must not be empty.")
            unknown = [token for token in tokens if token not in self.jamo_to_id]
            if unknown:
                raise ValueError(f"Unsupported Jamo tokens: {unknown!r}")
            token_ids = [bos_id] + [self.jamo_to_id[token] for token in tokens] + [eos_id]
            if len(token_ids) > self.max_text_length:
                raise ValueError(
                    f"Jamo target length {len(token_ids)} exceeds "
                    f"max_length={self.max_text_length}."
                )
            encoded.append(token_ids)

        batch_length = max(len(item) for item in encoded)
        input_ids = torch.full(
            (len(encoded), batch_length),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, token_ids in enumerate(encoded):
            length = len(token_ids)
            input_ids[index, :length] = torch.tensor(
                token_ids,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask[index, :length] = 1
        return input_ids, attention_mask

    def _prepare_decoder_batch(self, text):
        input_ids, attention_mask = self.encode_jamo_batch(text)
        decoder_targets = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id,
            -100,
        )
        decoder_input_ids = self.shift_tokens_right(
            decoder_targets,
            self.decoder.config.pad_token_id,
            self.decoder.config.decoder_start_token_id,
        )
        return input_ids, attention_mask, decoder_targets, decoder_input_ids

    def forward_decoder(self, text, encoder_outputs, encoder_attention_mask):
        _, attention_mask, decoder_targets, decoder_input_ids = (
            self._prepare_decoder_batch(text)
        )

        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=attention_mask,
            inputs_embeds=None,
            labels=None,
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )
        lm_logits = decoder_outputs["logits"]
        return self.loss_fct(
            lm_logits.reshape(-1, lm_logits.size(-1)),
            decoder_targets.reshape(-1),
        )

    def forward(self, audio, text, factors=None, factor_mask=None):
        encoder_outputs, encoder_attention_mask, _ = self.encode_memory(
            audio,
            factors,
            factor_mask,
        )
        return self.forward_decoder(
            text,
            encoder_outputs,
            encoder_attention_mask,
        )

    @torch.no_grad()
    def teacher_forced_factor_attention(
        self,
        samples,
        text,
        factors,
        factor_mask,
        layer_index=-1,
    ):
        if not self.factor_enabled:
            raise RuntimeError("Acoustic factor conditioning is disabled.")
        encoder_outputs, encoder_attention_mask, metadata = self.encode_memory(
            samples,
            factors,
            factor_mask,
        )
        input_ids, attention_mask, _, decoder_input_ids = (
            self._prepare_decoder_batch(text)
        )
        outputs = self.decoder(
            input_ids=None,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=True,
            return_dict=True,
        )
        if outputs.cross_attentions is None:
            raise RuntimeError("BART did not return decoder cross-attention weights.")
        decoder_layer = self.decoder.model.decoder.layers[layer_index]
        if decoder_layer.last_factor_attention is None:
            raise RuntimeError("BART did not return factor cross-attention weights.")
        factor_source_attention = decoder_layer.last_factor_attention.mean(dim=1)
        time_length = metadata["factor_time_length"]
        factor_count = len(self.factor_names)
        expected_factor_tokens = time_length * factor_count
        if factor_source_attention.size(-1) != expected_factor_tokens:
            raise RuntimeError(
                "Factor attention/source layout mismatch: "
                f"{factor_source_attention.size(-1)} != {expected_factor_tokens}."
            )
        factor_attention = factor_source_attention.reshape(
            factor_source_attention.size(0),
            factor_source_attention.size(1),
            time_length,
            factor_count,
        )
        factor_attention_mass = factor_attention.sum(dim=2)
        factor_attention_by_time = factor_attention.permute(0, 1, 3, 2).contiguous()
        target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.id_to_jamo:
            target_mask |= input_ids == token_id
        target_mask &= attention_mask.bool()
        return {
            "target_ids": input_ids,
            "target_mask": target_mask,
            "factor_attention_mass": factor_attention_mass,
            "factor_attention_by_time": factor_attention_by_time,
            "factor_time_mask": factor_mask.bool(),
            "layer_index": layer_index,
        }

    def _allowed_next_tokens(self, _batch_id, input_ids):
        decoder_start_id = self.decoder.config.decoder_start_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        prefix = [
            token_id
            for token_id in input_ids.tolist()
            if token_id not in {decoder_start_id, pad_id}
        ]
        if not prefix:
            return [bos_id]
        if prefix[0] != bos_id:
            raise ValueError(f"Generated prefix does not start with BOS: {prefix!r}")

        payload = prefix[1:]
        if not payload:
            return sorted(self.choseong_ids)
        if payload[-1] == eos_id:
            return [eos_id]

        state = "choseong"
        completed_syllable = False
        for token_id in payload:
            if state == "choseong":
                if token_id not in self.choseong_ids:
                    raise ValueError(f"Expected choseong token ID, got {token_id}.")
                state = "jungseong"
            elif state == "jungseong":
                if token_id not in self.jungseong_ids:
                    raise ValueError(f"Expected jungseong token ID, got {token_id}.")
                state = "optional_jongseong"
                completed_syllable = True
            elif token_id in self.jongseong_ids:
                state = "choseong"
            elif token_id in self.choseong_ids:
                state = "jungseong"
            else:
                raise ValueError(f"Invalid Jamo token ID in prefix: {token_id}.")

        if state == "jungseong":
            return sorted(self.jungseong_ids)
        if state == "optional_jongseong":
            remaining = self._active_generation_max_length - len(input_ids)
            if remaining <= 1:
                return [eos_id]
            if remaining == 2:
                return sorted(self.jongseong_ids | {eos_id})
            return sorted(self.choseong_ids | self.jongseong_ids | {eos_id})
        allowed = set(self.choseong_ids)
        if completed_syllable:
            allowed.add(eos_id)
        return sorted(allowed)

    def _decode_generated_ids(self, outputs):
        ignored_ids = {
            self.decoder.config.decoder_start_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        }
        captions = []
        for output in outputs.tolist():
            tokens = [
                self.id_to_jamo[token_id]
                for token_id in output
                if token_id not in ignored_ids and token_id in self.id_to_jamo
            ]
            if not tokens:
                raise ValueError("Generation produced no Jamo tokens.")
            captions.append(jamo_to_hangul_caption(" ".join(tokens)))
        return captions

    def generate(
        self,
        samples,
        factors=None,
        factor_mask=None,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=None,
        min_length=4,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        if max_length is None:
            max_length = self.max_text_length
        self._active_generation_max_length = max_length

        encoder_outputs, encoder_attention_mask, _ = self.encode_memory(
            samples,
            factors,
            factor_mask,
        )
        decoder_input_ids = torch.full(
            (encoder_outputs["last_hidden_state"].size(0), 1),
            self.decoder.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )
        decoder_attention_mask = torch.ones_like(decoder_input_ids)

        generation_args = dict(
            input_ids=None,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            prefix_allowed_tokens_fn=self._allowed_next_tokens,
        )
        if use_nucleus_sampling:
            outputs = self.decoder.generate(
                **generation_args,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
            )
        else:
            outputs = self.decoder.generate(
                **generation_args,
                num_beams=num_beams,
            )
        return self._decode_generated_ids(outputs)
