#!/usr/bin/env python3
# coding: utf-8

"""HTSAT-BART captioning model with canonical-Jamo targets."""

import torch
import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer

from models.audio_encoder import AudioEncoderModel
from models.audio_encoder_config import AudioEncoderConfig
from tools.jamo_preprocessing import (
    CHOSEONG,
    JAMO_VOCAB,
    JONGSEONG,
    JUNGSEONG,
    jamo_to_hangul_caption,
)


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
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

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

    def forward_decoder(self, text, encoder_outputs):
        encoder_outputs = self.decoder.model.encoder(
            input_ids=None,
            inputs_embeds=encoder_outputs,
            return_dict=True,
        )["last_hidden_state"]

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

        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=attention_mask,
            inputs_embeds=None,
            labels=None,
            encoder_outputs=(encoder_outputs,),
            return_dict=True,
        )
        lm_logits = decoder_outputs["logits"]
        return self.loss_fct(
            lm_logits.reshape(-1, lm_logits.size(-1)),
            decoder_targets.reshape(-1),
        )

    def forward(self, audio, text):
        return self.forward_decoder(text, self.forward_encoder(audio))

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

        audio_embeds = self.forward_encoder(samples)
        encoder_outputs = self.decoder.model.encoder(
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=audio_embeds,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
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
            attention_mask=None,
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
