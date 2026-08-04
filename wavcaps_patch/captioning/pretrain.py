#!/usr/bin/env python3
# coding: utf-8

import time

import torch
import wandb
from loguru import logger
from tqdm import tqdm

from eval_metrics import evaluate_metrics
from tools.jamo_preprocessing import jamo_to_hangul_caption
from tools.utils import AverageMeter, decode_output


def train(model, dataloader, optimizer, scheduler, device, epoch, clip_grad=0):
    model.train()
    epoch_loss = AverageMeter()
    start_time = time.time()

    if device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16.")

    for batch_id, (audio, text, _audio_names, _) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
    ):
        optimizer.zero_grad(set_to_none=True)
        step = len(dataloader) * (epoch - 1) + batch_id
        if scheduler is not None:
            scheduler(step)
        wandb.log(
            {
                "global_step": step,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        audio = audio.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(audio, text)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite BF16 loss at epoch={epoch}, batch={batch_id}: "
                f"{loss.item()}"
            )
        loss.backward()

        max_norm = clip_grad if clip_grad != 0 else float("inf")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        epoch_loss.update(loss.detach().float().cpu().item())

        if batch_id % 100 == 0:
            wandb.log(
                {
                    "global_step": step,
                    "train/loss_step": float(loss.detach().cpu()),
                    "train/gradient_norm": float(gradient_norm.detach().cpu()),
                }
            )

    elapsed_time = time.time() - start_time
    wandb.log({"train/loss_epoch": epoch_loss.avg, "epoch": epoch})
    return {"loss": epoch_loss.avg, "time": elapsed_time}


@torch.no_grad()
def validate(data_loader, model, device, log_dir, epoch, beam_size):
    val_logger = logger.bind(indent=1)
    model.eval()
    predicted_captions = []
    reference_captions = []
    file_names = []
    start_time = time.time()

    for batch_data in tqdm(data_loader, total=len(data_loader)):
        audios, caption_lists, audio_names, _audio_ids = batch_data
        audios = audios.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.generate(samples=audios, num_beams=beam_size)

        predicted_captions.extend(output)
        reference_captions.extend(
            [
                [jamo_to_hangul_caption(caption) for caption in captions]
                for captions in caption_lists
            ]
        )
        file_names.extend(audio_names)

    captions_pred, captions_gt = decode_output(
        predicted_captions,
        reference_captions,
        file_names,
        log_dir,
        epoch,
        beam_size=beam_size,
    )
    metrics = evaluate_metrics(captions_pred, captions_gt)
    eval_time = time.time() - start_time

    for metric, values in metrics.items():
        val_logger.info(
            f"beam search (size {beam_size}): {metric:<7s}: "
            f"{values['score']:7.4f}"
        )
    val_logger.info(f"Evaluation time: {eval_time:.1f}")
    return metrics
