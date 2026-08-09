#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import platform
import random
from pathlib import Path
from pprint import PrettyPrinter

import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
from loguru import logger
from warmup_scheduler import GradualWarmupScheduler

from data_handling.datamodule import AudioCaptionDataModule
from models.bart_captioning import BartCaptionModel
from models.bert_captioning import BertCaptionModel
from pretrain import train, validate
from tools.optim_utils import cosine_lr, get_optimizer, step_lr
from tools.utils import set_logger, setup_seed


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def atomic_torch_save(state, path):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def main():
    parser = argparse.ArgumentParser(description="Settings.")
    parser.add_argument("-n", "--exp_name", default="htsat_test", type=str)
    parser.add_argument("-c", "--config", default="settings/settings.yaml", type=str)
    parser.add_argument("-l", "--lr", default=1e-4, type=float)
    parser.add_argument("-s", "--seed", default=20, type=int)
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    config["exp_name"] = args.exp_name
    config["seed"] = args.seed
    config["optim_args"]["lr"] = args.lr
    setup_seed(config["seed"])

    folder_name = "{}_lr_{}_batch_{}_seed_{}".format(
        config["exp_name"],
        config["optim_args"]["lr"],
        config["data_args"]["batch_size"],
        config["seed"],
    )
    _model_output_dir, log_output_dir = set_logger(folder_name)
    main_logger = logger.bind(indent=1)

    device, device_name = (
        ("cuda", torch.cuda.get_device_name(torch.cuda.current_device()))
        if torch.cuda.is_available()
        else ("cpu", platform.processor())
    )
    if device != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("OnomaCap Jamo BF16 training requires a BF16 CUDA GPU.")
    main_logger.info(f"Process on {device_name}")

    datamodule = AudioCaptionDataModule(config, config["data_args"]["dataset"])
    train_loader = datamodule.train_dataloader(is_distributed=False)
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    if "bart" in config["text_decoder_args"]["name"]:
        model = BartCaptionModel(config)
    elif "bert" in config["text_decoder_args"]["name"]:
        model = BertCaptionModel(config)
    else:
        raise ValueError(f"Unsupported decoder: {config['text_decoder_args']['name']}")
    model = model.to(device)

    main_logger.info(
        "Training setting:\n" + PrettyPrinter().pformat(config)
    )
    wandb.init(project="audio-captioning", name=folder_name, config=config)
    wandb.watch(model)
    main_logger.info(
        f"Total number of parameters: {sum(item.numel() for item in model.parameters())}"
    )

    if config["pretrain"]:
        pretrain_checkpoint = load_checkpoint(config["pretrain_path"], device)
        strict = bool(config.get("pretrain_strict", True))
        incompatible = model.load_state_dict(
            pretrain_checkpoint["model"],
            strict=strict,
        )
        if not strict:
            if any(
                ".factor_attn." in key
                for key in incompatible.missing_keys
            ):
                model.initialize_factor_attention_from_audio()
            unexpected = list(incompatible.unexpected_keys)
            invalid_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith("factor_")
                and ".factor_attn." not in key
                and ".factor_attn_layer_norm." not in key
            ]
            if unexpected or invalid_missing:
                raise RuntimeError(
                    "Unexpected warm-start mismatch: "
                    f"missing={incompatible.missing_keys}, "
                    f"unexpected={unexpected}"
                )
            main_logger.info(
                f"New factor parameters: {list(incompatible.missing_keys)}"
            )
        main_logger.info(f"Loaded weights from {config['pretrain_path']}")

    optimizer = get_optimizer(
        model.parameters(),
        lr=config["optim_args"]["lr"],
        betas=config["optim_args"]["betas"],
        eps=config["optim_args"]["eps"],
        momentum=config["optim_args"]["momentum"],
        weight_decay=config["optim_args"]["weight_decay"],
        optimizer_name=config["optim_args"]["optimizer_name"],
    )
    if config["optim_args"]["scheduler"] == "cosine":
        scheduler = cosine_lr(
            optimizer,
            base_lr=config["optim_args"]["lr"],
            warmup_length=config["optim_args"]["warmup_epochs"] * len(train_loader),
            steps=len(train_loader) * config["training"]["epochs"],
        )
    elif config["optim_args"]["scheduler"] == "step":
        scheduler = step_lr(
            optimizer,
            base_lr=config["optim_args"]["lr"],
            warmup_length=config["optim_args"]["warmup_epochs"] * len(train_loader),
            adjust_steps=config["optim_args"]["step_epochs"] * len(train_loader),
            gamma=config["optim_args"]["gamma"],
        )
    elif config["optim_args"]["scheduler"] == "old":
        scheduler = None
        scheduler_temp = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
        scheduler_warmup = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=5,
            after_scheduler=scheduler_temp,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config['optim_args']['scheduler']}")

    checkpoint_root = Path(config["training"]["checkpoint_root"])
    checkpoint_dir = checkpoint_root / folder_name
    epoch_checkpoint_dir = checkpoint_dir / "epochs"
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pt"
    evaluation_beam_size = int(
        config.get("evaluation", {}).get("beam_size", 3)
    )

    start_epoch = 1
    loss_stats = []
    selection_scores = []
    epoch_checkpoints = [
        path
        for path in epoch_checkpoint_dir.glob("epoch_*.pt")
        if path.stem.removeprefix("epoch_").isdigit()
    ]
    resume_path = max(
        epoch_checkpoints,
        key=lambda path: int(path.stem.removeprefix("epoch_")),
        default=None,
    )
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        loss_stats = checkpoint.get("loss_stats", [])
        selection_scores = checkpoint.get("selection_scores", [])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state"]]
            )
        main_logger.info(f"Resume: {resume_path} (next epoch: {start_epoch})")
    else:
        main_logger.info("No epoch checkpoint found; start at epoch 1.")

    main_logger.info(
        f"Size of training set: {len(train_loader.dataset)}, "
        f"size of batches: {len(train_loader)}"
    )
    main_logger.info(
        f"Size of validation set: {len(val_loader.dataset)}, "
        f"size of batches: {len(val_loader)}"
    )
    main_logger.info(
        f"Size of test set: {len(test_loader.dataset)}, "
        f"size of batches: {len(test_loader)}"
    )

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        main_logger.info(f"Training for epoch [{epoch}]")
        phase = model.set_training_epoch(epoch)
        main_logger.info(
            f"Training phase: {phase['phase']}, "
            f"audio modality dropout: {phase['audio_modality_dropout']:.3f}, "
            f"trainable parameters: {phase['trainable_parameters']:,}"
        )
        wandb.log(
            {
                "epoch": epoch,
                "train/factor_only": int(phase["factor_only"]),
                "train/audio_modality_dropout": phase[
                    "audio_modality_dropout"
                ],
                "train/trainable_parameters": phase["trainable_parameters"],
            }
        )
        if scheduler is None:
            scheduler_warmup.step()
        train_statistics = train(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            config["training"]["clip_grad"],
        )
        loss = train_statistics["loss"]
        loss_stats.append(loss)
        main_logger.info(
            f"Training statistics:\tloss for epoch [{epoch}]: {loss:.3f}, "
            f"\ttime: {train_statistics['time']:.1f}, "
            f"lr: {optimizer.param_groups[0]['lr']:.6f}."
        )

        main_logger.info("Validating factor-only OnomaCap metrics...")
        factor_only_metrics = validate(
            val_loader,
            model,
            device=device,
            log_dir=Path(log_output_dir) / "factor_only",
            epoch=epoch,
            beam_size=evaluation_beam_size,
            disable_audio=True,
            condition="factor_only",
        )
        main_logger.info("Validating joint OnomaCap metrics...")
        joint_metrics = validate(
            val_loader,
            model,
            device=device,
            log_dir=Path(log_output_dir) / "joint",
            epoch=epoch,
            beam_size=evaluation_beam_size,
            disable_audio=False,
            condition="joint",
        )
        selection_score = float(joint_metrics["bleu_1"]["score"])
        selection_scores.append(selection_score)
        wandb.log(
            {
                f"val/factor_only/{name}": float(values["score"])
                for name, values in factor_only_metrics.items()
            }
            | {
                f"val/joint/{name}": float(values["score"])
                for name, values in joint_metrics.items()
            }
            | {"epoch": epoch}
        )

        factor_only_val_scores = {
            name: float(values["score"])
            for name, values in factor_only_metrics.items()
        }
        joint_val_scores = {
            name: float(values["score"])
            for name, values in joint_metrics.items()
        }
        if selection_score >= max(selection_scores):
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "beam_size": evaluation_beam_size,
                    "epoch": epoch,
                    "selection_metric": "val/joint/bleu_1",
                    "selection_score": selection_score,
                    "factor_only_val_scores": factor_only_val_scores,
                    "joint_val_scores": joint_val_scores,
                    "val_scores": joint_val_scores,
                    "config": config,
                },
                best_model_path,
            )

        epoch_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": epoch * len(train_loader),
            "loss_stats": loss_stats,
            "selection_scores": selection_scores,
            "factor_only_val_scores": factor_only_val_scores,
            "joint_val_scores": joint_val_scores,
            "val_scores": joint_val_scores,
            "config": config,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }
        epoch_path = epoch_checkpoint_dir / f"epoch_{epoch:02d}.pt"
        atomic_torch_save(epoch_state, epoch_path)
        main_logger.info(f"Saved full checkpoint: {epoch_path}")

    main_logger.info("Training done. Start evaluating OnomaCap metrics.")
    best_checkpoint = load_checkpoint(best_model_path, device)
    model.load_state_dict(best_checkpoint["model"])
    main_logger.info(
        f"Best BLEU-1 checkpoint occurred at epoch {best_checkpoint['epoch']}."
    )
    test_metrics = validate(
        test_loader,
        model,
        device=device,
        log_dir=log_output_dir,
        epoch=0,
        beam_size=evaluation_beam_size,
    )
    wandb.log(
        {f"test/{name}": float(values["score"]) for name, values in test_metrics.items()}
    )
    main_logger.info("Evaluation done.")
    wandb.finish()


if __name__ == "__main__":
    main()
