"""Compact notebook helpers for Jamo factor-attention analyses."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as functional
import ruamel.yaml as yaml


def configure_korean_plots(
    font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    sns.set_theme(style="white", font=font_name)
    plt.rcParams.update({
        "font.family": font_name,
        "font.sans-serif": [font_name],
        "axes.unicode_minus": False,
    })
    return font_name


class FactorAttentionAnalyzer:
    """Lazily load one checkpoint and expose aggregate/time attention plots."""

    def __init__(
        self,
        captioning_dir,
        config_path,
        checkpoint_path,
        token_groups,
        display_groups,
    ):
        self.captioning_dir = Path(captioning_dir)
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.token_groups = {name: tuple(tokens) for name, tokens in token_groups.items()}
        self.display_groups = {
            name: tuple(tokens) for name, tokens in display_groups.items()
        }
        self.model = None
        self.data = None
        self.config = None

    @property
    def token_order(self):
        return tuple(
            token
            for category in ("choseong", "jungseong", "jongseong")
            for token in self.token_groups[category]
        )

    @property
    def display_by_token(self):
        return {
            token: display
            for category, tokens in self.token_groups.items()
            for token, display in zip(tokens, self.display_groups[category])
        }

    def load(self):
        if self.model is not None:
            return self.model, self.data

        previous_cwd = Path.cwd()
        try:
            os.chdir(self.captioning_dir)
            if str(self.captioning_dir) not in sys.path:
                sys.path.insert(0, str(self.captioning_dir))
            from data_handling.datamodule import AudioCaptionDataModule
            from models.bart_captioning import BartCaptionModel

            with self.config_path.open("r", encoding="utf-8") as stream:
                self.config = yaml.YAML(typ="safe").load(stream)
            self.data = AudioCaptionDataModule(self.config, "OnomaCap")
            self.model = BartCaptionModel(self.config).to("cuda")
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.model.load_state_dict(checkpoint["model"])
            self.model.eval()
        finally:
            os.chdir(previous_cwd)
        return self.model, self.data

    def aggregate(self, output_csv=None):
        model, data = self.load()
        token_order = self.token_order
        token_to_row = {token: index for index, token in enumerate(token_order)}
        factor_names = list(model.factor_names)
        mass_sum = np.zeros((len(token_order), len(factor_names)), dtype=np.float64)
        counts = np.zeros(len(token_order), dtype=np.int64)

        for audios, caption_lists, _, _, factors, factor_mask in data.test_dataloader():
            audios = audios.to("cuda", non_blocking=True)
            factors = factors.to("cuda", non_blocking=True)
            factor_mask = factor_mask.to("cuda", non_blocking=True)
            for reference_index in range(5):
                captions = [items[reference_index] for items in caption_lists]
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    attention = model.teacher_forced_factor_attention(
                        audios,
                        captions,
                        factors,
                        factor_mask,
                        layer_index=-1,
                    )
                target_ids = attention["target_ids"].detach().cpu()
                target_mask = attention["target_mask"].detach().cpu()
                factor_mass = attention["factor_attention_mass"].float().detach().cpu()
                for token, token_id in model.jamo_to_id.items():
                    selected = target_mask & target_ids.eq(token_id)
                    count = int(selected.sum())
                    if count:
                        row = token_to_row[token]
                        mass_sum[row] += factor_mass[selected].sum(dim=0).numpy()
                        counts[row] += count

        mean_mass = np.divide(
            mass_sum,
            counts[:, None],
            out=np.zeros_like(mass_sum),
            where=counts[:, None] > 0,
        )
        shares = np.divide(
            mean_mass,
            mean_mass.sum(axis=1, keepdims=True),
            out=np.zeros_like(mean_mass),
            where=mean_mass.sum(axis=1, keepdims=True) > 0,
        )
        categories = {
            token: category
            for category, tokens in self.token_groups.items()
            for token in tokens
        }
        rows = []
        for index, token in enumerate(token_order):
            row = {
                "jamo": token,
                "category": categories[token],
                "occurrences": int(counts[index]),
            }
            row.update({f"{name}_mass": mean_mass[index, i] for i, name in enumerate(factor_names)})
            row.update({f"{name}_share": shares[index, i] for i, name in enumerate(factor_names)})
            rows.append(row)
        table = pd.DataFrame(rows)
        if output_csv is not None:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(output_csv, index=False)
        return table

    def plot_category_heatmaps(self, table, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model, _ = self.load()
        share_columns = [f"{name}_share" for name in model.factor_names]
        for category in ("choseong", "jungseong", "jongseong"):
            selected = table.query("category == @category and occurrences > 0")
            labels = [
                f"{self.display_by_token[token]}  n={count}"
                for token, count in zip(selected["jamo"], selected["occurrences"])
            ]
            figure, axis = plt.subplots(figsize=(10, max(5, len(selected) * 0.42)))
            sns.heatmap(
                selected[share_columns],
                annot=True,
                fmt=".2f",
                cmap="mako",
                vmin=0.0,
                vmax=1.0,
                xticklabels=model.factor_names,
                yticklabels=labels,
                ax=axis,
            )
            axis.set(title=f"Teacher-forced {category} factor attention share", xlabel="Acoustic factor", ylabel="Canonical Jamo")
            figure.tight_layout()
            figure.savefig(output_dir / f"{category}_factor_attention.png", dpi=200, bbox_inches="tight")
            plt.show()
            plt.close(figure)

    def _resolve_jamo(self, category, jamo):
        if category not in self.token_groups:
            raise ValueError("category must be choseong, jungseong, or jongseong.")
        canonical = self.token_groups[category]
        if jamo in canonical:
            return jamo
        mapping = dict(zip(self.display_groups[category], canonical))
        if jamo not in mapping:
            raise ValueError(f"{jamo!r} is not a valid {category} Jamo.")
        return mapping[jamo]

    def plot_time(
        self,
        category,
        jamo,
        reference_index=None,
        layer_index=-1,
        time_bins=32,
        max_batches=None,
        cmap="mako",
    ):
        if time_bins < 2:
            raise ValueError("time_bins must be at least 2.")
        if reference_index is None:
            reference_indices = range(5)
        elif 0 <= reference_index < 5:
            reference_indices = (reference_index,)
        else:
            raise ValueError("reference_index must be None or an integer from 0 to 4.")

        canonical = self._resolve_jamo(category, jamo)
        model, data = self.load()
        token_id = model.jamo_to_id[canonical]
        total = torch.zeros(len(model.factor_names), time_bins, dtype=torch.float64)
        occurrences = 0

        for batch_index, batch in enumerate(data.test_dataloader()):
            if max_batches is not None and batch_index >= max_batches:
                break
            audios, caption_lists, _, _, factors, factor_mask = batch
            for current_reference in reference_indices:
                indices = [
                    index
                    for index, captions in enumerate(caption_lists)
                    if canonical in captions[current_reference].split()
                ]
                if not indices:
                    continue
                captions = [caption_lists[index][current_reference] for index in indices]
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    attention = model.teacher_forced_factor_attention(
                        audios[indices].to("cuda", non_blocking=True),
                        captions,
                        factors[indices].to("cuda", non_blocking=True),
                        factor_mask[indices].to("cuda", non_blocking=True),
                        layer_index=layer_index,
                    )
                target_ids = attention["target_ids"].detach().cpu()
                target_mask = attention["target_mask"].detach().cpu()
                by_time = attention["factor_attention_by_time"].float().detach().cpu()
                time_mask = attention["factor_time_mask"].detach().cpu()
                for sample_index in range(len(indices)):
                    valid_time = int(time_mask[sample_index].sum())
                    positions = torch.where(
                        target_mask[sample_index] & target_ids[sample_index].eq(token_id)
                    )[0]
                    for position in positions.tolist():
                        curve = by_time[sample_index, position, :, :valid_time]
                        curve = functional.interpolate(
                            curve.unsqueeze(0),
                            size=time_bins,
                            mode="linear",
                            align_corners=False,
                        ).squeeze(0)
                        total += (curve / curve.sum().clamp_min(1e-12)).double()
                        occurrences += 1

        if not occurrences:
            raise ValueError(f"No occurrences found for {category} {jamo!r}.")
        result = pd.DataFrame(
            total.numpy() / occurrences * 100.0,
            index=model.factor_names,
            columns=np.linspace(0.0, 1.0, time_bins),
        )
        figure, axis = plt.subplots(figsize=(12, 4.5))
        sns.heatmap(result, cmap=cmap, xticklabels=False, cbar_kws={"label": "Mean attention mass (%)"}, ax=axis)
        axis.set_xticks(np.linspace(0.5, time_bins - 0.5, 5))
        axis.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        axis.set(
            title=f"{category} {jamo} factor × relative-time attention (n={occurrences}, layer={layer_index})",
            xlabel="Relative acoustic time",
            ylabel="Acoustic factor",
        )
        figure.tight_layout()
        plt.show()
        return result

    def clear(self):
        self.model = None
        self.data = None
        gc.collect()
        torch.cuda.empty_cache()
