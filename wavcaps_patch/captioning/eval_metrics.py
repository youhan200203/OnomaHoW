#!/usr/bin/env python3

import csv
from pathlib import Path

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def _read_rows(value):
    if isinstance(value, list):
        return [dict(row) for row in value]
    path = Path(value)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _pack_metric(score, per_sample, file_names):
    return {
        "score": float(score),
        "scores": {
            file_name: float(per_sample[index])
            for index, file_name in enumerate(file_names)
        },
    }


def evaluate_metrics(prediction_file, reference_file, nb_reference_captions=5):
    predictions = _read_rows(prediction_file)
    references = _read_rows(reference_file)
    reference_by_name = {row["file_name"]: row for row in references}
    predictions.sort(key=lambda row: row["file_name"])
    file_names = [row["file_name"] for row in predictions]
    if not all(name in reference_by_name for name in file_names):
        raise ValueError("Every prediction must have matching references.")

    gts = {
        index: [
            reference_by_name[row["file_name"]][f"caption_{caption_index}"]
            for caption_index in range(1, nb_reference_captions + 1)
        ]
        for index, row in enumerate(predictions)
    }
    res = {
        index: [row["caption_predicted"]]
        for index, row in enumerate(predictions)
    }

    bleu_scores, bleu_per_sample = Bleu(4).compute_score(gts, res)
    rouge_score, rouge_per_sample = Rouge().compute_score(gts, res)
    meteor_scorer = Meteor()
    try:
        meteor_score, meteor_per_sample = meteor_scorer.compute_score(gts, res)
    finally:
        if hasattr(meteor_scorer, "close"):
            meteor_scorer.close()
        elif hasattr(meteor_scorer, "meteor_p"):
            meteor_scorer.meteor_p.terminate()

    metrics = {
        f"bleu_{index + 1}": _pack_metric(
            bleu_scores[index],
            bleu_per_sample[index],
            file_names,
        )
        for index in range(4)
    }
    metrics["meteor"] = _pack_metric(
        meteor_score,
        meteor_per_sample,
        file_names,
    )
    metrics["rouge_l"] = _pack_metric(
        rouge_score,
        rouge_per_sample,
        file_names,
    )
    return metrics
