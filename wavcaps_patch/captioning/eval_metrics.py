#!/usr/bin/env python3

import csv
from pathlib import Path

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


class RobustMeteor(Meteor):
    """METEOR 1.5 wrapper tolerant of delayed statistics lines."""

    def _write_line(self, line):
        payload = f"{line}\n"
        try:
            self.meteor_p.stdin.write(payload.encode("utf-8"))
        except TypeError:
            self.meteor_p.stdin.write(payload)
        self.meteor_p.stdin.flush()

    def _read_numeric_line(self, expected):
        while True:
            raw_line = self.meteor_p.stdout.readline()
            if raw_line in (b"", ""):
                stderr = ""
                return_code = self.meteor_p.poll()
                if return_code is not None:
                    stderr = self.meteor_p.stderr.read()
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode("utf-8", errors="replace")
                raise RuntimeError(
                    "METEOR terminated before returning "
                    f"{expected} (return code {return_code}): "
                    f"{stderr.strip()}"
                )

            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="strict").strip()
            else:
                line = raw_line.strip()
            if not line:
                continue

            try:
                values = [float(value) for value in line.split()]
            except ValueError as error:
                raise RuntimeError(
                    f"Unexpected METEOR {expected} response: {line!r}"
                ) from error

            if expected == "statistics" and len(values) > 1:
                return line
            if expected == "score" and len(values) == 1:
                return values[0]
            if expected == "score" and len(values) > 1:
                # Some Java/runtime combinations leave a delayed SCORE
                # sufficient-statistics row in stdout before EVAL scores.
                continue
            raise RuntimeError(
                f"Unexpected METEOR {expected} response: {line!r}"
            )

    def _stat(self, hypothesis_str, reference_list):
        hypothesis_str = hypothesis_str.replace("|||", "").replace("  ", " ")
        score_line = " ||| ".join(
            ("SCORE", " ||| ".join(reference_list), hypothesis_str)
        )
        self._write_line(score_line)
        return self._read_numeric_line("statistics")

    def compute_score(self, gts, res):
        if gts.keys() != res.keys():
            raise ValueError("METEOR references and predictions do not match.")
        image_ids = list(gts.keys())
        scores = []

        self.lock.acquire()
        try:
            eval_line = "EVAL"
            for image_id in image_ids:
                if len(res[image_id]) != 1:
                    raise ValueError(
                        "METEOR requires exactly one prediction per sample."
                    )
                statistics = self._stat(res[image_id][0], gts[image_id])
                eval_line += f" ||| {statistics}"

            self._write_line(eval_line)
            for _ in image_ids:
                scores.append(self._read_numeric_line("score"))
            score = self._read_numeric_line("score")
        finally:
            self.lock.release()

        return score, scores


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
    meteor_scorer = RobustMeteor()
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
