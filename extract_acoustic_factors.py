#!/usr/bin/env python3
"""Extract aligned OnomaCap acoustic-factor sequences into one NPZ file."""

import argparse
import os
import shutil
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal


FACTOR_NAMES = (
    "brightness",
    "roughness",
    "onset",
    "sustain",
    "offset",
    "noise",
)
SUMMARY_NAMES = (
    "brightness_centroid_hz",
    "brightness_nyquist_ratio",
    "roughness_modulation_mean",
    "roughness_modulation_p95",
    "onset_strength_max",
    "onset_strength_p95",
    "sustain_total_seconds",
    "sustain_longest_seconds",
    "sustain_active_ratio",
    "offset_strength_max",
    "offset_strength_p95",
    "noise_flatness_median",
    "noise_flatness_p95",
)

ACTIVE_FLOOR_DB = -40.0
SUSTAIN_ENTER_RATE_DB_PER_S = 20.0
SUSTAIN_EXIT_RATE_DB_PER_S = 60.0
SUSTAIN_ONSET_MIN_RISE_DB = 3.0
SUSTAIN_ONSET_LOOKBACK_MS = 100
MIN_SUSTAIN_MS = 100
EPS = 1e-10


def audio_key(name):
    return unicodedata.normalize("NFKC", Path(str(name)).name).strip().casefold()


def boolean_segments(mask):
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    return list(
        zip(
            np.flatnonzero(changes == 1),
            np.flatnonzero(changes == -1),
        )
    )


def frame_signal(y, window_length, hop_length):
    if len(y) < window_length:
        frame_count = 1
    else:
        frame_count = 1 + int(
            np.ceil((len(y) - window_length) / hop_length)
        )
    padded_length = (frame_count - 1) * hop_length + window_length
    y = np.pad(y, (0, max(0, padded_length - len(y))))
    frames = librosa.util.frame(
        y,
        frame_length=window_length,
        hop_length=hop_length,
    ).T
    return np.ascontiguousarray(frames)


def envelope_modulation_strength(
    y,
    frames,
    sr,
    window_length,
    hop_length,
    modulation_band_hz,
):
    envelope = np.abs(scipy.signal.hilbert(y.astype(np.float64)))
    required_length = (len(frames) - 1) * hop_length + window_length
    envelope = np.pad(envelope, (0, max(0, required_length - len(envelope))))
    envelope_frames = librosa.util.frame(
        envelope,
        frame_length=window_length,
        hop_length=hop_length,
    ).T

    strengths = np.zeros(len(envelope_frames), dtype=np.float32)
    low_hz, high_hz = modulation_band_hz
    for index, frame in enumerate(envelope_frames):
        mean_envelope = float(np.mean(frame))
        if mean_envelope <= EPS:
            continue
        frequencies, psd = scipy.signal.periodogram(
            frame / mean_envelope,
            fs=sr,
            window="hann",
            detrend="constant",
            scaling="density",
        )
        band = (frequencies >= low_hz) & (frequencies <= high_hz)
        if np.count_nonzero(band) >= 2:
            power = np.trapezoid(psd[band], frequencies[band])
            strengths[index] = np.sqrt(max(float(power), 0.0))
    return strengths


def spectral_flux(power):
    if power.shape[0] == 1:
        return np.zeros(1, dtype=np.float32)
    spectrogram_db = librosa.power_to_db(
        np.maximum(power.T, EPS),
        ref=np.max,
    )
    return librosa.onset.onset_strength(
        S=spectrogram_db,
        lag=1,
        max_size=1,
        center=False,
        aggregate=np.mean,
    ).astype(np.float32)


def sustain_mask(rms_db, onset, sr, hop_length):
    smoothed_db = scipy.ndimage.median_filter(rms_db, size=5, mode="nearest")
    frame_seconds = hop_length / sr
    absolute_rate = np.abs(np.gradient(smoothed_db, frame_seconds))

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset,
        sr=sr,
        hop_length=hop_length,
        units="frames",
    )
    usable = onset_frames[onset_frames < len(rms_db)]
    backtracked = (
        librosa.onset.onset_backtrack(usable, 10.0 ** (rms_db / 20.0))
        if len(usable)
        else np.asarray([], dtype=int)
    )

    onset_exclusion = np.zeros(len(rms_db), dtype=bool)
    lookback = max(
        1,
        int(round(SUSTAIN_ONSET_LOOKBACK_MS / 1000 / frame_seconds)),
    )
    for start, peak in zip(backtracked, usable):
        start = int(np.clip(start, 0, len(rms_db) - 1))
        peak = int(np.clip(peak, start, len(rms_db) - 1))
        previous = rms_db[max(0, peak - lookback) : peak]
        if not len(previous):
            continue
        if rms_db[peak] - float(np.median(previous)) >= SUSTAIN_ONSET_MIN_RISE_DB:
            onset_exclusion[start : peak + 1] = True

    active = rms_db >= ACTIVE_FLOOR_DB
    mask = np.zeros(len(rms_db), dtype=bool)
    sustaining = False
    for index in range(len(rms_db)):
        if onset_exclusion[index] or not active[index]:
            sustaining = False
        elif not sustaining and absolute_rate[index] <= SUSTAIN_ENTER_RATE_DB_PER_S:
            sustaining = True
        elif sustaining and absolute_rate[index] >= SUSTAIN_EXIT_RATE_DB_PER_S:
            sustaining = False
        mask[index] = sustaining

    minimum_frames = max(
        1,
        int(np.ceil(MIN_SUSTAIN_MS / 1000 / frame_seconds)),
    )
    filtered = np.zeros_like(mask)
    for start, end in boolean_segments(mask):
        if end - start >= minimum_frames:
            filtered[start:end] = True
    return filtered, active


def analyze_signal(
    y,
    sr=32_000,
    window_ms=200,
    hop_ms=100,
    modulation_band_hz=(20.0, 150.0),
):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or not len(y):
        raise ValueError("Expected a non-empty mono waveform.")
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak

    window_length = int(round(sr * window_ms / 1000))
    hop_length = int(round(sr * hop_ms / 1000))
    frames = frame_signal(y, window_length, hop_length)

    window = scipy.signal.windows.hann(window_length, sym=False)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1))
    power = magnitude**2
    frequencies = np.fft.rfftfreq(window_length, d=1.0 / sr)

    brightness = np.sum(magnitude * frequencies, axis=1) / np.maximum(
        np.sum(magnitude, axis=1), EPS
    )
    noise = np.exp(np.mean(np.log(power + EPS), axis=1)) / np.maximum(
        np.mean(power + EPS, axis=1), EPS
    )
    rms = np.sqrt(np.mean(frames**2, axis=1))
    rms_db = librosa.amplitude_to_db(np.maximum(rms, EPS), ref=np.max)

    roughness = envelope_modulation_strength(
        y,
        frames,
        sr,
        window_length,
        hop_length,
        modulation_band_hz,
    )
    onset = spectral_flux(power)

    reversed_frames = frame_signal(y[::-1], window_length, hop_length)
    reversed_power = np.abs(
        np.fft.rfft(reversed_frames * window, axis=1)
    ) ** 2
    offset = spectral_flux(reversed_power)[::-1].copy()

    sustain, active = sustain_mask(rms_db, onset, sr, hop_length)
    active_values = active if np.any(active) else np.ones_like(active)
    active_brightness = brightness[active_values]
    active_roughness = roughness[active_values]
    active_noise = noise[active_values]

    factor_values = np.stack(
        [brightness, roughness, onset, sustain.astype(float), offset, noise]
    ).astype(np.float32)
    if not np.isfinite(factor_values).all():
        raise FloatingPointError("Non-finite factor value detected.")

    segments = boolean_segments(sustain)
    frame_seconds = hop_length / sr
    sustain_durations = [
        (end - start) * frame_seconds for start, end in segments
    ]
    summary = np.asarray(
        [
            np.median(active_brightness),
            np.median(active_brightness) / (sr / 2),
            np.mean(active_roughness),
            np.percentile(active_roughness, 95),
            np.max(onset, initial=0.0),
            np.percentile(onset, 95),
            np.sum(sustain) * frame_seconds,
            max(sustain_durations, default=0.0),
            np.sum(sustain) / max(np.count_nonzero(active), 1),
            np.max(offset, initial=0.0),
            np.percentile(offset, 95),
            np.median(active_noise),
            np.percentile(active_noise, 95),
        ],
        dtype=np.float32,
    )
    return factor_values, summary


def find_audio_files(audio_root):
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_by_name = {}
    for path in audio_root.rglob("*"):
        if path.suffix.lower() not in extensions:
            continue
        key = audio_key(path.name)
        if key in audio_by_name:
            raise ValueError(
                f"Duplicate normalized audio filename: {path.name}"
            )
        audio_by_name[key] = path
    return audio_by_name


def save_npz(path, arrays, overwrite=False):
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Output must end in .npz: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    os.replace(temporary_path, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-csv", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/content/onomacap_acoustic_factors.npz"),
    )
    parser.add_argument("--drive-output", type=Path)
    parser.add_argument("--sr", type=int, default=32_000)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--hop-ms", type=int, default=100)
    parser.add_argument("--roughness-low-hz", type=float, default=20.0)
    parser.add_argument("--roughness-high-hz", type=float, default=150.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    annotations = pd.read_csv(args.annotation_csv)
    if "audio_file" not in annotations:
        raise ValueError("Annotation CSV must contain an audio_file column.")

    audio_by_name = find_audio_files(args.audio_root)
    factor_sequences = []
    summaries = []
    lengths = []

    audio_files = annotations["audio_file"]
    for index, audio_file in enumerate(audio_files, start=1):
        path = audio_by_name.get(audio_key(audio_file))
        if path is None:
            raise FileNotFoundError(f"Audio not found: {audio_file}")
        y, _ = librosa.load(path, sr=args.sr, mono=True)
        factors, summary = analyze_signal(
            y,
            sr=args.sr,
            window_ms=args.window_ms,
            hop_ms=args.hop_ms,
            modulation_band_hz=(
                args.roughness_low_hz,
                args.roughness_high_hz,
            ),
        )
        factor_sequences.append(factors)
        summaries.append(summary)
        lengths.append(factors.shape[1])
        if index % 100 == 0 or index == len(audio_files):
            print(f"Extracted {index:,}/{len(audio_files):,}")

    lengths = np.asarray(lengths, dtype=np.int32)
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)]
    )
    arrays = {
        "factor_values": np.concatenate(factor_sequences, axis=1),
        "offsets": offsets,
        "lengths": lengths,
        "summaries": np.stack(summaries).astype(np.float32),
        "audio_files": annotations["audio_file"].astype(str).to_numpy(),
        "factor_names": np.asarray(FACTOR_NAMES),
        "summary_names": np.asarray(SUMMARY_NAMES),
        "sample_rate": np.asarray(args.sr, dtype=np.int32),
        "window_ms": np.asarray(args.window_ms, dtype=np.int32),
        "hop_ms": np.asarray(args.hop_ms, dtype=np.int32),
        "roughness_band_hz": np.asarray(
            [args.roughness_low_hz, args.roughness_high_hz],
            dtype=np.float32,
        ),
    }
    save_npz(args.output, arrays, overwrite=args.overwrite)
    print(f"Saved {len(annotations):,} files to {args.output}")

    if args.drive_output is not None:
        if args.drive_output.exists() and not args.overwrite:
            raise FileExistsError(
                f"Drive output already exists: {args.drive_output}"
            )
        args.drive_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, args.drive_output)
        print(f"Copied to {args.drive_output}")


if __name__ == "__main__":
    main()
