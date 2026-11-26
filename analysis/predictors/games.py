"""Predictors for videogame runs, from the events and the recording of each play.

stim       game on/off. A play lasts its `duration`, or the number of frames
           of its recording when the events give no duration.
resp       button presses: buttons going down on each frame (bk2 input log).
luminance  mean screen luminance per frame, in-game mean removed.
flow       optical flow per frame, in-game mean removed.
audio      audio envelope per frame, in-game mean removed.

The frame-wise features come from `<recording>_lowlevel.npy`, next to each
bk2. A predictor is used for a run only when it covers enough of the play time
(see settings.AVAILABILITY_MIN), which the `availability` entry reports.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import settings, signals

GAME_ROW = "gym-retro_game"
FEATURES = {"luminance": "luminance", "flow": "optical_flow", "audio": "audio_envelope"}
PREDICTORS = ["stim", "resp", "luminance", "flow", "audio"]
DEFAULT_FRAME_RATE = 60.0
DEFAULT_PLAY_LENGTH = 30.0


def is_game_events(events):
    return "trial_type" in events and events.trial_type.astype(str).str.contains(GAME_ROW).any()


def button_presses(bk2_path):
    """Number of buttons going down on each frame of a recording, or None."""
    try:
        lines = zipfile.ZipFile(bk2_path).read("Input Log.txt").decode().splitlines()
    except (OSError, KeyError, zipfile.BadZipFile):
        return None
    frames = [line for line in lines if line.startswith("|")]
    if not frames:
        return None
    held = np.array([[char not in ".|" for char in line] for line in frames], dtype=np.int8)
    before = np.vstack([np.zeros((1, held.shape[1]), np.int8), held])
    return np.clip(np.diff(before, axis=0), 0, None).sum(1).astype(float)


def frame_features(lowlevel_path):
    """{feature: per-frame values} from a *_lowlevel.npy file, or None."""
    try:
        loaded = np.load(lowlevel_path, allow_pickle=True)
    except (OSError, ValueError):
        return None
    if loaded.dtype != object:
        return None
    content = loaded.item()
    return {name: np.asarray(content[key], float) for name, key in FEATURES.items()
            if key in content}


def load_recording(bids_root, stim_file):
    """Everything known about one play's recording: n frames, features, presses."""
    if not str(stim_file).endswith(".bk2"):
        return None
    bk2_path = Path(bids_root) / stim_file
    recording = frame_features(str(bk2_path)[:-4] + "_lowlevel.npy") or {}
    presses = button_presses(bk2_path) if bk2_path.is_file() else None
    if presses is not None:
        recording["resp"] = presses
    lengths = [len(values) for values in recording.values()]
    if not lengths:
        return None
    recording["n_frames"] = len(recording["luminance"]) if "luminance" in recording else lengths[0]
    return recording


def play_timing(row, next_onsets, n_frames):
    """(duration in seconds, frames per second) of one play."""
    duration = pd.to_numeric(row.get("duration", np.nan), errors="coerce")
    recorded = pd.to_numeric(row.get("duration_bk2", np.nan), errors="coerce")
    if pd.notna(recorded) and recorded > 0 and n_frames:
        frame_rate = n_frames / recorded
    elif pd.notna(duration) and duration > 0 and n_frames:
        frame_rate = n_frames / duration
    else:
        frame_rate = DEFAULT_FRAME_RATE
    if pd.notna(duration) and duration > 0:
        return float(duration), frame_rate
    if n_frames:
        return n_frames / frame_rate, frame_rate
    if len(next_onsets):
        return float(next_onsets.iloc[0] - row.onset), frame_rate
    return DEFAULT_PLAY_LENGTH, frame_rate


def frames_to_grid(onset, frame_rate, n_frames, n_samples):
    times = onset + np.arange(n_frames) / frame_rate
    index = ((times - settings.GRID_START) / settings.GRID_STEP).astype(int)
    return np.clip(index, 0, n_samples - 1)


def add_play(raw, play, n_samples):
    """Write one play's signals into the run-long raw signals."""
    onset, duration, frame_rate, recording = play
    on_grid = (signals.grid_times(n_samples) >= onset) \
        & (signals.grid_times(n_samples) < onset + duration)
    raw["stim"][on_grid] = 1
    covered = {"stim"}
    if recording is None:
        return on_grid, covered
    for name in FEATURES:
        if name not in recording:
            continue
        index = frames_to_grid(onset, frame_rate, len(recording[name]), n_samples)
        total = np.bincount(index, recording[name], n_samples)
        count = np.bincount(index, None, n_samples)
        raw[name][count > 0] = total[count > 0] / count[count > 0]
        covered.add(name)
    if "resp" in recording:
        index = frames_to_grid(onset, frame_rate, len(recording["resp"]), n_samples)
        raw["resp"] += np.bincount(index, recording["resp"], n_samples) / settings.GRID_STEP / 10
        covered.add("resp")
    return on_grid, covered


def build(events, n_volumes, tr, bids_root):
    """Predictors and facts about one videogame run.

    Returns (predictors, info). info holds events_start, events_end,
    `availability` (share of play time each predictor covers) and `plays`: one
    (onset, end, n_frames, frame_rate, stim_file, level) per play.
    """
    n_samples = signals.grid_length(n_volumes, tr)
    events = events.sort_values("onset").reset_index(drop=True)
    games = events[events.trial_type.astype(str).str.contains(GAME_ROW)]
    raw = {name: np.zeros(n_samples) for name in PREDICTORS}
    in_game = np.zeros(n_samples, dtype=bool)
    covered_time = {name: 0.0 for name in PREDICTORS}
    feature_values = {name: [] for name in FEATURES}
    plays, total_time = [], 0.0
    for _, row in games.iterrows():
        stim_file = str(row.get("stim_file", ""))
        recording = load_recording(bids_root, stim_file)
        n_frames = recording["n_frames"] if recording else 0
        duration, frame_rate = play_timing(row, events.onset[events.onset > row.onset], n_frames)
        on_grid, covered = add_play(raw, (row.onset, duration, frame_rate, recording), n_samples)
        in_game |= on_grid
        total_time += duration
        for name in covered:
            covered_time[name] += duration
            if name in FEATURES:
                feature_values[name].append(recording[name])
        plays.append((float(row.onset), float(row.onset + duration), n_frames, frame_rate,
                      stim_file, str(row.get("level", ""))))
    for name, values in feature_values.items():
        if values:
            raw[name] = np.where(in_game, raw[name] - np.concatenate(values).mean(), 0)
    availability = {name: covered_time[name] / total_time if total_time else 0.0
                    for name in PREDICTORS}
    predictors = {name: signals.convolve_hrf(raw[name]) for name in PREDICTORS
                  if availability[name] >= settings.AVAILABILITY_MIN}
    other = events[~events.trial_type.astype(str).str.contains(GAME_ROW)]
    other_end = other_events_end(other)
    info = dict(availability=availability, plays=plays,
                events_start=min((play[0] for play in plays), default=None),
                events_end=max([play[1] for play in plays] + [other_end], default=None))
    return predictors, info


def other_events_end(other):
    """End of the last non-game event of the run (0 when there is none)."""
    if not len(other):
        return 0.0
    if "duration" in other:
        duration = pd.to_numeric(other["duration"], errors="coerce").fillna(0)
    else:
        duration = 0
    return float((pd.to_numeric(other.onset, errors="coerce") + duration).max())
