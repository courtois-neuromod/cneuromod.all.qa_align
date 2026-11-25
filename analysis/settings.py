"""Constants shared by every step: time grids, scan ranges and flag thresholds.

Everything a reader might want to tune lives here, with the reason for its
value. The dashboard's "Rules" table is generated from `RULES`, so a threshold
changed here is also the threshold the page documents.
"""

import numpy as np

# --- time grids -------------------------------------------------------------
DEFAULT_TR = 1.49          # seconds; used when a run's sidecar has no RepetitionTime
GRID_STEP = 0.1            # seconds; resolution of the predictors before sampling at TR
GRID_START = -60.0         # seconds; predictors start before t=0 so they can be shifted later
GRID_PAD = 60.0            # seconds kept after the end of the run
HRF_LENGTH = 32.0          # seconds

# --- BOLD -------------------------------------------------------------------
DROPPED_VOLUMES = 2        # first volumes left out (not yet at steady state)
MIN_VOLUMES = DROPPED_VOLUMES + 20
N_TOP_PARCELS = 50         # parcels averaged: those that follow the predictor best elsewhere

# --- shift scan -------------------------------------------------------------
LAGS = np.round(np.arange(-30, 50.001, 0.25), 2)   # shifts tried, seconds
PAIRING_LAG_STEP = 2       # pairing uses every 2nd lag (0.5 s)
CURVE_STEP = 4             # curves are stored every 4th lag (1 s)
MIN_RUNS_PER_PREDICTOR = 3  # a subject needs this many runs to pick parcels leave-one-out
AVAILABILITY_MIN = 0.6     # a recording-based predictor needs data for this share of play time

# --- drift ------------------------------------------------------------------
DRIFT_WINDOW = 100         # volumes
DRIFT_STEP = 10            # volumes
DRIFT_SEARCH = np.round(np.arange(-9, 9.001, 0.25), 2)  # seconds around the run's best shift
SIMULATED_VOLUMES = 3      # size of the simulated shift, jump and drift
N_SIMULATED_RUNS = 40      # runs per dataset that also get a simulated jump and drift
RANDOM_SEED = 0

# --- flags ------------------------------------------------------------------
SHIFT_MIN = 1.5            # volumes; smallest shift that counts
SHIFT_FLAG = 2.0           # volumes; smallest converging shift that is flagged
R_MIN = 0.05               # below this best fit the brain signal is too weak to judge
CONCENTRATION_MIN = 0.7    # share of runs within SHIFT_MIN of the median for a reliable predictor
RELIABLE_MIN_RUNS = 8      # a predictor needs this many runs in a task to be judged reliable
GAIN_ONE = 0.05            # gain a lone predictor needs to flag a shift on its own
GAIN_CONFIRM = 0.04        # gain a predictor needs to confirm a shift the clocks already show
NO_FIT_FRACTION = 0.3      # "no fit": every reliable predictor below this share of its usual fit
PAIR_Z = -1.4              # pairing score below which another run's events fit better
DRIFT_Z = 2.0              # drift score from which a run is flagged
SLOPE_MIN = 1.0            # volumes per 10 minutes; smallest drift that is flagged
CLOCK_MAX_DEVIATION = 300.0  # seconds; beyond this the task clock was restarted, not shifted
TAIL_MAD = 4.0             # an unusual tail is this many MADs from the task's median
SHORT_RUN_FRACTION = 0.9   # a run shorter than this share of its stimulus' usual length
SLOT_TOLERANCE = 1.0       # seconds; a recording must fill its slot within this
SWAP_TOLERANCE = 1.5       # seconds; two recordings that each fill the other's slot
REPLAY_Z = 2.0             # brain evidence needed to prefer another recording

RULES = [
    ("shift_min_vols", SHIFT_MIN, "A shift counts from this many volumes"),
    ("shift_flag_vols", SHIFT_FLAG, "A shift is flagged when the predictors converge on at "
                                    "least this many volumes"),
    ("r_min", R_MIN, "Below this best fit, the brain signal is too weak to judge"),
    ("lag_min_s", float(LAGS[0]), "Earliest shift tried, seconds"),
    ("lag_max_s", float(LAGS[-1]), "Latest shift tried, seconds"),
    ("n_parcels", N_TOP_PARCELS, "Parcels averaged: those that follow the predictor best in "
                                 "the subject's other runs"),
    ("dropped_vols", DROPPED_VOLUMES, "First volumes left out (not yet at steady state)"),
    ("concentration_min", CONCENTRATION_MIN, "A predictor counts for a task when this share of "
                                             "its runs agree on the same shift"),
    ("reliable_min_runs", RELIABLE_MIN_RUNS, "Runs a task needs before a predictor can count"),
    ("gain_one", GAIN_ONE, "Fit gained by shifting that a lone predictor needs to flag a run"),
    ("gain_confirm", GAIN_CONFIRM, "Fit gained by shifting that confirms a shift of the clocks"),
    ("availability_min", AVAILABILITY_MIN, "A recording-based predictor is used for a run when "
                                           "it covers this share of the play time"),
    ("pair_z", PAIR_Z, "Pairing: combined score below which another run's events fit better"),
    ("drift_z", DRIFT_Z, "Drift: combined score from which a run is flagged"),
    ("slope_min", SLOPE_MIN, "Drift: smallest slope flagged, volumes per 10 minutes"),
    ("drift_window_vols", DRIFT_WINDOW, "Drift: window length, volumes"),
    ("drift_step_vols", DRIFT_STEP, "Drift: window step, volumes"),
    ("drift_search_s", float(DRIFT_SEARCH[-1]), "Drift: search range around the run's own best "
                                                "shift, seconds"),
    ("simulated_vols", SIMULATED_VOLUMES, "Size of the simulated shifts, jumps and drifts"),
    ("no_fit_fraction", NO_FIT_FRACTION, "No fit: every reliable predictor below this share of "
                                         "its usual fit"),
    ("clock_max_deviation_s", CLOCK_MAX_DEVIATION, "Clocks: beyond this deviation the task clock "
                                                   "was restarted, and gives no evidence"),
    ("tail_mad", TAIL_MAD, "End of run: an unusual tail is this many times the task's spread "
                           "(MAD) from its median, and at least the dataset's smallest tail"),
    ("short_run_fraction", SHORT_RUN_FRACTION, "End of run: a run shorter than this share of "
                                               "the usual length of its stimulus"),
    ("swap_tolerance_s", SWAP_TOLERANCE, "Repetitions: two replays that each fill the other's "
                                         "slot within this many seconds are swapped"),
    ("slot_tolerance_s", SLOT_TOLERANCE, "Repetitions: a recording must fill its slot within "
                                         "this many seconds"),
    ("replay_z", REPLAY_Z, "Repetitions: brain evidence needed to prefer another recording"),
]


def shift_tolerance(shift_vols):
    """How far two shifts may differ and still count as the same shift (volumes)."""
    return max(SHIFT_MIN, 0.2 * abs(shift_vols))
