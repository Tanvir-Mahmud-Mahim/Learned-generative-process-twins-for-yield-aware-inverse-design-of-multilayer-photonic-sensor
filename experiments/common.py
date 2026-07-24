"""Shared configuration and IO helpers for all experiments."""

import json
import os
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

CENTER = 0.532            # primary specification: 532 nm fluorescence notch
TRAIN_SPECS = [0.48, 0.50, 0.54, 0.56, 0.60]
TEST_SPECS = [0.52, 0.58, 0.62]

N_RECIPES = 300           # released benchmark designs
TRACE_RECIPES = 200       # recipes observed by the process twin
RUNS_PER_RECIPE = 2       # historical runs per recipe -> 400 traces
SEED = 20260711


def save_json(name, obj):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print("saved", name)


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def save_pickle(name, obj):
    with open(os.path.join(RESULTS, name), "wb") as f:
        pickle.dump(obj, f)
    print("saved", name)


def load_pickle(name):
    with open(os.path.join(RESULTS, name), "rb") as f:
        return pickle.load(f)
