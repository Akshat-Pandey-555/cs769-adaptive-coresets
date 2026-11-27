import argparse
import glob
import os

import numpy as np

from config import SCHEDULES, STREAM_START
from temporal.stream import class_mix

T_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
PAIRS = [("B", "C"), ("B", "C80"), ("B", "C20"), ("B", "C0"), ("B", "C0U"),
         ("C20", "C80"), ("C", "P23"), ("C", "D1C"), ("C", "D1A"), ("D1B", "D1A"), ("B", "D1B"),
         ("BAL", "B"), ("RES", "B"), ("BAL", "C20")]
COLUMNS = ["acc", "up", "down", "glob", "overlap", "swaps"]


def acc_lambda(run, lam, cycles):
    rounds = len(run["stat"])
    mix = np.array([class_mix(t, cycles) for t in range(rounds)])
    current = (run["stat"] * mix[:, None, :]).sum(axis=2)
    return lam * current + (1 - lam) * run["phist"]


def summarize(run, start, end, cycles):
    window = slice(start, end)
    stream = slice(STREAM_START, end)
    return {
        "acc": acc_lambda(run, 0.5, cycles)[window].mean(),
        "up": run["up"][window].mean(),
        "down": run["down"][window].mean(),
        "glob": run["glob"][window].mean(),
        "overlap": run["overlap"][stream].mean(),
        "swaps": run["swaps"][stream].mean(),
    }


def load(results_dir, start, end, cycles):
    table = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_seed*.npz"))):
        arm, seed = os.path.basename(path)[:-4].split("_seed")
        run = np.load(path)
        if len(run["stat"]) >= end:
            table.setdefault(arm, {})[int(seed)] = summarize(run, start, end, cycles)
    return table


def interval(values):
    n = len(values)
    mean = np.mean(values)
    if n not in T_95:
        return mean, float("nan")
    return mean, T_95[n] * np.std(values, ddof=1) / np.sqrt(n)


def print_table(table):
    print("%-5s %5s " % ("arm", "seeds") + " ".join("%8s" % c for c in COLUMNS))
    for arm, seeds in sorted(table.items(), key=lambda item: np.mean([s["acc"] for s in item[1].values()])):
        cells = ["%8.3f" % np.mean([s[c] for s in seeds.values()]) for c in COLUMNS]
        if arm == "FULL":
            cells[-2:] = ["%8s" % "-", "%8s" % "-"]
        print("%-5s %5d " % (arm, len(seeds)) + " ".join(cells))


def print_pairs(table):
    for a, b in PAIRS:
        shared = sorted(set(table.get(a, {})) & set(table.get(b, {})))
        if not shared:
            continue
        differences = [table[a][s]["acc"] - table[b][s]["acc"] for s in shared]
        mean, half_width = interval(differences)
        spread = "" if np.isnan(half_width) else " +- %.3f" % half_width
        print("%-4s - %-4s  %+.3f%s  (%d seeds)" % (a, b, mean, spread, len(shared)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default=None)
    parser.add_argument("--schedule", default="one_cycle", choices=list(SCHEDULES))
    args = parser.parse_args()
    folder = "results" if args.schedule == "one_cycle" else "results_" + args.schedule
    schedule = SCHEDULES[args.schedule]
    for name, start, end in schedule["windows"]:
        table = load(args.results_dir or folder, start, end, schedule["cycles"])
        print("%s (rounds %d-%d)" % (name, start, end - 1))
        print_table(table)
        print()
        print_pairs(table)
        print()


if __name__ == "__main__":
    main()
