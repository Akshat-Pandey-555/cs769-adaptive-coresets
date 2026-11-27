import argparse
import os

import numpy as np

from config import ARMS, SCHEDULES
from src.serverModels.serverpFedbayes import pFedBayes

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "cifar-10-batches-py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--schedule", default="one_cycle", choices=list(SCHEDULES))
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    folder = "results" if args.schedule == "one_cycle" else "results_" + args.schedule
    out = args.out or os.path.join(HERE, folder)

    server = pFedBayes(args.arm, args.seed, args.data, args.schedule)
    history = server.train(args.rounds)

    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "%s_seed%d.npz" % (args.arm, args.seed)), **history)


if __name__ == "__main__":
    main()
