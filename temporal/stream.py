import os
import pickle

import numpy as np
import torch

from config import (
    ARRIVALS,
    DOWN_SIZE,
    GLOBAL_PER_CLASS,
    NUM_CLASSES,
    NUM_CLIENTS,
    PHIST_SIZE,
    SCHEDULES,
    STAT_SIZE,
    STREAM_START,
    UP_SIZE,
    WINDOW,
)


def read_batch(data_dir, name):
    with open(os.path.join(data_dir, name), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch[b"data"], np.array(batch[b"labels"])


def load_cifar(data_dir):
    train = [read_batch(data_dir, "data_batch_%d" % i) for i in range(1, 6)]
    train_X = np.concatenate([X for X, _ in train])
    train_Y = np.concatenate([Y for _, Y in train])
    test_X, test_Y = read_batch(data_dir, "test_batch")
    return torch.tensor(train_X), torch.tensor(train_Y), torch.tensor(test_X), torch.tensor(test_Y)


def pixels(X):
    return X.float() / 255


def client_classes(cid):
    return sorted({(3 * cid + k) % NUM_CLASSES for k in range(3)})


def alpha(t, cycle):
    if t < cycle["out_start"]:
        return 0.0
    if t <= cycle["out_end"]:
        return (t - cycle["out_start"]) / (cycle["out_end"] - cycle["out_start"])
    if t < cycle["back_start"]:
        return 1.0
    if t <= cycle["back_end"]:
        return (cycle["back_end"] - t) / (cycle["back_end"] - cycle["back_start"])
    return 0.0


def class_mix(t, cycles):
    mix = [1 / 3, 1 / 3, 1 / 3]
    for cycle in cycles:
        a = alpha(t, cycle)
        if a > 0:
            mix = [(1 - a) / 3 + a * target for target in cycle["target"]]
    return mix


def history_mix(schedule):
    mixes = [class_mix(t, schedule["cycles"]) for t in range(STREAM_START, schedule["rounds"])]
    return np.mean(mixes, axis=0).tolist()


def quota(proportions, total):
    exact = [total * p for p in proportions]
    counts = [int(np.floor(x)) for x in exact]
    by_remainder = sorted(range(len(exact)), key=lambda i: (counts[i] - exact[i], i))
    for i in by_remainder[: total - sum(counts)]:
        counts[i] += 1
    return counts


class Stream:
    def __init__(self, seed, data_dir, schedule):
        self.schedule = SCHEDULES[schedule]
        self.rng = np.random.default_rng(seed)
        self.train_X, self.train_Y, self.test_X, self.test_Y = load_cifar(data_dir)
        self.train_pool = [np.flatnonzero(self.train_Y.numpy() == c) for c in range(NUM_CLASSES)]
        self.test_pool = [np.flatnonzero(self.test_Y.numpy() == c) for c in range(NUM_CLASSES)]
        self.classes = [client_classes(cid) for cid in range(NUM_CLIENTS)]
        self.used_train = [set() for _ in range(NUM_CLIENTS)]
        self.used_test = [set() for _ in range(NUM_CLIENTS)]
        self.windows = [self.initial_window(cid) for cid in range(NUM_CLIENTS)]
        self.tests = [self.client_tests(cid) for cid in range(NUM_CLIENTS)]
        self.global_test = self.balanced_global_test()

    def draw(self, pool, used, count, in_window=frozenset()):
        candidates = [i for i in pool if i not in used]
        if len(candidates) < count:
            used.difference_update(set(pool.tolist()) - in_window)
            candidates = [i for i in pool if i not in used]
        chosen = self.rng.choice(candidates, count, replace=False).tolist()
        used.update(chosen)
        return chosen

    def draw_train(self, cid, counts, in_window=frozenset()):
        ids = []
        for c, count in zip(self.classes[cid], counts):
            ids += self.draw(self.train_pool[c], self.used_train[cid], count, in_window)
        return ids

    def draw_test(self, cid, counts):
        ids = []
        for c, count in zip(self.classes[cid], counts):
            ids += self.draw(self.test_pool[c], self.used_test[cid], count)
        return pixels(self.test_X[ids]), self.test_Y[ids]

    def initial_window(self, cid):
        return self.draw_train(cid, quota([1 / 3] * 3, WINDOW))

    def client_tests(self, cid):
        stat_counts = quota([1 / 3] * 3, STAT_SIZE)
        stat = []
        for k in range(3):
            only_k = [stat_counts[j] if j == k else 0 for j in range(3)]
            stat.append(self.draw_test(cid, only_k))
        return {
            "stat": stat,
            "up": self.draw_test(cid, [0, 0, UP_SIZE]),
            "down": self.draw_test(cid, [DOWN_SIZE, 0, 0]),
            "phist": self.draw_test(cid, quota(history_mix(self.schedule), PHIST_SIZE)),
        }

    def balanced_global_test(self):
        ids = []
        for c in range(NUM_CLASSES):
            ids += self.rng.choice(self.test_pool[c], GLOBAL_PER_CLASS, replace=False).tolist()
        return pixels(self.test_X[ids]), self.test_Y[ids]

    def advance_window(self, cid, t, protected):
        expired = set([i for i in self.windows[cid] if i not in protected][:ARRIVALS])
        survivors = [i for i in self.windows[cid] if i not in expired]
        mix = class_mix(t, self.schedule["cycles"])
        arrivals = self.draw_train(cid, quota(mix, ARRIVALS), set(survivors))
        self.windows[cid] = survivors + arrivals

    def window_data(self, cid):
        ids = self.windows[cid]
        return pixels(self.train_X[ids]), self.train_Y[ids], np.array(ids)
