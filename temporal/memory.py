import numpy as np

from config import CORESET_SIZE


def class_quotas(labels):
    classes = sorted(set(labels.tolist()))
    shares = np.array_split(np.arange(CORESET_SIZE), len(classes))
    return {c: len(share) for c, share in zip(classes, shares)}


def fill_classes(weights, labels, rng):
    for c, quota in class_quotas(labels).items():
        selected = (labels == c) & (weights > 0)
        free = np.flatnonzero((labels == c) & (weights == 0))
        weights[rng.choice(free, quota - int(selected.sum()), replace=False)] = 1.0
    return weights


def initial_weights(method, labels, rng):
    n = len(labels)
    if method == "balanced":
        return fill_classes(np.zeros(n), labels, rng)
    if method in ("randomsubset", "reservoir"):
        weights = np.zeros(n)
        weights[rng.choice(n, CORESET_SIZE, replace=False)] = 1.0
        return weights
    return np.ones(n) / n


def carry_over(old_ids, old_weights, new_ids):
    weight_of = dict(zip(old_ids.tolist(), old_weights.tolist()))
    return np.array([weight_of.get(i, 0.0) for i in new_ids.tolist()])


def random_replace(old_ids, old_weights, new_ids, rng):
    weights = carry_over(old_ids, old_weights, new_ids)
    unselected = np.flatnonzero(weights == 0)
    missing = CORESET_SIZE - int((weights > 0).sum())
    weights[rng.choice(unselected, missing, replace=False)] = 1.0
    return weights


def balanced_replace(old_ids, old_weights, new_ids, labels, rng):
    weights = carry_over(old_ids, old_weights, new_ids)
    return fill_classes(weights, labels, rng)


def reservoir_replace(old_ids, old_weights, new_ids, seen, rng):
    weights = carry_over(old_ids, old_weights, new_ids)
    arrivals = np.flatnonzero(~np.isin(new_ids, old_ids))
    for position in arrivals:
        seen += 1
        if rng.random() < CORESET_SIZE / seen:
            evicted = rng.choice(np.flatnonzero(weights > 0))
            weights[evicted] = 0.0
            weights[position] = 1.0
    return weights, seen


def should_solve(solve, t):
    if solve == "always":
        return True
    if solve == "once":
        return t == 0
    if solve == "skip_third":
        return t % 3 != 2
    return False


def cap_swaps(proposed, carried, cap):
    positions = range(len(proposed))
    incoming = [i for i in positions if proposed[i] > 0 and carried[i] == 0]
    outgoing = [i for i in positions if carried[i] > 0 and proposed[i] == 0]
    incoming.sort(key=lambda i: (-proposed[i], i))
    outgoing.sort(key=lambda i: (carried[i], i))
    evicted = set(outgoing[:cap])
    kept = [i for i in positions if carried[i] > 0 and i not in evicted]
    room = int((proposed > 0).sum()) - len(kept)
    admitted = incoming[: max(room, 0)]
    weights = np.zeros(len(proposed))
    weights[kept] = carried[kept]
    weights[admitted] = proposed[admitted]
    return weights
