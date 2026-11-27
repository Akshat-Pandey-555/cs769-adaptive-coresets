import numpy as np
import torch

from config import ARMS, HIDDEN_DIM, INPUT_DIM, NUM_CLASSES, NUM_CLIENTS, STREAM_START, WINDOW
from src.clientModels.clientModelClass import ClientModelClass
from src.model import federatedBNN, net
from temporal.memory import (
    balanced_replace,
    cap_swaps,
    carry_over,
    initial_weights,
    random_replace,
    reservoir_replace,
    should_solve,
)
from temporal.stream import Stream


def accuracy(mus, test):
    X, Y = test
    with torch.no_grad():
        predictions = net(X, list(mus)).argmax(dim=1)
    return (predictions == Y).float().mean().item()


class pFedBayes:
    def __init__(self, arm, seed, data_dir, schedule):
        torch.manual_seed(seed)
        self.arm = ARMS[arm]
        self.seed = seed
        self.stream = Stream(seed, data_dir, schedule)
        self.model = federatedBNN(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
        self.users = []
        self.seen = [WINDOW for _ in range(NUM_CLIENTS)]
        self.previous_support = [set() for _ in range(NUM_CLIENTS)]
        self.history = {name: [] for name in ["stat", "phist", "up", "down", "glob", "overlap", "swaps"]}
        for cid in range(NUM_CLIENTS):
            user = ClientModelClass(cid, self.model, self.arm["method"], self.arm["uniform"])
            X, Y, ids = self.stream.window_data(cid)
            user.set_data(X, Y, ids, initial_weights(user.method, Y.numpy(), self.memory_rng(cid, 0)))
            self.users.append(user)

    def memory_rng(self, cid, t):
        return np.random.default_rng(self.seed * 100000 + cid * 1000 + t)

    def refill_memory(self, user, old_ids, old_weights, ids, labels, t):
        rng = self.memory_rng(user.id, t)
        if user.method == "randomsubset":
            return random_replace(old_ids, old_weights, ids, rng)
        if user.method == "balanced":
            return balanced_replace(old_ids, old_weights, ids, labels, rng)
        if user.method == "reservoir":
            weights, self.seen[user.id] = reservoir_replace(old_ids, old_weights, ids, self.seen[user.id], rng)
            return weights
        return carry_over(old_ids, old_weights, ids)

    def advance_stream(self, t):
        for user in self.users:
            old_ids, old_weights = user.ids, user.weights
            protected = user.support() if self.arm["protect"] else set()
            self.stream.advance_window(user.id, t, protected)
            X, Y, ids = self.stream.window_data(user.id)
            weights = self.refill_memory(user, old_ids, old_weights, ids, Y.numpy(), t)
            user.set_data(X, Y, ids, weights)

    def send_parameters(self):
        for user in self.users:
            user.set_parameters(self.model)

    def update_memory(self, user, t):
        if not should_solve(self.arm["solve"], t):
            return
        proposed = user.select_coreset()
        if self.arm["cap"] is not None and t > 0:
            proposed = cap_swaps(proposed, user.weights, self.arm["cap"])
        user.weights = proposed

    def aggregate_parameters(self):
        for k in range(len(self.model.mus)):
            self.model.mus[k].data = torch.stack([user.v_z[0][k].data for user in self.users]).mean(dim=0)
            self.model.rhos[k].data = torch.stack([user.v_z[1][k].data for user in self.users]).mean(dim=0)

    def memory_change(self, user):
        previous, current = self.previous_support[user.id], user.support()
        self.previous_support[user.id] = current
        if not previous or not current:
            return 0.0, 0.0
        expired = previous - set(user.ids.tolist())
        overlap = len(current & previous) / len(current)
        swaps = len(current - previous) - len(expired)
        return overlap, swaps

    def evaluate(self):
        rows = {name: [] for name in ["stat", "phist", "up", "down", "overlap", "swaps"]}
        for user in self.users:
            tests = self.stream.tests[user.id]
            q_mus = user.q[0]
            rows["stat"].append([accuracy(q_mus, test) for test in tests["stat"]])
            rows["phist"].append(accuracy(q_mus, tests["phist"]))
            rows["up"].append(accuracy(q_mus, tests["up"]))
            rows["down"].append(accuracy(q_mus, tests["down"]))
            overlap, swaps = self.memory_change(user)
            rows["overlap"].append(overlap)
            rows["swaps"].append(swaps)
        for name in rows:
            self.history[name].append(rows[name])
        self.history["glob"].append(accuracy(self.model.mus, self.stream.global_test))

    def train(self, rounds):
        rounds = rounds or self.stream.schedule["rounds"]
        for t in range(rounds):
            if t >= STREAM_START:
                self.advance_stream(t)
            self.send_parameters()
            for user in self.users:
                user.train()
                self.update_memory(user, t)
            self.aggregate_parameters()
            self.evaluate()
            stat = np.mean(self.history["stat"][-1])
            print("round %3d  personal %.3f  global %.3f" % (t, stat, self.history["glob"][-1]), flush=True)
        return {name: np.array(values) for name, values in self.history.items()}
