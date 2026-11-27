import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    BATCH_SIZE,
    CORESET_DRAWS,
    CORESET_SIZE,
    GLOBAL_LR,
    IHT_ITERS,
    IHT_TOL,
    LOCAL_EPOCHS,
    MC_SAMPLES,
    PERSONAL_LR,
    ZETA,
)
from src.bayesianCoresets.accelerated_iht import AcceleratedIHT
from src.model import detached, kl, net, sample


class ClientModelClass:
    def __init__(self, cid, model, method, uniform):
        self.id = cid
        self.method = method
        self.uniform = uniform
        self.model = copy.deepcopy(model)
        self.personal_model = copy.deepcopy(model)
        self.v_z = (self.model.mus, self.model.rhos)
        self.q = (self.personal_model.mus, self.personal_model.rhos)
        self.q_w = (self.personal_model.coreset_mus, self.personal_model.coreset_rhos)
        self.optimizer_q = torch.optim.Adam(list(self.q[0]) + list(self.q[1]), lr=PERSONAL_LR)
        self.optimizer_q_w = torch.optim.Adam(list(self.q_w[0]) + list(self.q_w[1]), lr=PERSONAL_LR)
        self.optimizer_v_z = torch.optim.Adam(list(self.v_z[0]) + list(self.v_z[1]), lr=GLOBAL_LR)

    def set_data(self, X, Y, ids, weights):
        self.X = X
        self.Y = Y
        self.ids = ids
        self.weights = weights
        self.kl_scale = ZETA * BATCH_SIZE / len(Y)

    def set_parameters(self, global_model):
        for mine, theirs in zip(self.v_z[0], global_model.mus):
            mine.data = theirs.data.clone()
        for mine, theirs in zip(self.v_z[1], global_model.rhos):
            mine.data = theirs.data.clone()

    def support(self):
        return set(self.ids[self.weights > 0].tolist())

    def training_weights(self):
        if self.uniform:
            return (self.weights > 0).astype(np.float32)
        return self.weights.astype(np.float32)

    def log_likelihood(self, posterior, X, Y):
        logits = net(X, sample(*posterior))
        return F.log_softmax(logits, dim=1).gather(1, Y.view(-1, 1)).squeeze(1)

    def personal_loss(self, X, Y):
        loss = 0
        for _ in range(MC_SAMPLES):
            fit = self.log_likelihood(self.q, X, Y).sum()
            loss = loss + self.kl_scale * kl(self.q, detached(self.v_z)) - fit
        return loss / MC_SAMPLES

    def coreset_loss(self, X, Y, w):
        scale = len(Y) / w.sum().clamp_min(1e-8)
        loss = 0
        for _ in range(MC_SAMPLES):
            fit = (w * self.log_likelihood(self.q_w, X, Y)).sum() * scale
            loss = loss + self.kl_scale * kl(self.q_w, detached(self.v_z)) - fit
        return loss / MC_SAMPLES

    def pull_global_towards(self, posterior):
        return self.kl_scale * kl(detached(posterior), self.v_z)

    def step(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train_one_batch(self, X, Y, w):
        self.step(self.optimizer_q, self.personal_loss(X, Y))
        if self.method == "pfedbayes":
            self.step(self.optimizer_v_z, self.pull_global_towards(self.q))
        else:
            self.step(self.optimizer_q_w, self.coreset_loss(X, Y, w))
            self.step(self.optimizer_v_z, self.pull_global_towards(self.q_w))

    def train(self):
        w = torch.tensor(self.training_weights())
        for _ in range(LOCAL_EPOCHS):
            order = torch.randperm(len(self.Y))
            for start in range(0, len(self.Y), BATCH_SIZE):
                batch = order[start : start + BATCH_SIZE]
                self.train_one_batch(self.X[batch], self.Y[batch], w[batch])

    def select_coreset(self):
        n = len(self.Y)
        with torch.no_grad():
            draws = [self.log_likelihood(self.q, self.X, self.Y) for _ in range(CORESET_DRAWS)]
        L = torch.stack(draws)
        G = (L - L.mean(dim=0, keepdim=True)) / math.sqrt(CORESET_DRAWS)
        y = G.sum(dim=1, keepdim=True)
        if not torch.isfinite(G).all() or G.norm() < 1e-12:
            return np.ones(n) / n
        w = AcceleratedIHT(y, G, CORESET_SIZE, IHT_TOL, IHT_ITERS).accelrated_IHT_II()
        return w.clamp(min=0).numpy().ravel().astype(np.float64)
