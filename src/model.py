import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from config import RHO_OFFSET, WEIGHT_SCALE


class federatedBNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        shapes = [(input_dim, hidden_dim), (hidden_dim,), (hidden_dim, output_dim), (output_dim,)]
        self.mus = nn.ParameterList()
        self.rhos = nn.ParameterList()
        self.coreset_mus = nn.ParameterList()
        self.coreset_rhos = nn.ParameterList()
        for shape in shapes:
            mu = WEIGHT_SCALE * torch.randn(shape)
            rho = RHO_OFFSET + torch.zeros(shape)
            self.mus.append(nn.Parameter(mu))
            self.rhos.append(nn.Parameter(rho))
            self.coreset_mus.append(nn.Parameter(mu.clone()))
            self.coreset_rhos.append(nn.Parameter(rho.clone()))


def sigmas(rhos):
    return [F.softplus(rho) for rho in rhos]


def sample(mus, rhos):
    return [mu + sigma * torch.randn_like(mu) for mu, sigma in zip(mus, sigmas(rhos))]


def net(X, params):
    W1, b1, W2, b2 = params
    hidden = F.relu(X @ W1 + b1)
    return hidden @ W2 + b2


def kl(p, q):
    total = 0
    for mu_p, sigma_p, mu_q, sigma_q in zip(p[0], sigmas(p[1]), q[0], sigmas(q[1])):
        total = total + kl_divergence(Normal(mu_p, sigma_p), Normal(mu_q, sigma_q)).sum()
    return total


def detached(posterior):
    mus, rhos = posterior
    return [mu.detach() for mu in mus], [rho.detach() for rho in rhos]
