import torch


def project(w, K):
    support = torch.argsort(w.squeeze(), descending=True)[:K].tolist()
    projected = torch.zeros_like(w)
    projected[support] = w[support].clamp(min=0)
    return projected, support


def step_size(A, gradient, support):
    g = gradient[support]
    Ag = A[:, support].mm(g)
    return g.T.mm(g) / Ag.T.mm(Ag) / 2


class AcceleratedIHT:
    def __init__(self, y, A, K, tol, max_iter_num):
        self.y = y
        self.A = A
        self.K = K
        self.tol = tol
        self.max_iter_num = max_iter_num

    def accelrated_IHT_II(self):
        y, A, K = self.y, self.A, self.K
        M, N = A.shape
        w = torch.zeros(N, 1)
        momentum_point = torch.zeros(N, 1)
        Aw = torch.zeros(M, 1)
        active = []
        inactive_mask = torch.ones(N, 1)

        for i in range(1, self.max_iter_num + 1):
            w_previous = w
            Aw_previous = Aw
            if i == 1:
                residual = y
            else:
                residual = y - Aw - tau * A_diff
            gradient = A.T.mm(residual)

            inactive_mask[active] = 0
            ranked = torch.argsort(torch.abs((gradient * inactive_mask).squeeze())).flip(0)
            inactive_mask[active] = 1
            candidates = active + ranked[:K].tolist()
            mu = step_size(A, gradient, candidates)
            w, support = project(momentum_point + mu * gradient, K)

            Aw = A[:, support].mm(w[support])
            gradient = A.T.mm(y - Aw)
            mu = step_size(A, gradient, support)
            w[support] = w[support] + mu * gradient[support]
            w, _ = project(w, K)

            Aw = A[:, support].mm(w[support])
            residual = y - Aw
            A_diff = Aw if i == 1 else Aw - Aw_previous
            denominator = A_diff.T.mm(A_diff)
            if denominator > 0:
                tau = residual.T.mm(A_diff) / denominator
            else:
                tau = residual.T.mm(A_diff) / 1e-6
            momentum_point = w + tau * (w - w_previous)
            active = momentum_point.squeeze().nonzero().squeeze(1).tolist()

            if not torch.isfinite(w).all():
                w = w_previous
                break
            if i > 1 and torch.norm(w - w_previous) < self.tol * torch.norm(w):
                break

        return w
