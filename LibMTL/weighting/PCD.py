from itertools import combinations

import torch
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class PCD(AbsWeighting):
    r"""Priority-Constrained Descent (PCD).

    This method is proposed in `Not All Objectives Are Born Equal: Priority-Constrained Descent for Hierarchical Multi-Objective Optimization (TMLR 2026) <https://openreview.net/forum?id=HT01yGHLEt>`_ \
    and implemented by adapting from the `official implementation <https://github.com/DaraVaram/priority-constrained-descent>`_.

    PCD is designed for a primary task with auxiliary tasks. The update stays as close as possible to the \
    primary task's gradient, subject to every other task receiving at least a ``PCD_tau`` fraction of the \
    first-order progress a step along its own normalized gradient would give. Task gradients are normalized \
    by a per-task EMA of their squared norms, so ``PCD_tau`` is scale-free. The quadratic program is solved \
    exactly from the Gram matrix of the task gradients, and the resulting direction is rescaled to the norm \
    of the primary task's gradient.

    Args:
        PCD_tau (float or list, default=0.02): The tolerance in [0, 1]. Either one value for all \
            auxiliary tasks or one value per auxiliary task (in ``task_name`` order, primary task excluded).
        PCD_primary (str or int, default=None): The primary task, given by name or by index into ``task_name``. \
            If ``None``, the first task is the primary task.
        PCD_beta (float, default=0.999): The EMA decay of the per-task gradient normalization.

    .. note::
            With ``rep_grad=True``, the quadratic program and the norm rescaling use the gradients with respect \
            to the representations, as in the other gradient-based methods of LibMTL.

    """
    def __init__(self):
        super(PCD, self).__init__()

    def init_param(self):
        self.PCD_v = np.zeros(self.task_num)
        self.PCD_t = 0

    def backward(self, losses, **kwargs):
        tau = kwargs.get('PCD_tau', 0.02)
        primary = kwargs.get('PCD_primary', None)
        beta = kwargs.get('PCD_beta', 0.999)

        taus = np.asarray(tau, dtype=np.float64).reshape(-1)
        if taus.size not in [1, self.task_num-1] or np.any(taus < 0):
            raise ValueError('PCD_tau must be one non-negative value or one per auxiliary task ({}), got {}'.format(self.task_num-1, tau))

        grads = self._get_grads(losses, mode='backward')
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]

        # Gram matrix of the task gradients; pairwise dot products keep it exactly symmetric in floating point
        pairs = [(i, j) for i in range(self.task_num) for j in range(i, self.task_num)]
        dots = torch.stack([torch.dot(grads[i], grads[j]) for i, j in pairs]).double().cpu().numpy()
        gram = np.zeros([self.task_num, self.task_num])
        for (i, j), v in zip(pairs, dots):
            gram[i, j] = gram[j, i] = v

        # per-task EMA normalization, updated every step
        self.PCD_t += 1
        self.PCD_v = beta * self.PCD_v + (1 - beta) * np.diag(gram)
        scale = 1.0 / np.sqrt(self.PCD_v / (1 - beta ** self.PCD_t) + 1e-8)

        # primary task first, auxiliary tasks in task_name order
        p = self._primary_index(primary)
        order = [p] + [tn for tn in range(self.task_num) if tn != p]
        norm_p = np.sqrt(max(gram[p, p], 0.0))
        coef = np.zeros(self.task_num)
        if norm_p > 0:
            w = _solve_pcd_qp((gram * np.outer(scale, scale))[np.ix_(order, order)], taus)
            coef[order] = w * scale[order]
            d_norm = torch.matmul(torch.as_tensor(coef, dtype=grads.dtype, device=grads.device), grads).norm().item()
            coef = coef * (norm_p / d_norm) if d_norm > 0 else np.zeros(self.task_num)

        batch_weight = torch.as_tensor(coef, dtype=grads.dtype, device=grads.device)
        if self.rep_grad:
            self._backward_new_grads(batch_weight, per_grads=per_grads)
        else:
            self._backward_new_grads(batch_weight, grads=grads)
        return coef

    def _primary_index(self, primary):
        if primary is None:
            return 0
        if primary in self.task_name:
            return self.task_name.index(primary)
        try:
            p = int(primary)
        except (TypeError, ValueError):
            raise ValueError('PCD_primary must be a task name in {} or a task index, got {}'.format(self.task_name, primary))
        if not 0 <= p < self.task_num:
            raise ValueError('PCD_primary index {} out of range for {} tasks'.format(p, self.task_num))
        return p


def _solve_pcd_qp(G, tau, tol=1e-9):
    r"""Solve min_d 1/2 ||d - g_1||^2 s.t. g_j^T d >= tau_j ||g_j||^2 (j >= 2) from the Gram matrix ``G``
    of the gradients (row 0 is the primary), by KKT enumeration over working sets of constraints.

    Returns ``w`` with ``w[0] = 1`` and ``d = sum_i w[i] g_i``. If the constraints are mutually
    incompatible (only possible with three or more tasks), returns ``e_0``, i.e., ``d = g_1``.
    """
    K = G.shape[0]
    w = np.zeros(K)
    w[0] = 1.0
    if K == 1:
        return w
    taus = np.broadcast_to(np.asarray(tau, dtype=np.float64).reshape(-1), (K - 1,))  # scalar or one per auxiliary task
    Gs = G[1:, 1:]
    b = taus * np.diag(Gs) - G[1:, 0]
    if np.all(b <= 0):  # the primary gradient is feasible
        return w
    atol = tol * max(float(np.max(np.diag(G))), np.finfo(np.float64).tiny)
    candidates = [j for j in range(K - 1) if Gs[j, j] > 0]
    for size in range(1, len(candidates) + 1):
        for A in map(list, combinations(candidates, size)):
            GA = Gs[np.ix_(A, A)]
            if size > 1:
                dA = np.sqrt(np.diag(GA))
                if np.linalg.eigvalsh(GA / np.outer(dA, dA))[0] <= tol:
                    continue
            mu = np.linalg.solve(GA, b[A])
            if np.any(mu < -tol):
                continue
            mu = np.maximum(mu, 0.0)
            if np.all(Gs[:, A] @ mu - b >= -atol):
                w[1 + np.asarray(A)] = mu
                return w
    return w
