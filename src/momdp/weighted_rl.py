"""Weighted off-policy value estimators (WIS) for the tabular MOMDP setting.

Implements the *weighted* importance-sampling estimators from
    Jiang & Li (2016), "Doubly Robust Off-policy Value Evaluation for RL"
    (references/weighted_RL.pdf), specifically:

    Eq. (6)  trajectory-wise WIS :  V_WIS      = (rho_{1:H} / w_H) * sum_t gamma^{t-1} r_t
    Eq. (7)  step-wise   WIS :      V_step-WIS = sum_t gamma^{t-1} (rho_{1:t} / w_t) r_t

with the per-step / cumulative importance ratios and the dataset-average
normalizer

    rho_t     = pi1(a_t|s_t) / pi0(a_t|s_t)
    rho_{1:t} = prod_{t'<=t} rho_{t'}
    w_t       = (1/|D|) sum_i rho_{1:t}^{(i)}        (average cumulative ratio over D)

Trajectories are collected under a behavior policy pi0; the estimators return the
value of a target policy pi1 (off-policy evaluation). The unweighted IS
estimators (Eq. 4 / 5) are included as baselines for comparison.

MOMDP twist: rewards are d-dimensional (one entry per objective). The importance
ratios depend only on the policies, not on the reward, so a single pass estimates
the whole value VECTOR V^{pi1} in R^d at once. For any preference lambda on the
simplex the scalarized value is lambda . V^{pi1}, because (W)IS is linear in the
reward and the weights are reward-independent -- i.e. evaluating on the scalarized
reward equals scalarizing the per-objective estimates.

Run `python weighted_rl.py` (from src/momdp) for a convergence + variance demo.
"""

import numpy as np

from mdp import MOMDP


def uniform_policy(H, S, A):
    """Uniform-random policy of shape (H, S, A) -- full support, valid as pi0."""
    return np.full((H, S, A), 1.0 / A)


def epsilon_soft_policy(greedy_policy, eps):
    """Soften a (near-)deterministic policy: (1-eps)*greedy + eps*uniform.

    Keeps full support so that the trajectory-wise estimators are well defined
    even when pi1 is the (one-hot) optimal policy.
    """
    H, S, A = greedy_policy.shape
    return (1.0 - eps) * greedy_policy + eps / A


class MOMDP_OPE(object):
    """Off-policy value evaluation for a tabular MOMDP via (weighted) IS."""

    def __init__(self, momdp):
        self.momdp = momdp
        self.S = momdp.S
        self.A = momdp.A
        self.H = momdp.H
        self.d = momdp.d

    # ------------------------------------------------------------------ #
    # Data collection
    # ------------------------------------------------------------------ #
    def rollout(self, behavior_policy, K, rng=None):
        """Roll out K trajectories under ``behavior_policy`` (shape (H, S, A)).

        Episodes start at state 0 (matching ``MOMDP.play``). Returns a dict:
            states  : (K, H) int    state x_h visited at each step
            actions : (K, H) int    action a_h taken at each step
            rewards : (K, H, d)     vector reward r_h = momdp.rewards[:, h, x_h, a_h]
        """
        rng = np.random.default_rng() if rng is None else rng
        H, S, A, d = self.H, self.S, self.A, self.d
        states = np.zeros((K, H), dtype=int)
        actions = np.zeros((K, H), dtype=int)
        rewards = np.zeros((K, H, d), dtype=float)
        transit = self.momdp.transit          # (S, A, S)
        rew = self.momdp.rewards              # (d, H, S, A)
        for k in range(K):
            x = 0
            for h in range(H):
                p_a = behavior_policy[h, x, :]
                a = rng.choice(A, p=p_a / p_a.sum())
                states[k, h] = x
                actions[k, h] = a
                rewards[k, h, :] = rew[:, h, x, a]
                p_y = transit[x, a, :]
                x = rng.choice(S, p=p_y / p_y.sum())
        return {"states": states, "actions": actions, "rewards": rewards}

    # ------------------------------------------------------------------ #
    # Importance ratios
    # ------------------------------------------------------------------ #
    def cumulative_ratios(self, data, behavior_policy, target_policy):
        """Cumulative importance ratios rho_{1:t}, shape (K, H).

        rho_cum[k, h] = prod_{h' <= h} pi1(a|s) / pi0(a|s)   (= rho_{1:t}, t=h+1).
        """
        states, actions = data["states"], data["actions"]
        K, H = states.shape
        h_idx = np.arange(H)[None, :]                      # (1, H)
        p1 = target_policy[h_idx, states, actions]         # (K, H)
        p0 = behavior_policy[h_idx, states, actions]       # (K, H)
        return np.cumprod(p1 / p0, axis=1)                 # (K, H)

    # ------------------------------------------------------------------ #
    # Estimators
    # ------------------------------------------------------------------ #
    def estimate(self, data, behavior_policy, target_policy, gamma=1.0,
                 weighted=True, stepwise=True, return_per_traj=False):
        """Estimate the value vector V^{pi1} in R^d from off-policy ``data``.

        weighted=False -> IS   : Eq. (4) trajectory-wise / Eq. (5) step-wise.
        weighted=True  -> WIS  : Eq. (6) trajectory-wise / Eq. (7) step-wise.

        The dataset estimate is the average over trajectories of the per-trajectory
        value defined by the corresponding equation. With ``return_per_traj`` the
        (K, d) per-trajectory values are also returned.
        """
        rewards = data["rewards"]                                  # (K, H, d)
        K, H, d = rewards.shape
        rho_cum = self.cumulative_ratios(data, behavior_policy, target_policy)  # (K, H)
        disc = gamma ** np.arange(H)                               # (H,)

        if stepwise:
            if weighted:
                w = rho_cum.mean(axis=0)                           # (H,)  w_t  [Eq. 7]
                # rho_{1:t} / w_t, guarding the 0/0 that arises if no trajectory has
                # nonzero weight at step t (e.g. a deterministic pi1 never matched).
                coeff = np.divide(rho_cum, w[None, :],
                                  out=np.zeros_like(rho_cum),
                                  where=w[None, :] > 0)
            else:
                coeff = rho_cum                                    # [Eq. 5]
            # per trajectory: sum_t gamma^{t-1} * coeff_t * r_t
            per_traj = np.einsum("h,kh,khd->kd", disc, coeff, rewards)   # (K, d)
        else:
            rho_full = rho_cum[:, -1]                              # (K,)  rho_{1:H}
            ret = np.einsum("h,khd->kd", disc, rewards)           # (K, d)  trajectory return
            if weighted:
                wH = rho_full.mean()                               # w_H  [Eq. 6]
                scale = rho_full / wH if wH > 0 else np.zeros_like(rho_full)
            else:
                scale = rho_full                                  # [Eq. 4]
            per_traj = scale[:, None] * ret                       # (K, d)

        value = per_traj.mean(axis=0)                             # (1/|D|) sum_i  -> (d,)
        return (value, per_traj) if return_per_traj else value

    # Convenience wrappers (named after the paper's equations) -------------- #
    def wis_trajectory(self, data, pi0, pi1, gamma=1.0):
        """Eq. (6): trajectory-wise weighted importance sampling."""
        return self.estimate(data, pi0, pi1, gamma, weighted=True, stepwise=False)

    def wis_stepwise(self, data, pi0, pi1, gamma=1.0):
        """Eq. (7): step-wise weighted importance sampling."""
        return self.estimate(data, pi0, pi1, gamma, weighted=True, stepwise=True)

    def is_trajectory(self, data, pi0, pi1, gamma=1.0):
        """Eq. (4): trajectory-wise importance sampling (baseline)."""
        return self.estimate(data, pi0, pi1, gamma, weighted=False, stepwise=False)

    def is_stepwise(self, data, pi0, pi1, gamma=1.0):
        """Eq. (5): step-wise importance sampling (baseline)."""
        return self.estimate(data, pi0, pi1, gamma, weighted=False, stepwise=True)

    # ------------------------------------------------------------------ #
    # Ground truth (for validation)
    # ------------------------------------------------------------------ #
    def true_value(self, target_policy):
        """Exact value vector V^{pi}(s_0) in R^d (undiscounted, gamma=1).

        Computed per objective with the MOMDP's exact dynamic program
        ``MOMDP._get_V_pi``.
        """
        return np.array([self.momdp._get_V_pi(self.momdp.rewards[j], target_policy)
                         for j in range(self.d)])


def _demo():
    np.random.seed(0)                 # reproducible MOMDP (uses global np.random)
    rng = np.random.default_rng(0)    # reproducible rollouts

    S, A, H, d = 5, 3, 5, 4
    momdp = MOMDP(S, A, H, d)
    ope = MOMDP_OPE(momdp)

    # Behavior: uniform random (full support). Target: epsilon-soft-optimal for a
    # fixed preference -- an off-policy target worth evaluating.
    pi0 = uniform_policy(H, S, A)
    pref = np.ones(d) / d
    scalar_reward = np.tensordot(pref, momdp.rewards, axes=(0, 0))   # (H, S, A)
    _, greedy = momdp._get_V_opt(scalar_reward)
    pi1 = epsilon_soft_policy(greedy, eps=0.2)

    V_true = ope.true_value(pi1)
    scalar_true = float(pref @ V_true)
    print("True V^pi1 (per objective):", np.round(V_true, 4))
    print("True scalarized value     :", round(scalar_true, 4))

    methods = {"IS": (False, False), "stepIS": (False, True),
               "WIS": (True, False), "stepWIS": (True, True)}

    print("\nConvergence of |scalarized estimate - truth| as K grows:")
    print(f"{'K':<7}" + "".join(f"{m:>10}" for m in methods))
    for K in [50, 200, 1000, 5000]:
        data = ope.rollout(pi0, K, rng=rng)
        row = []
        for w, s in methods.values():
            est = float(pref @ ope.estimate(data, pi0, pi1, weighted=w, stepwise=s))
            row.append(abs(est - scalar_true))
        print(f"{K:<7}" + "".join(f"{e:>10.5f}" for e in row))

    print("\nRMSE over 300 datasets of size K=200 (scalarized value):")
    R, K = 300, 200
    sq = {m: 0.0 for m in methods}
    for _ in range(R):
        data = ope.rollout(pi0, K, rng=rng)
        for m, (w, s) in methods.items():
            est = float(pref @ ope.estimate(data, pi0, pi1, weighted=w, stepwise=s))
            sq[m] += (est - scalar_true) ** 2
    for m in methods:
        print(f"  {m:<8} RMSE = {(sq[m] / R) ** 0.5:.5f}")
    print("\n(Weighted estimators should match the truth in the limit and show "
          "lower RMSE than their unweighted IS counterparts.)")


if __name__ == "__main__":
    _demo()
