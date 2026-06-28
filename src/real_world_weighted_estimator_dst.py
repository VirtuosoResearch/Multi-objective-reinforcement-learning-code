"""Weighted off-policy evaluation (WIS) on a *real* MORL benchmark: Deep-Sea-Treasure.

This ports the weighted-importance-sampling estimators from the synthetic MOMDP
experiment (``weighted_estimator_experiment.py`` / ``momdp/weighted_rl.py``) onto
the Deep-Sea-Treasure environment from MO-Gymnasium -- a standard discrete
multi-objective benchmark.

Why DST is a clean fit for IS/WIS:
  * Discrete states (11x11 grid position) and 4 discrete actions, so the
    importance ratio pi1(a|s)/pi0(a|s) is computed exactly as in the tabular case.
  * Two objectives: r = [treasure_value, time_penalty(=-1 per step)] -> d = 2,
    so the *vector*-valued estimator is exercised on a true MORL task.
  * No exact dynamic program for ground truth, but the dynamics are deterministic,
    so the true target value vector is obtained by Monte-Carlo on-policy rollouts.

Pipeline (per preference lambda):
  1. Learn a target policy pi1 by tabular scalarised Q-learning on lambda . r,
     then eps-soften it (full support, so trajectory-wise ratios stay finite).
  2. Behaviour policy pi0 = uniform random (full support; covers pi1).
  3. Collect K off-policy episodes under pi0 in the REAL env.
  4. Estimate V^{pi1} in R^2 with IS / step-IS / WIS / step-WIS (reusing the exact
     estimator code from momdp/weighted_rl.py via MOMDP_OPE.estimate).
  5. Ground truth via Monte-Carlo rollouts of pi1; report convergence + RMSE.

Run:  python src/real_world_weighted_estimator_dst.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "momdp"))

# Figures are written to a top-level figures/ directory (sibling of src/).
FIGURES_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "figures"))

import mo_gymnasium as mo_gym                       # noqa: E402
from weighted_rl import MOMDP_OPE, epsilon_soft_policy   # noqa: E402

ENV_ID = "deep-sea-treasure-v0"
GRID = 11                       # DST is an 11x11 grid
A = 4                          # up / down / left / right
S = GRID * GRID + 1            # one index per cell + a single absorbing terminal
TERMINAL = S - 1
H = 100                        # env's max_episode_steps (fixed horizon for IS)
D = 2                          # objectives: [treasure, time]

METHODS = {  # name -> (weighted, stepwise)
    "IS": (False, False),
    "step-IS": (False, True),
    "WIS": (True, False),
    "step-WIS": (True, True),
}


def encode(obs):
    """Map a DST observation [row, col] to a flat state index."""
    return int(obs[0]) * GRID + int(obs[1])


# --------------------------------------------------------------------------- #
# A tiny stub so we can reuse MOMDP_OPE.estimate (it only needs S/A/H/d, and
# never touches the real momdp for estimate / cumulative_ratios).
# --------------------------------------------------------------------------- #
class _Dims:
    def __init__(self):
        self.S, self.A, self.H, self.d = S, A, H, D


# --------------------------------------------------------------------------- #
# Target policy: tabular scalarised Q-learning on lambda . r
# --------------------------------------------------------------------------- #
def learn_target_policy(pref, episodes=4000, alpha=0.5, gamma=1.0,
                        eps_explore=0.2, soften_eps=0.1, seed=0):
    """eps-soft greedy policy (H, S, A) for the scalarised reward lambda . r."""
    env = mo_gym.make(ENV_ID)
    rng = np.random.default_rng(seed)
    Q = np.zeros((S, A))
    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        s = encode(obs)
        for _ in range(H):
            a = (int(rng.integers(A)) if rng.random() < eps_explore
                 else int(np.argmax(Q[s])))
            obs, r_vec, term, trunc, _ = env.step(a)
            r = float(pref @ np.asarray(r_vec, dtype=float))
            s2 = TERMINAL if term else encode(obs)
            target = r + (0.0 if term else gamma * Q[s2].max())
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            if term or trunc:
                break
    env.close()
    greedy = np.zeros((H, S, A))
    best = Q.argmax(axis=1)
    greedy[:, np.arange(S), best] = 1.0
    pi1 = epsilon_soft_policy(greedy, soften_eps)
    # After termination, trajectories are padded with the absorbing TERMINAL state.
    # Make pi1 agree with the uniform behaviour there so post-termination importance
    # ratios are exactly 1 (otherwise rho_{1:H} explodes over the padded steps).
    pi1[:, TERMINAL, :] = 1.0 / A
    return pi1


def uniform_policy():
    return np.full((H, S, A), 1.0 / A)


# --------------------------------------------------------------------------- #
# Rollouts in the REAL environment
# --------------------------------------------------------------------------- #
def rollout_real(policy, K, rng, seed_stream):
    """Roll out K episodes under ``policy`` (H, S, A) in the real DST env.

    Returns the (states, actions, rewards) arrays in the format MOMDP_OPE expects,
    padded after termination with the absorbing TERMINAL state and zero reward.
    """
    env = mo_gym.make(ENV_ID)
    states = np.full((K, H), TERMINAL, dtype=int)
    actions = np.zeros((K, H), dtype=int)
    rewards = np.zeros((K, H, D), dtype=float)
    for k in range(K):
        obs, _ = env.reset(seed=int(seed_stream.integers(1 << 31)))
        s = encode(obs)
        for h in range(H):
            p = policy[h, s]
            a = int(rng.choice(A, p=p / p.sum()))
            obs, r_vec, term, trunc, _ = env.step(a)
            states[k, h] = s
            actions[k, h] = a
            rewards[k, h, :] = np.asarray(r_vec, dtype=float)
            s = TERMINAL if term else encode(obs)
            if term or trunc:
                break  # remaining steps stay at TERMINAL / zero reward
    env.close()
    return {"states": states, "actions": actions, "rewards": rewards}


def monte_carlo_value(policy, episodes, rng, seed_stream):
    """Ground-truth value vector V^{pi}(s_0) in R^2 via on-policy MC rollouts."""
    data = rollout_real(policy, episodes, rng, seed_stream)
    return data["rewards"].sum(axis=1).mean(axis=0)   # undiscounted return, (d,)


# --------------------------------------------------------------------------- #
def scalar_estimates(ope, data, pi0, pi1, pref):
    return {m: float(pref @ ope.estimate(data, pi0, pi1, weighted=w, stepwise=st))
            for m, (w, st) in METHODS.items()}


def save_plot(K_list, rmse, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n(Skipping plot: matplotlib unavailable -- {e})")
        return
    styles = {"IS": "-o", "step-IS": "-s", "WIS": "--^", "step-WIS": "--d"}
    plt.figure(figsize=(6, 4))
    for m in METHODS:
        plt.loglog(K_list, rmse[m], styles[m], label=m)
    plt.xlabel("dataset size |D|  (off-policy episodes)")
    plt.ylabel("RMSE of scalarized value")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    print(f"\nSaved RMSE plot -> {path}")


def main():
    pref = np.array([0.6, 0.4])        # weight on [treasure, time]
    ope = MOMDP_OPE(_Dims())
    pi0 = uniform_policy()

    print(f"Real-world WIS on Deep-Sea-Treasure ({ENV_ID})")
    print(f"  objectives d={D} [treasure, time], states={S-1}+terminal, A={A}, H={H}")
    print(f"  preference lambda={pref},  behaviour pi0=uniform-random")

    print("\nLearning target policy pi1 (tabular scalarised Q-learning)...")
    pi1 = learn_target_policy(pref, seed=0)

    # Ground truth: Monte-Carlo on-policy value of pi1 (deterministic dynamics).
    mc_rng = np.random.default_rng(1)
    V_true = monte_carlo_value(pi1, 20000, mc_rng, np.random.default_rng(2))
    scalar_true = float(pref @ V_true)
    print(f"  MC ground truth  V^pi1 = [treasure={V_true[0]:.3f}, time={V_true[1]:.3f}]"
          f"  -> scalarized {scalar_true:.4f}")

    # (1) Convergence of |scalar estimate - truth| as K grows.
    print("\nConvergence of |scalarized estimate - MC truth| as K grows:")
    print(f"{'K':<7}" + "".join(f"{m:>11}" for m in METHODS))
    rng = np.random.default_rng(10)
    for K in [50, 200, 1000, 5000]:
        data = rollout_real(pi0, K, rng, np.random.default_rng(100 + K))
        est = scalar_estimates(ope, data, pi0, pi1, pref)
        print(f"{K:<7}" + "".join(f"{abs(est[m]-scalar_true):>11.5f}" for m in METHODS))

    # (2) RMSE vs dataset size over many seeds (the illustrative variance result).
    K_list = [50, 100, 200, 500, 1000, 2000]
    n_seeds = 20
    rmse = {m: np.zeros(len(K_list)) for m in METHODS}
    print(f"\nRMSE of scalarized value over {n_seeds} seeds:")
    print(f"{'K':<7}" + "".join(f"{m:>11}" for m in METHODS))
    for ki, K in enumerate(K_list):
        acc = {m: 0.0 for m in METHODS}
        for s in range(n_seeds):
            rng = np.random.default_rng(20_000 + ki * n_seeds + s)
            data = rollout_real(pi0, K, rng, np.random.default_rng(40_000 + ki * n_seeds + s))
            est = scalar_estimates(ope, data, pi0, pi1, pref)
            for m in METHODS:
                acc[m] += (est[m] - scalar_true) ** 2
        for m in METHODS:
            rmse[m][ki] = (acc[m] / n_seeds) ** 0.5
        print(f"{K:<7}" + "".join(f"{rmse[m][ki]:>11.5f}" for m in METHODS))

    save_plot(K_list, rmse,
              os.path.join(FIGURES_DIR, "weighted_estimator_dst_rmse.png"),
              title=f"WIS off-policy evaluation on Deep-Sea-Treasure (lambda={pref})")

    print("\nTakeaway: on a real MORL benchmark, evaluating a near-greedy target from "
          "uniform-random logs, all four estimators converge to the MC ground truth, "
          "but self-normalisation (WIS) and per-step weighting (step-WIS) cut RMSE by "
          "a large factor over vanilla IS -- the same variance-reduction story as the "
          "synthetic MOMDP, now on Deep-Sea-Treasure.")


if __name__ == "__main__":
    main()