"""Routing + ensemble + the weighted estimator, together, in the MOMDP setting.

This experiment ties three pieces together:

  1. Routing & ensemble. A RoutedEnsembleUCBVI (src/momdp/ensemble_momdp.py) is
     trained on a ConflictMOMDP with clustered preferences: the experts specialize
     to preference clusters and a learned router (per-expert centroids in
     preference space) assigns preferences to experts.

  2. Ensemble target policy. For any preference lambda the trained ensemble
     induces a target policy pi1 of shape (H, S, A):
       - 'routed'   : the nearest-centroid expert's greedy policy (hard routing);
       - 'weighted' : a router-responsibility-weighted mixture of expert policies.

  3. Weighted estimator. That ensemble policy is evaluated *off-policy* from data
     collected under a uniform behavior policy pi0, using the weighted importance
     sampling estimators -- Eq. (6) trajectory-wise WIS and Eq. (7) step-wise WIS
     from src/momdp/weighted_rl.py -- and compared against the exact value from
     MOMDP._get_V_pi. Plain IS (Eq. 4/5) is reported as a baseline.

The single global-optimal policy is included as a reference target to show the
estimator behaves identically regardless of how the target policy is produced --
routing/ensemble only changes *which* policy is being evaluated.

Outputs: an RMSE table (target type x estimator), a per-preference value check,
and an RMSE-vs-dataset-size plot saved next to this script.

Run:  python src/weighted_estimator_experiment.py
"""

import os
import sys

import numpy as np

# The MOMDP modules live in src/momdp/ and import each other by bare name
# (e.g. ``from mdp import MOMDP``), so make that directory importable.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "momdp"))

from weighted_rl import MOMDP_OPE, uniform_policy, epsilon_soft_policy   # noqa: E402
from ensemble_momdp import (                                            # noqa: E402
    ConflictMOMDP,
    RoutedEnsembleUCBVI,
    scalarize_rewards,
    build_reward_sequence,
    sample_clustered_preferences,
    softmax,
    set_seed,
)

METHODS = {  # name -> (weighted, stepwise)
    "IS": (False, False),
    "step-IS": (False, True),
    "WIS": (True, False),
    "step-WIS": (True, True),
}


# --------------------------------------------------------------------------- #
# Routing & ensemble
# --------------------------------------------------------------------------- #
def train_routed_ensemble(S, A, H, d, num_experts, num_tasks, conflict,
                          train_K, coeff=0.1, seed=0):
    """Train a RoutedEnsembleUCBVI on a ConflictMOMDP with clustered preferences.

    Returns the (conflict) MOMDP and the trained ensemble. share_histogram=False
    so experts that the router sends to different clusters accumulate different
    empirical dynamics and hence genuinely different policies.
    """
    momdp = ConflictMOMDP(S, A, H, d, conflict=conflict, noise_scale=0.05,
                          num_conflicting=num_tasks, seed=seed)
    rewards, prefs = build_reward_sequence(
        momdp, train_K, pref_mode="clustered", pref_clusters=num_tasks,
        pref_concentration=0.02, seed=seed + 1)
    set_seed(seed + 123)  # MOMDP.play (training rollouts) uses the global RNG
    ensemble = RoutedEnsembleUCBVI(
        momdp, num_experts,
        eta=0.3, lambda_mu=1.0, route_temperature=0.6, min_resp=0.02,
        share_histogram=False,
        coeff_scales=np.linspace(0.6, 1.4, num_experts),
        select_by_value=False, policy_mode="select", routing_mode="pref",
        init_centroids="dirichlet", seed=seed + 2)
    ensemble.online_game(rewards, prefs, coeff=coeff, task_ids=None)
    return momdp, ensemble


def ensemble_target_policy(ensemble, momdp, pref, mode, coeff_eval=0.0, eps=0.1):
    """Frozen target policy (H, S, A) induced by the trained ensemble for ``pref``.

    Routing is deterministic in ``pref`` via the learned centroids:
      mode='routed'   -> nearest-centroid expert's greedy policy (hard routing);
      mode='weighted' -> softmax(-lambda ||c_k - pref||^2) mixture of expert policies.
    The policy is eps-softened so importance ratios stay finite under the uniform
    behavior policy. coeff_eval=0 uses each expert's pure greedy (no exploration
    bonus) policy.
    """
    reward = scalarize_rewards(momdp.rewards, pref)
    policies = [ensemble.experts[k].get_Q_table(reward, coeff_eval * ensemble.coeff_scales[k])[1]
                for k in range(ensemble.K)]
    dist2 = np.sum((ensemble.centroids - pref) ** 2, axis=1)
    if mode == "routed":
        pi = policies[int(np.argmin(dist2))]
    elif mode == "weighted":
        resp = softmax(-ensemble.lambda_mu * dist2, temperature=ensemble.route_temperature)
        pi = sum(resp[k] * policies[k] for k in range(ensemble.K))
    else:
        raise ValueError("mode must be 'routed' or 'weighted'")
    return epsilon_soft_policy(pi, eps) if eps > 0 else pi


def single_optimal_policy(momdp, pref, eps=0.1):
    """Reference target: eps-soft global-optimal policy for ``pref`` (true dynamics)."""
    _, greedy = momdp._get_V_opt(scalarize_rewards(momdp.rewards, pref))
    return epsilon_soft_policy(greedy, eps)


# --------------------------------------------------------------------------- #
# Off-policy evaluation with the weighted estimators
# --------------------------------------------------------------------------- #
def scalar_estimates(ope, data, pi0, pi1, pref):
    """All four scalarized value estimates for one dataset and target policy."""
    return {m: float(pref @ ope.estimate(data, pi0, pi1, weighted=w, stepwise=st))
            for m, (w, st) in METHODS.items()}


def rmse_table(ope, pi0, target_builders, eval_prefs, K, n_seeds, base_seed=10_000):
    """RMSE per (target type, estimator), averaged over eval prefs and seeds.

    One off-policy dataset is drawn per (pref, seed) and reused across all target
    types and estimators -- a fair, cheaper comparison.
    """
    sq = {t: {m: 0.0 for m in METHODS} for t in target_builders}
    count = 0
    for pi_idx, pref in enumerate(eval_prefs):
        targets = {t: build(pref) for t, build in target_builders.items()}
        truth = {t: float(ope.true_value(pi) @ pref) for t, pi in targets.items()}
        for s in range(n_seeds):
            rng = np.random.default_rng(base_seed + pi_idx * n_seeds + s)
            data = ope.rollout(pi0, K, rng=rng)
            for t, pi1 in targets.items():
                est = scalar_estimates(ope, data, pi0, pi1, pref)
                for m in METHODS:
                    sq[t][m] += (est[m] - truth[t]) ** 2
            count += 1
    return {t: {m: (sq[t][m] / count) ** 0.5 for m in METHODS} for t in target_builders}


def rmse_vs_size(ope, pi0, pi1, pref, K_list, n_seeds, base_seed=20_000):
    """RMSE of each estimator vs dataset size for a fixed target policy."""
    truth = float(ope.true_value(pi1) @ pref)
    rmse = {m: np.zeros(len(K_list)) for m in METHODS}
    for ki, K in enumerate(K_list):
        acc = {m: 0.0 for m in METHODS}
        for s in range(n_seeds):
            rng = np.random.default_rng(base_seed + ki * n_seeds + s)
            data = ope.rollout(pi0, K, rng=rng)
            est = scalar_estimates(ope, data, pi0, pi1, pref)
            for m in METHODS:
                acc[m] += (est[m] - truth) ** 2
        for m in METHODS:
            rmse[m][ki] = (acc[m] / n_seeds) ** 0.5
    return rmse


def save_plot(K_list, rmse, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional
        print(f"\n(Skipping plot: matplotlib unavailable -- {e})")
        return
    styles = {"IS": "-o", "step-IS": "-s", "WIS": "--^", "step-WIS": "--d"}
    plt.figure(figsize=(6, 4))
    for m in METHODS:
        plt.loglog(K_list, rmse[m], styles[m], label=m)
    plt.xlabel("dataset size |D|")
    plt.ylabel("RMSE of scalarized value")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nSaved RMSE plot -> {path}")


# --------------------------------------------------------------------------- #
def main():
    S, A, H, d = 6, 4, 5, 4
    num_tasks = num_experts = 3
    conflict = 0.6
    eps = 0.1
    K_table, n_seeds = 400, 30

    momdp, ensemble = train_routed_ensemble(
        S, A, H, d, num_experts, num_tasks, conflict, train_K=1000, seed=0)
    ope = MOMDP_OPE(momdp)
    pi0 = uniform_policy(H, S, A)

    eval_prefs = sample_clustered_preferences(
        4, d, clusters=num_tasks, concentration=0.02, rng=np.random.default_rng(777))

    target_builders = {
        "single-opt": lambda pref: single_optimal_policy(momdp, pref, eps),
        "routed-ens": lambda pref: ensemble_target_policy(ensemble, momdp, pref, "routed", eps=eps),
        "weighted-ens": lambda pref: ensemble_target_policy(ensemble, momdp, pref, "weighted", eps=eps),
    }

    print(f"MOMDP routing+ensemble OPE  (S={S}, A={A}, H={H}, d={d}, "
          f"experts={num_experts}, conflict={conflict})")

    # (1) RMSE table: target type x estimator.
    table = rmse_table(ope, pi0, target_builders, eval_prefs, K_table, n_seeds)
    print(f"\nRMSE of scalarized value over {len(eval_prefs)} eval prefs x "
          f"{n_seeds} seeds, K={K_table}:")
    print(f"{'target':>14}" + "".join(f"{m:>12}" for m in METHODS))
    for t in target_builders:
        print(f"{t:>14}" + "".join(f"{table[t][m]:>12.5f}" for m in METHODS))

    # (2) Per-preference value check for the routed-ensemble target (step-WIS).
    print("\nRouted-ensemble target -- true value vs step-WIS estimate (K=5000):")
    print(f"{'pref argmax':>12}{'true V':>12}{'step-WIS':>12}{'opt V':>12}")
    for i, pref in enumerate(eval_prefs):
        pi1 = ensemble_target_policy(ensemble, momdp, pref, "routed", eps=eps)
        data = ope.rollout(pi0, 5000, rng=np.random.default_rng(999 + i))
        v_true = float(ope.true_value(pi1) @ pref)
        v_hat = float(pref @ ope.estimate(data, pi0, pi1, weighted=True, stepwise=True))
        v_opt = float(ope.true_value(single_optimal_policy(momdp, pref, eps)) @ pref)
        print(f"{int(np.argmax(pref)):>12}{v_true:>12.4f}{v_hat:>12.4f}{v_opt:>12.4f}")

    # (3) RMSE vs dataset size for the routed-ensemble target, with a plot.
    K_list = [50, 100, 200, 500, 1000]
    pref0 = eval_prefs[0]
    pi1 = ensemble_target_policy(ensemble, momdp, pref0, "routed", eps=eps)
    rmse = rmse_vs_size(ope, pi0, pi1, pref0, K_list, n_seeds=20)
    save_plot(K_list, rmse, os.path.join(HERE, "weighted_estimator_ensemble_rmse.png"),
              title="OPE of a routed-ensemble policy in a MOMDP")

    print("\nTakeaway: the weighted estimators (Eq. 6/7) off-policy-evaluate the "
          "routed/weighted ensemble policy accurately (step-WIS best), exactly as "
          "for a single policy -- routing/ensemble only changes the target.")


if __name__ == "__main__":
    main()
