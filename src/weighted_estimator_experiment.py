"""Routing + ensemble + the weighted estimator, together, in the MOMDP setting.

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
                          train_K, coeff=0.1, seed=0, is_weighting=None,
                          is_eps=0.1, momdp=None):
    """Train a RoutedEnsembleUCBVI on a ConflictMOMDP with clustered preferences.
    """
    if momdp is None:
        momdp = ConflictMOMDP(S, A, H, d, conflict=conflict, noise_scale=0.05,
                              num_conflicting=num_tasks, seed=seed)
    rewards, prefs = build_reward_sequence(
        momdp, train_K, pref_mode="clustered", pref_clusters=num_tasks,
        pref_concentration=0.02, seed=seed + 1)
    set_seed(seed + 123)  # MOMDP.play (training rollouts) uses the global RNG
    ensemble = RoutedEnsembleUCBVI(
        momdp, num_experts,
        eta=0.3, lambda_mu=1.0, route_temperature=0.3, min_resp=0.0,
        share_histogram=False,
        coeff_scales=np.linspace(0.6, 1.4, num_experts),
        select_by_value=False, policy_mode="select", routing_mode="pref",
        init_centroids="dirichlet", is_weighting=is_weighting, is_eps=is_eps,
        seed=seed + 2)
    ensemble.online_game(rewards, prefs, coeff=coeff, task_ids=None)
    return momdp, ensemble


def ensemble_target_policy(ensemble, momdp, pref, mode="routed", coeff_eval=0.1, eps=0.1):
    """Frozen target policy (H, S, A) induced by the trained ensemble for ``pref``.

    Routing is deterministic in ``pref`` via the learned centroids:
      mode='routed'   -> nearest-centroid expert's deployed policy (hard routing);
      mode='weighted' -> softmax(-lambda ||c_k - pref||^2) mixture of expert policies.
    The policy is eps-softened so importance ratios stay finite under the uniform
    behavior policy. coeff_eval>0 uses each expert's *deployed* optimistic UCBVI
    policy (with exploration bonus), for which routed vs weighted genuinely differ.
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
# Learning with vs without importance weighting
# --------------------------------------------------------------------------- #
def optimal_value(momdp, pref):
    """Exact optimal scalarized value V*(s_0) for ``pref`` (true dynamics)."""
    return float(momdp._get_V_opt(scalarize_rewards(momdp.rewards, pref))[0])


def deployed_value(ope, ensemble, momdp, pref, mode, eps):
    """True scalarized value of the ensemble's deployed policy for ``pref``."""
    pi = ensemble_target_policy(ensemble, momdp, pref, mode, eps=eps)
    return float(ope.true_value(pi) @ pref)


def learning_weighting_comparison(ope, momdp, ensembles, eval_prefs, eps,
                                  mode="routed"):
    """Compare learned-policy quality across learning-weighting variants.

    For each eval preference, reports the true scalarized value of each variant's
    deployed (``mode``) policy against the optimal value, and the regret
    (V* - V^pi). Lower regret == the importance weighting produced a better
    ensemble policy. Returns per-variant arrays of value and regret.
    """
    names = list(ensembles)
    value = {n: np.zeros(len(eval_prefs)) for n in names}
    regret = {n: np.zeros(len(eval_prefs)) for n in names}
    v_opt = np.array([optimal_value(momdp, p) for p in eval_prefs])
    for i, pref in enumerate(eval_prefs):
        for n in names:
            v = deployed_value(ope, ensembles[n], momdp, pref, mode, eps)
            value[n][i] = v
            regret[n][i] = v_opt[i] - v

    print(f"\nLearned {mode}-ensemble policy: true value vs optimal "
          f"(over {len(eval_prefs)} eval prefs)")
    print(f"{'pref argmax':>11}" + "".join(f"{n:>12}" for n in names)
          + f"{'optimal':>12}")
    for i, pref in enumerate(eval_prefs):
        print(f"{int(np.argmax(pref)):>11}"
              + "".join(f"{value[n][i]:>12.4f}" for n in names)
              + f"{v_opt[i]:>12.4f}")
    print(f"{'mean regret':>11}"
          + "".join(f"{regret[n].mean():>12.4f}" for n in names)
          + f"{0.0:>12.4f}")
    return value, regret


def save_regret_bar(names, regret, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional
        print(f"\n(Skipping plot: matplotlib unavailable -- {e})")
        return
    means = [regret[n].mean() for n in names]
    errs = [regret[n].std() / max(len(regret[n]), 1) ** 0.5 for n in names]
    plt.figure(figsize=(5, 4))
    plt.bar(range(len(names)), means, yerr=errs, capsize=4,
            color=["#888888", "#4c72b0", "#dd8452"][:len(names)])
    plt.xticks(range(len(names)), names)
    plt.ylabel("mean regret  (V* - V^pi)")
    plt.title(title)
    plt.grid(True, axis="y", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nSaved regret comparison plot -> {path}")


# --------------------------------------------------------------------------- #
def main():
    S, A, H, d = 12, 4, 6, 4
    num_tasks = num_experts = 3
    conflict = 0.7
    eps = 0.1
    K_table, n_seeds = 400, 30

    # Train three ensembles that differ ONLY in the importance weighting applied
    # to their learning updates, on a shared MOMDP + training sequence (same seed).
    #   raw : no weighting (each trajectory updates an expert with weight resp_k);
    #   IS  : weight resp_k by the importance ratio rho_{j->k};
    #   WIS : the self-normalized (weighted-importance) version of that ratio.
    variants = {"raw": None, "IS": "IS", "WIS": "WIS"}
    momdp = ConflictMOMDP(S, A, H, d, conflict=conflict, noise_scale=0.05,
                          num_conflicting=num_tasks, seed=0)
    ensembles = {}
    for name, mode in variants.items():
        _, ensembles[name] = train_routed_ensemble(
            S, A, H, d, num_experts, num_tasks, conflict, train_K=600, seed=0,
            is_weighting=mode, is_eps=eps, momdp=momdp)
    ope = MOMDP_OPE(momdp)
    pi0 = uniform_policy(H, S, A)

    eval_prefs = sample_clustered_preferences(
        4, d, clusters=num_tasks, concentration=0.02, rng=np.random.default_rng(777))

    print(f"MOMDP ensemble learning: weighted-IS vs raw  (S={S}, A={A}, H={H}, "
          f"d={d}, experts={num_experts}, conflict={conflict})")

    # (1) Headline: does importance weighting during learning yield a better
    #     ensemble policy? Compare true value / regret of each variant's policy.
    #     Deployment uses hard routing (nearest-centroid expert).
    _, regret = learning_weighting_comparison(
        ope, momdp, ensembles, eval_prefs, eps, mode="routed")
    save_regret_bar(list(variants), regret,
                    os.path.join(HERE, "ensemble_learning_weighting_regret.png"),
                    title="Ensemble learning: weighted-IS vs raw (routed policy)")

    # (2) Sanity check: the weighted off-policy estimator still recovers each
    #     learned policy's value. RMSE (target type x estimator) for the routed
    #     policy of the raw vs WIS-learned ensemble.
    target_builders = {
        f"routed-{n}": (lambda pref, e=ensembles[n]:
                        ensemble_target_policy(e, momdp, pref, "routed", eps=eps))
        for n in ("raw", "WIS")
    }
    table = rmse_table(ope, pi0, target_builders, eval_prefs, K_table, n_seeds)
    print(f"\nOPE RMSE of scalarized value over {len(eval_prefs)} eval prefs x "
          f"{n_seeds} seeds, K={K_table}:")
    print(f"{'target':>14}" + "".join(f"{m:>12}" for m in METHODS))
    for t in target_builders:
        print(f"{t:>14}" + "".join(f"{table[t][m]:>12.5f}" for m in METHODS))

    print("\nTakeaway: 'raw' learns the ensemble with no off-policy correction; "
          "'IS'/'WIS' reweight each expert's learning updates by the importance "
          "ratio (WIS self-normalized). Lower mean regret == the weighting "
          "produced a better ensemble policy; the OPE table confirms the weighted "
          "estimator still recovers each learned policy off-policy.")


if __name__ == "__main__":
    main()
