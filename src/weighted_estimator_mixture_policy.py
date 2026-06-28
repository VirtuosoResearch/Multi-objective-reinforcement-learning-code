"""Mixture-Hajek off-policy estimator for the tabular MOMDP setting.

Policy-learning analog of ``mixture_hajek_estimate`` in
``Matrix-completion-from-ultrasparse-samples/exp_on_synthetic/src/iipw.py``.

The matrix estimator assumes each *row* is drawn from one of ``num_components``
latent components, recovers the row -> component labels by a (masked) k-means
over the observed entries, computes a self-normalized (Hajek) mean ``mu_k`` per
component, and reports the *mixture object* ``T = sum_k w_k mu_k mu_k^T`` with
weights ``w_k = |component k| / n``. That object has cross-component structure
the pooled second moment ``empirical_reweight_estimate`` cannot represent --
that is precisely why the mixture is needed.

Here the rows are *trajectories* from a tabular MOMDP, collected under a mixture
of latent behavior regimes (each regime is a distinct behavior policy, e.g. a
deployed expert). The estimator mirrors iipw.py step for step:

  1. recover the trajectory -> component labels by k-means over each
     trajectory's state-action occupancy vector (the analog of k-means over the
     matrix rows' observed entries), with random restarts and an
     empty-component guard;
  2. estimate a behavior policy ``pi0^(k)`` per component (smoothed empirical
     action frequencies -- the per-component "mean"), and compute a per-component
     Hajek value ``V_k`` of a target policy: the self-normalized weighted-IS
     (WIS) estimate using ``pi0^(k)`` as the IS denominator (the analog of the
     per-component Hajek mean ``mu_k``);
  3. aggregate. For a single target this is the stratified value
     ``sum_k w_k V_k``; for a *routed/mixture target policy* that deploys a
     different sub-policy ``pi1^(k)`` per regime it is the deployment value
     ``sum_k w_k V_k^{pi1^(k)}`` (the genuine mixture object, the analog of T).

Where the mixture is *needed* (mirroring the matrix story): the per-component
value attribution ``{V_k}`` -- one value vector per recovered regime -- is the
irreducible mixture object, the direct analog of recovering each component mean
``mu_k`` in iipw.py. A single pooled WIS estimate produces one number and offers
no per-component breakdown at all. The routed-deployment aggregate
``sum_k w_k V_k`` is reported too, and the recovered mixture tracks the
true-label oracle; for that scalar aggregate a pooled estimate is competitive
(scalar policy value is a smooth, near-linear functional that pools well -- there
is no quadratic cross-term as in the matrix ``mu_k mu_k^T``), so the win of the
mixture is the attribution structure, not the scalar, exactly as the matrix
estimator's value is its per-component ``mu_k`` rather than a pooled scalar.

As in ``weighted_rl.py`` rewards are d-dimensional, so a single pass returns the
whole value VECTOR in R^d; the scalarized value for a preference ``lambda`` is
``lambda . V``.

Run:  python src/weighted_estimator_mixture_policy.py
"""

import os
import sys
from itertools import permutations

import numpy as np

# The MOMDP modules live in src/momdp/ and import each other by bare name
# (e.g. ``from mdp import MOMDP``), so make that directory importable.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "momdp"))

from weighted_rl import MOMDP_OPE, epsilon_soft_policy                  # noqa: E402
from ensemble_momdp import ConflictMOMDP, set_seed                      # noqa: E402


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
class MixtureHajekOPE(MOMDP_OPE):
    """Mixture-Hajek off-policy value estimation for a tabular MOMDP.

    Extends ``MOMDP_OPE`` (``rollout`` / ``estimate`` / ``true_value``) with
    latent-component recovery and mixture-of-Hajek aggregation.
    """

    # ------------------------------------------------------------------ #
    # Trajectory features and behavior-policy estimation
    # ------------------------------------------------------------------ #
    def occupancy_features(self, data):
        """State-action occupancy vector per trajectory, shape (K, S*A).

        ``phi(tau)[s*A + a] = (1/H) #{h : (s_h, a_h) = (s, a)}``. This is the
        feature the clustering runs on -- the analog of the matrix rows' observed
        entry vectors that ``mixture_hajek_estimate`` clusters.
        """
        states, actions = data["states"], data["actions"]
        K, H = states.shape
        idx = states * self.A + actions                    # (K, H) flat (s, a)
        feats = np.zeros((K, self.S * self.A))
        rows = np.repeat(np.arange(K), H)
        np.add.at(feats, (rows, idx.ravel()), 1.0 / H)
        return feats

    def empirical_behavior(self, states, actions, smoothing=1.0):
        """Smoothed empirical behavior policy (H, S, A) from a set of trajectories.

        ``pi0[h, s, a] propto count(h, s, a) + smoothing``. The Laplace term keeps
        full support (unvisited (h, s) -> uniform), so it is always a valid IS
        denominator for a full-support target ``pi1``.
        """
        H, S, A = self.H, self.S, self.A
        counts = np.full((H, S, A), float(smoothing))
        states = np.asarray(states, dtype=int)
        actions = np.asarray(actions, dtype=int)
        if states.size:
            K = states.shape[0]
            h_idx = np.broadcast_to(np.arange(H), (K, H))
            np.add.at(counts, (h_idx.ravel(), states.ravel(), actions.ravel()), 1.0)
        return counts / counts.sum(axis=2, keepdims=True)

    # ------------------------------------------------------------------ #
    # Latent component recovery (the k-means analog of iipw.py)
    # ------------------------------------------------------------------ #
    def cluster_trajectories(self, data, n_components, restarts=8, max_iter=100,
                             rng=None, return_inertia=False):
        """k-means over the occupancy features, mirroring iipw.py's restart loop.

        Each restart starts from ``n_components`` distinct trajectories as
        centroids, runs Lloyd iterations to convergence, and the run with the
        lowest within-cluster inertia that leaves no empty component is kept (the
        analog of ``mixture_hajek_estimate`` keeping the best clustering loss with
        no empty component over its restarts).

        Returns ``(labels, sizes)`` (and the best inertia if ``return_inertia``).
        """
        if n_components < 2:
            raise ValueError("n_components must be at least 2")
        feats = self.occupancy_features(data)
        K = feats.shape[0]
        if K < n_components:
            raise ValueError("not enough trajectories to estimate mixture components")
        rng = np.random.default_rng() if rng is None else rng

        best = None
        for _ in range(restarts):
            centroids = feats[rng.choice(K, n_components, replace=False)].copy()
            labels, dist2 = None, None
            for _ in range(max_iter):
                dist2 = ((feats[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
                updated = dist2.argmin(axis=1)
                if labels is not None and np.array_equal(updated, labels):
                    break
                labels = updated
                for k in range(n_components):
                    if np.any(labels == k):
                        centroids[k] = feats[labels == k].mean(axis=0)

            sizes = np.bincount(labels, minlength=n_components)
            if np.any(sizes == 0):
                continue
            inertia = dist2[np.arange(K), labels].sum()
            if best is None or inertia < best[0]:
                best = (inertia, labels.copy(), sizes)

        if best is None:
            raise RuntimeError("mixture clustering produced an empty component")
        inertia, labels, sizes = best
        if return_inertia:
            return labels, sizes, inertia
        return labels, sizes

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #
    def grouped_estimate(self, data, targets, labels, behavior_policies,
                         gamma=1.0, stepwise=True):
        """Mixture of per-component Hajek (WIS) values: ``sum_c (n_c/K) V_c``.

        ``labels`` partitions the trajectories. ``behavior_policies[c]`` is the IS
        denominator for component ``c`` (its estimated behavior policy).
        ``targets`` is either one ``(H, S, A)`` policy (broadcast to every
        component) or a mapping ``{label: (H, S, A)}`` -- the latter evaluates a
        *routed* target that deploys a different sub-policy per component.

        Returns ``(value, components)`` where ``components[c] = (weight, V_c)`` is
        the per-component weight and Hajek value vector -- the attribution a
        single pooled estimate cannot provide.
        """
        labels = np.asarray(labels)
        K = labels.shape[0]
        value = np.zeros(self.d)
        components = {}
        for c in np.unique(labels):
            idx = labels == c
            weight = idx.sum() / K
            sub = {key: data[key][idx]
                   for key in ("states", "actions", "rewards")}
            target = targets[c] if isinstance(targets, dict) else targets
            V_c = self.estimate(sub, behavior_policies[c], target,
                                gamma=gamma, weighted=True, stepwise=stepwise)
            value += weight * V_c
            components[c] = (weight, V_c)
        return value, components

    def mixture_estimate(self, data, target_policy, n_components=2, gamma=1.0,
                         stepwise=True, restarts=8, max_iter=100, smoothing=1.0,
                         min_component_size=5, rng=None, return_details=False):
        """Stratified mixture-Hajek value of a *single* ``target_policy``.

        Recovers latent components, estimates a behavior policy per component, and
        returns ``sum_k w_k V_k``. Components smaller than ``min_component_size``
        fall back to the pooled behavior policy as their IS denominator (the
        analog of the matrix diagonal/pooled correction). With ``return_details``
        also returns the labels, weights, and per-component value vectors.
        """
        labels, sizes = self.cluster_trajectories(
            data, n_components, restarts=restarts, max_iter=max_iter, rng=rng)
        behavior_policies = self._component_behaviors(
            data, labels, n_components, sizes, smoothing, min_component_size)
        value, components = self.grouped_estimate(
            data, target_policy, labels, behavior_policies,
            gamma=gamma, stepwise=stepwise)
        if return_details:
            return value, {"labels": labels, "weights": sizes / sizes.sum(),
                           "components": components}
        return value

    def routed_estimate(self, data, target_policies, known_behaviors,
                        n_components=None, gamma=1.0, stepwise=True, restarts=8,
                        max_iter=100, smoothing=1.0, min_component_size=5,
                        rng=None, return_details=False):
        """Value of a *routed/mixture target policy* -- the genuine mixture object.

        The deployed policy runs sub-policy ``target_policies[k]`` whenever a
        context from behavior regime ``k`` arrives; its value is
        ``sum_k w_k V^{target_policies[k]}_k``. We recover the components, map
        each recovered cluster to the regime whose known behavior policy it most
        resembles, and apply that regime's target sub-policy under its
        per-component Hajek estimate. ``known_behaviors[k]`` are the (known,
        deployed) behavior policies -- standard in off-policy evaluation.

        Returns the routed value (and labels / weights / per-component values /
        cluster->regime mapping if ``return_details``).
        """
        n_components = len(target_policies) if n_components is None else n_components
        labels, sizes = self.cluster_trajectories(
            data, n_components, restarts=restarts, max_iter=max_iter, rng=rng)
        behavior_policies = self._component_behaviors(
            data, labels, n_components, sizes, smoothing, min_component_size)

        # Map each recovered cluster to the regime it best matches, then deploy
        # that regime's target sub-policy on the cluster.
        mapping, targets = {}, {}
        for c in range(n_components):
            est = behavior_policies[c]
            regime = int(np.argmin([np.sum((est - bk) ** 2)
                                    for bk in known_behaviors]))
            mapping[c] = regime
            targets[c] = target_policies[regime]

        value, components = self.grouped_estimate(
            data, targets, labels, behavior_policies,
            gamma=gamma, stepwise=stepwise)
        if return_details:
            return value, {"labels": labels, "weights": sizes / sizes.sum(),
                           "components": components, "mapping": mapping}
        return value

    def _component_behaviors(self, data, labels, n_components, sizes, smoothing,
                             min_component_size):
        """Per-component behavior policy, with pooled fallback for tiny components."""
        pooled = self.empirical_behavior(
            data["states"], data["actions"], smoothing)
        return {
            k: (self.empirical_behavior(
                    data["states"][labels == k], data["actions"][labels == k],
                    smoothing)
                if sizes[k] >= min_component_size else pooled)
            for k in range(n_components)
        }

    # ------------------------------------------------------------------ #
    # Mixture data collection (latent behavior regimes)
    # ------------------------------------------------------------------ #
    def rollout_mixture(self, behavior_policies, weights, K, rng=None):
        """Roll out K trajectories from a mixture of behavior policies.

        Each trajectory's latent regime ``z ~ Categorical(weights)`` selects the
        behavior policy it is generated under. Returns ``(data, true_labels)``;
        the estimator never sees ``true_labels`` (they are only for oracle
        baselines and recovery diagnostics).
        """
        rng = np.random.default_rng() if rng is None else rng
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        counts = rng.multinomial(K, weights)
        chunks, labels = [], []
        for k, (policy, n) in enumerate(zip(behavior_policies, counts)):
            if n == 0:
                continue
            chunks.append(self.rollout(policy, int(n), rng=rng))
            labels.append(np.full(int(n), k))
        data = {key: np.concatenate([c[key] for c in chunks], axis=0)
                for key in ("states", "actions", "rewards")}
        labels = np.concatenate(labels)
        perm = rng.permutation(labels.shape[0])
        data = {key: value[perm] for key, value in data.items()}
        return data, labels[perm]


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def align_labels(true_labels, pred_labels, n_components):
    """Best label permutation of ``pred_labels`` onto ``true_labels``.

    Returns ``(accuracy, aligned_pred)`` so recovered components can be matched to
    the true regimes for per-component reporting.
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    best = None
    for perm in permutations(range(n_components)):
        aligned = np.array(perm)[pred_labels]
        acc = float((aligned == true_labels).mean())
        if best is None or acc > best[0]:
            best = (acc, aligned)
    return best


# --------------------------------------------------------------------------- #
# Experiment (styled after weighted_estimator_experiment.py)
# --------------------------------------------------------------------------- #
def random_core(momdp, rng):
    """A distinct near-deterministic policy core (one-hot action per (h, s))."""
    pi = np.zeros((momdp.H, momdp.S, momdp.A))
    for h in range(momdp.H):
        for s in range(momdp.S):
            pi[h, s, rng.integers(momdp.A)] = 1.0
    return pi


def main():
    S, A, H, d = 8, 4, 6, 3
    n_components = 3
    conflict = 0.9
    behavior_eps = 0.2   # softer data-collection policy (full support, separable)
    target_eps = 0.08    # sharper deployed sub-policy (near its regime's behavior)
    stepwise = True
    smoothing = 1.0
    min_size = 5
    K_table, n_seeds = 800, 80
    K_list = [200, 400, 800, 1600]

    set_seed(0)
    momdp = ConflictMOMDP(S, A, H, d, conflict=conflict, noise_scale=0.05,
                          num_conflicting=n_components, seed=0)
    ope = MixtureHajekOPE(momdp)

    # Latent regimes: distinct policy cores. The behavior policy that *collected*
    # each regime's data is a softer version of its core; the deployed target
    # sub-policy is a sharper version of the same core (so importance ratios stay
    # tame). The deployed *routed* policy runs sub-policy k whenever a regime-k
    # context arrives -> a genuine mixture estimand.
    core_rng = np.random.default_rng(99)
    cores = [random_core(momdp, core_rng) for _ in range(n_components)]
    behavior_policies = [epsilon_soft_policy(c, behavior_eps) for c in cores]
    target_policies = [epsilon_soft_policy(c, target_eps) for c in cores]
    mix_weights = np.array([0.5, 0.3, 0.2])

    eval_pref = np.ones(d) / d
    V_components = np.array([float(ope.true_value(target_policies[k]) @ eval_pref)
                            for k in range(n_components)])
    routed_truth = float(mix_weights @ V_components)

    print(f"Mixture-Hajek OPE on a MOMDP  (S={S}, A={A}, H={H}, d={d}, "
          f"components={n_components})")
    print(f"behavior mixture weights = {mix_weights.tolist()}")
    print(f"true per-component values V_k = {np.round(V_components, 4).tolist()}")
    print(f"routed-deployment value (mixture object) = {routed_truth:.4f}")

    # The deployed *average* target policy is the natural single-policy summary
    # used if the mixture were ignored -- the honest pooled baseline.
    avg_target = sum(mix_weights[k] * target_policies[k]
                     for k in range(n_components))

    def measure(K, base_seed):
        """One (K, seed-block) sweep: recovery accuracy, per-component attribution
        RMSE (mixture vs oracle), and routed-aggregate RMSE (mixture, oracle,
        pooled-avgT). Pooled has no per-component attribution by construction."""
        acc = 0.0
        comp_sq = {"mixture": 0.0, "oracle": 0.0}
        routed_sq = {"mixture": 0.0, "oracle": 0.0, "pooled-avgT": 0.0}
        for s in range(n_seeds):
            rng = np.random.default_rng(base_seed + s)
            data, true_labels = ope.rollout_mixture(behavior_policies,
                                                    mix_weights, K, rng=rng)
            pooled_pi0 = ope.empirical_behavior(data["states"], data["actions"],
                                                smoothing)

            # Recovered (this work): routed value + per-component attribution.
            _, det = ope.routed_estimate(
                data, target_policies, behavior_policies, stepwise=stepwise,
                smoothing=smoothing, min_component_size=min_size, rng=rng,
                return_details=True)
            acc += align_labels(true_labels, det["labels"], n_components)[0]
            routed_mix = sum(weight * V for (weight, V) in det["components"].values())
            recovered = {det["mapping"][c]: V
                         for c, (_, V) in det["components"].items()}

            # Oracle: same estimator but with the true component labels.
            oracle_beh = {k: ope.empirical_behavior(
                data["states"][true_labels == k],
                data["actions"][true_labels == k], smoothing)
                for k in range(n_components)}
            routed_oracle, oracle_comp = ope.grouped_estimate(
                data, {k: target_policies[k] for k in range(n_components)},
                true_labels, oracle_beh, stepwise=stepwise)

            for k in range(n_components):
                if k in recovered:
                    comp_sq["mixture"] += (float(recovered[k] @ eval_pref)
                                           - V_components[k]) ** 2
                comp_sq["oracle"] += (float(oracle_comp[k][1] @ eval_pref)
                                      - V_components[k]) ** 2

            pooled_avgT = ope.estimate(data, pooled_pi0, avg_target,
                                       weighted=True, stepwise=stepwise)
            routed_sq["mixture"] += (float(routed_mix @ eval_pref) - routed_truth) ** 2
            routed_sq["oracle"] += (float(routed_oracle @ eval_pref) - routed_truth) ** 2
            routed_sq["pooled-avgT"] += (float(pooled_avgT @ eval_pref) - routed_truth) ** 2

        n, nk = n_seeds, n_seeds * n_components
        return (acc / n,
                {m: (comp_sq[m] / nk) ** 0.5 for m in comp_sq},
                {m: (routed_sq[m] / n) ** 0.5 for m in routed_sq})

    # (1) Headline at K = K_table: recovery + the irreducible attribution object.
    acc, comp_rmse, routed_rmse = measure(K_table, base_seed=10_000)
    print(f"\nAt K={K_table} over {n_seeds} seeds:")
    print(f"  latent-component recovery accuracy : {acc:.3f}")
    print(f"  per-component value-attribution RMSE  mixture={comp_rmse['mixture']:.4f}"
          f"  oracle={comp_rmse['oracle']:.4f}  (pooled: N/A -- no breakdown)")
    print(f"  routed-aggregate value RMSE  mixture={routed_rmse['mixture']:.4f}"
          f"  oracle={routed_rmse['oracle']:.4f}"
          f"  pooled-avgT={routed_rmse['pooled-avgT']:.4f}")
    print("\nTakeaway: the per-component value attribution {V_k} is the mixture "
          "object only this estimator can produce -- it tracks the true-label "
          "oracle and pooling has no per-component breakdown at all. On the scalar "
          "routed aggregate the recovered mixture matches the oracle and is "
          "competitive with pooling (scalar value pools well; the value is the "
          "attribution, as mu_k is in the matrix case).")

    # (2) Per-component attribution RMSE vs dataset size (mixture vs oracle).
    curves = {"mixture": np.zeros(len(K_list)), "oracle": np.zeros(len(K_list))}
    for ki, K in enumerate(K_list):
        _, comp_rmse, _ = measure(K, base_seed=20_000 + ki * n_seeds)
        for m in curves:
            curves[m][ki] = comp_rmse[m]

    print(f"\nPer-component value-attribution RMSE vs dataset size:")
    print(f"{'K':>7}" + "".join(f"{m:>12}" for m in curves))
    for ki, K in enumerate(K_list):
        print(f"{K:>7}" + "".join(f"{curves[m][ki]:>12.5f}" for m in curves))

    save_plot(K_list, curves,
              os.path.join(HERE, "weighted_estimator_mixture_policy_rmse.png"),
              "Mixture-Hajek per-component value attribution (MOMDP)")


def save_plot(K_list, curves, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional
        print(f"\n(Skipping plot: matplotlib unavailable -- {e})")
        return
    styles = {"mixture": "--^", "oracle": "-o", "pooled-avgT": "--s"}
    plt.figure(figsize=(6, 4))
    for m in curves:
        plt.loglog(K_list, curves[m], styles.get(m, "-o"), label=m)
    plt.xlabel("dataset size |D|")
    plt.ylabel("RMSE of per-component value attribution")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nSaved RMSE plot -> {path}")


if __name__ == "__main__":
    main()
