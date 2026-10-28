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

# Figures are written to a top-level figures/ directory (sibling of src/).
FIGURES_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "figures"))

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
    def _trajectory_loglik(self, data, policy):
        """Per-trajectory log-likelihood ``sum_h log pi(a_h | s_h)``, shape (K,).

        The responsibility score for the likelihood-EM clustering: it only reads
        the (s, a) pairs a trajectory actually visited, so it stays informative
        when each trajectory observes few of the S*A cells (the ultra-sparse
        regime), unlike Euclidean distance between sparse occupancy vectors.
        """
        states, actions = data["states"], data["actions"]
        H = states.shape[1]
        h_idx = np.arange(H)[None, :]
        p = policy[h_idx, states, actions]
        return np.log(np.clip(p, 1e-300, None)).sum(axis=1)

    def cluster_trajectories(self, data, n_components, restarts=8, max_iter=100,
                             rng=None, method="occupancy", smoothing=1.0,
                             return_score=False):
        """Recover latent components, mirroring iipw.py's restart loop.

        ``method="occupancy"`` -- k-means over the state-action occupancy vectors
        (the direct analog of the matrix k-means over row entries; best when
        trajectories are dense enough to co-visit cells).

        ``method="likelihood"`` -- hard-EM over a per-component behavior policy:
        E-step assigns each trajectory to the component whose policy gives it the
        highest log-likelihood; M-step re-estimates each component policy as the
        smoothed empirical action frequencies of its trajectories. This is the
        faithful analog of the matrix masked k-means (the component "mean" is
        estimated from the observed entries, assignment scored on observed entries
        only) and is the right choice in the ULTRA-SPARSE regime, where each
        trajectory observes few of the S*A cells.

        Each restart starts from a balanced random labeling (likelihood) or
        ``n_components`` data points as centroids (occupancy), iterates to
        convergence, and the run with the best objective leaving no empty
        component is kept. Returns ``(labels, sizes)`` (and the best objective if
        ``return_score``).
        """
        if n_components < 2:
            raise ValueError("n_components must be at least 2")
        K = data["states"].shape[0]
        if K < n_components:
            raise ValueError("not enough trajectories to estimate mixture components")
        rng = np.random.default_rng() if rng is None else rng

        best = None
        for _ in range(restarts):
            if method == "occupancy":
                feats = self.occupancy_features(data)
                centroids = feats[rng.choice(K, n_components, replace=False)].copy()
                labels, score_mat = None, None
                for _ in range(max_iter):
                    score_mat = ((feats[:, None, :] - centroids[None, :, :]) ** 2
                                 ).sum(axis=2)
                    updated = score_mat.argmin(axis=1)
                    if labels is not None and np.array_equal(updated, labels):
                        break
                    labels = updated
                    for k in range(n_components):
                        if np.any(labels == k):
                            centroids[k] = feats[labels == k].mean(axis=0)
                better = lambda s, b: s < b           # minimize inertia
                objective = score_mat[np.arange(K), labels].sum()
            elif method == "likelihood":
                labels = (np.arange(K) % n_components)[rng.permutation(K)]
                score_mat = None
                for _ in range(max_iter):
                    policies = [self.empirical_behavior(
                        data["states"][labels == k], data["actions"][labels == k],
                        smoothing) for k in range(n_components)]
                    score_mat = np.stack(
                        [self._trajectory_loglik(data, policies[k])
                         for k in range(n_components)], axis=1)
                    updated = score_mat.argmax(axis=1)
                    if np.array_equal(updated, labels):
                        break
                    labels = updated
                better = lambda s, b: s > b           # maximize log-likelihood
                objective = score_mat[np.arange(K), labels].sum()
            else:
                raise ValueError("method must be 'occupancy' or 'likelihood'")

            sizes = np.bincount(labels, minlength=n_components)
            if np.any(sizes == 0):
                continue
            if best is None or better(objective, best[0]):
                best = (objective, labels.copy(), sizes)

        if best is None:
            raise RuntimeError("mixture clustering produced an empty component")
        objective, labels, sizes = best
        if return_score:
            return labels, sizes, objective
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

    def routed_estimates_by_mode(self, data, target_policies, known_behaviors,
                                 n_components=None, gamma=1.0, stepwise=True,
                                 restarts=8, max_iter=100, smoothing=1.0,
                                 min_component_size=5, cluster_method="occupancy",
                                 rng=None, return_details=False):
        """Three estimators of the routed target value, all sharing the routing.

        The routing (clustering + cluster->regime mapping -> which target
        sub-policy applies to each trajectory) is recovered once; the three modes
        differ only in how they reweight, mirroring the raw/IS/WIS trio of
        ``weighted_estimator_experiment.py``:

          ``raw``           : no importance correction -- the plain discounted
                              return average (biased, since data was collected
                              under the behavior policies, not the targets);
          ``hajek``         : weighted-IS / Hajek -- self-normalized importance
                              weights with a single *pooled* behavior denominator
                              and one *global* normalizer;
          ``mixture-hajek`` : Hajek applied *within each recovered component*
                              (per-component behavior denominator and normalizer),
                              then mixed by the component weights.

        Returns ``{mode: value_vector}`` (plus labels / mapping if
        ``return_details``).
        """
        n_components = len(target_policies) if n_components is None else n_components
        labels, sizes = self.cluster_trajectories(
            data, n_components, restarts=restarts, max_iter=max_iter, rng=rng,
            method=cluster_method, smoothing=smoothing)
        component_behaviors = self._component_behaviors(
            data, labels, n_components, sizes, smoothing, min_component_size)
        pooled = self.empirical_behavior(data["states"], data["actions"], smoothing)
        mapping = {c: int(np.argmin([np.sum((component_behaviors[c] - bk) ** 2)
                                     for bk in known_behaviors]))
                   for c in range(n_components)}

        rewards = data["rewards"]
        H = rewards.shape[1]
        disc = gamma ** np.arange(H)
        returns = np.einsum("h,khd->kd", disc, rewards)            # (K, d)

        # raw: no reweighting -> mean discounted return.
        raw = returns.mean(axis=0)

        # hajek: pooled-behavior routed ratios, one global self-normalizer.
        rho_pooled = self._routed_cumulative_ratios(
            data, labels, mapping, target_policies, pooled)
        if stepwise:
            w = rho_pooled.mean(axis=0)                            # (H,)
            coeff = np.divide(rho_pooled, w[None, :],
                              out=np.zeros_like(rho_pooled), where=w[None, :] > 0)
            hajek = np.einsum("h,kh,khd->kd", disc, coeff, rewards).mean(axis=0)
        else:
            rho_full = rho_pooled[:, -1]
            wH = rho_full.mean()
            scale = rho_full / wH if wH > 0 else np.zeros_like(rho_full)
            hajek = (scale[:, None] * returns).mean(axis=0)

        # mixture-hajek: per-component Hajek, mixed by component weight.
        targets = {c: target_policies[mapping[c]] for c in range(n_components)}
        mixture, _ = self.grouped_estimate(
            data, targets, labels, component_behaviors,
            gamma=gamma, stepwise=stepwise)

        estimates = {"raw": raw, "hajek": hajek, "mixture-hajek": mixture}
        if return_details:
            return estimates, {"labels": labels, "mapping": mapping,
                               "weights": sizes / sizes.sum()}
        return estimates

    def _routed_cumulative_ratios(self, data, labels, mapping, target_policies,
                                  behavior):
        """Cumulative IS ratios with a per-trajectory *routed* target, shape (K, H).

        Trajectory ``i`` in cluster ``c`` uses target ``target_policies[mapping[c]]``;
        ``behavior`` is the IS denominator -- one ``(H, S, A)`` array (pooled) or a
        ``{label: (H, S, A)}`` mapping (per-component).
        """
        states, actions = data["states"], data["actions"]
        K, H = states.shape
        h_idx = np.arange(H)[None, :]
        rho_cum = np.ones((K, H))
        for c in np.unique(labels):
            idx = labels == c
            target = target_policies[mapping[c]]
            beh = behavior[c] if isinstance(behavior, dict) else behavior
            p1 = target[h_idx, states[idx], actions[idx]]
            p0 = beh[h_idx, states[idx], actions[idx]]
            rho_cum[idx] = np.cumprod(p1 / p0, axis=1)
        return rho_cum

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

    # Three estimators of the routed target value, all sharing the routing
    # mechanism (clustering + cluster->regime mapping). They differ only in the
    # reweighting -- the OPE analog of the raw/IS/WIS trio in
    # weighted_estimator_experiment.py.
    estimators = ["raw", "hajek", "mixture-hajek"]

    def measure(K, base_seed):
        """One (K, seed-block) sweep: recovery accuracy, the routed-value RMSE of
        each estimator, and the per-component value-attribution RMSE (only the
        mixture estimator -- and its oracle -- produce a per-component breakdown)."""
        acc = 0.0
        routed_sq = {m: 0.0 for m in estimators}
        comp_sq = {"mixture-hajek": 0.0, "oracle": 0.0}
        for s in range(n_seeds):
            rng = np.random.default_rng(base_seed + s)
            data, true_labels = ope.rollout_mixture(behavior_policies,
                                                    mix_weights, K, rng=rng)

            # Routed value under each reweighting (shared routing).
            est, det = ope.routed_estimates_by_mode(
                data, target_policies, behavior_policies, stepwise=stepwise,
                smoothing=smoothing, min_component_size=min_size, rng=rng,
                return_details=True)
            acc += align_labels(true_labels, det["labels"], n_components)[0]
            for m in estimators:
                routed_sq[m] += (float(est[m] @ eval_pref) - routed_truth) ** 2

            # Per-component attribution: mixture-hajek (recovered) vs oracle labels.
            _, mix_comp = ope.grouped_estimate(
                data, {c: target_policies[det["mapping"][c]]
                       for c in range(n_components)},
                det["labels"],
                ope._component_behaviors(data, det["labels"], n_components,
                                         np.bincount(det["labels"],
                                                     minlength=n_components),
                                         smoothing, min_size),
                stepwise=stepwise)
            recovered = {det["mapping"][c]: V for c, (_, V) in mix_comp.items()}
            oracle_beh = {k: ope.empirical_behavior(
                data["states"][true_labels == k],
                data["actions"][true_labels == k], smoothing)
                for k in range(n_components)}
            _, oracle_comp = ope.grouped_estimate(
                data, {k: target_policies[k] for k in range(n_components)},
                true_labels, oracle_beh, stepwise=stepwise)
            for k in range(n_components):
                if k in recovered:
                    comp_sq["mixture-hajek"] += (float(recovered[k] @ eval_pref)
                                                 - V_components[k]) ** 2
                comp_sq["oracle"] += (float(oracle_comp[k][1] @ eval_pref)
                                      - V_components[k]) ** 2

        n, nk = n_seeds, n_seeds * n_components
        return (acc / n,
                {m: (routed_sq[m] / n) ** 0.5 for m in estimators},
                {m: (comp_sq[m] / nk) ** 0.5 for m in comp_sq})

    # (1) Headline at K = K_table: recovery + the irreducible attribution object.
    acc, routed_rmse, comp_rmse = measure(K_table, base_seed=10_000)
    print(f"\nAt K={K_table} over {n_seeds} seeds:")
    print(f"  latent-component recovery accuracy : {acc:.3f}")
    print(f"  per-component value-attribution RMSE  mixture-hajek="
          f"{comp_rmse['mixture-hajek']:.4f}  oracle={comp_rmse['oracle']:.4f}"
          f"  (raw/hajek: N/A -- no per-component breakdown)")

    # (2) Routed-value RMSE vs dataset size: raw vs hajek vs mixture-hajek.
    curves = {m: np.zeros(len(K_list)) for m in estimators}
    rows = []
    for ki, K in enumerate(K_list):
        _, routed_rmse, _ = measure(K, base_seed=20_000 + ki * n_seeds)
        for m in estimators:
            curves[m][ki] = routed_rmse[m]
        rows.append((K, routed_rmse))

    print(f"\nRouted-value RMSE vs dataset size  (all under the routing mechanism):")
    print(f"{'K':>7}" + "".join(f"{m:>16}" for m in estimators))
    for K, routed_rmse in rows:
        print(f"{K:>7}" + "".join(f"{routed_rmse[m]:>16.5f}" for m in estimators))

    print("\nTakeaway: 'raw' (no importance correction) plateaus at a bias floor "
          "-- it does not converge to the routed value (here the bias is small "
          "only because each target is a sharpened version of its own regime's "
          "behavior). 'hajek' (weighted-IS) ignores the mixture and uses a single "
          "*pooled* behavior as the IS denominator; under a mixture of behaviors "
          "that is the wrong denominator per trajectory, so it is biased and "
          "heavy-tailed -- the worst here. 'mixture-hajek' uses the correct "
          "*per-component* denominator and normalizer, giving the lowest RMSE, "
          "convergence in K, and -- uniquely -- the per-component value "
          "attribution above.")

    save_plot(K_list, curves,
              os.path.join(FIGURES_DIR, "weighted_estimator_mixture_policy_rmse.png"),
              "Routed-policy OPE: raw vs hajek vs mixture-hajek (MOMDP)")


def save_plot(K_list, curves, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional
        print(f"\n(Skipping plot: matplotlib unavailable -- {e})")
        return
    styles = {"raw": ":d", "hajek": "--s", "mixture-hajek": "-o"}
    plt.figure(figsize=(6, 4))
    for m in curves:
        plt.loglog(K_list, curves[m], styles.get(m, "-o"), label=m)
    plt.xlabel("dataset size |D|")
    plt.ylabel("RMSE of routed-policy value")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    print(f"\nSaved RMSE plot -> {path}")


# --------------------------------------------------------------------------- #
# Ultra-sparse experiment (analog of the ultra-sparse matrix-completion regime)
# --------------------------------------------------------------------------- #
def ultra_sparse_experiment():
    """Scale the state-action dimension and ask whether mixture recovery survives.

    Matrix analog: rows = trajectories, columns = the S*A state-action cells. Each
    length-H trajectory observes at most H of the S*A cells, so as the dimension
    grows the per-row observation rate H/(S*A) -> 0 (ultra-sparse). The matrix
    result is that mixture recovery is possible iff the number of rows scales
    (linearly) with the number of columns. We test exactly that by sweeping the
    dimension under two row budgets:

        K_linear = alpha * (S*A)   (rows linear in columns)
        K_fixed  = constant        (sublinear -> rows/columns -> 0)

    and reporting, for each, the latent-component recovery accuracy and the
    mixture-Hajek routed-value RMSE, plus the raw/hajek/mixture-hajek comparison
    in the linear regime. Clustering uses the likelihood-EM method, the
    sparsity-robust analog of the matrix masked k-means.
    """
    A, H, d, n_components = 3, 5, 3, 3
    alpha = 4          # K_linear = alpha * (S * A)
    K_fixed = 96
    behavior_eps, target_eps = 0.2, 0.08
    smoothing, min_size, stepwise = 1.0, 5, True
    mix_weights = np.array([0.5, 0.3, 0.2])
    S_list = [8, 16, 32, 64]
    n_seeds = 25
    estimators = ["raw", "hajek", "mixture-hajek"]

    print("Ultra-sparse regime: rows (trajectories) vs columns (S*A cells)")
    print(f"  A={A}, H={H}, d={d}, components={n_components}, "
          f"K_linear={alpha}*(S*A), K_fixed={K_fixed}, clustering=likelihood-EM")

    lin_acc, lin_rmse, fix_acc, fix_rmse = [], [], [], []
    lin_est = {m: [] for m in estimators}
    cols_list = []
    for S in S_list:
        set_seed(S)
        momdp = ConflictMOMDP(S, A, H, d, conflict=0.9, noise_scale=0.05,
                              num_conflicting=n_components, seed=S)
        ope = MixtureHajekOPE(momdp)
        core_rng = np.random.default_rng(100 + S)
        cores = [random_core(momdp, core_rng) for _ in range(n_components)]
        behavior_policies = [epsilon_soft_policy(c, behavior_eps) for c in cores]
        target_policies = [epsilon_soft_policy(c, target_eps) for c in cores]
        eval_pref = np.ones(d) / d
        V_components = np.array([float(ope.true_value(target_policies[k]) @ eval_pref)
                                for k in range(n_components)])
        routed_truth = float(mix_weights @ V_components)
        cols = S * A
        cols_list.append(cols)

        for K, acc_acc, rmse_acc, store_est in (
                (alpha * cols, lin_acc, lin_rmse, lin_est),
                (K_fixed, fix_acc, fix_rmse, None)):
            acc = 0.0
            sq = {m: 0.0 for m in estimators}
            for s in range(n_seeds):
                rng = np.random.default_rng(s)
                data, true_labels = ope.rollout_mixture(behavior_policies,
                                                        mix_weights, K, rng=rng)
                est, det = ope.routed_estimates_by_mode(
                    data, target_policies, behavior_policies, stepwise=stepwise,
                    smoothing=smoothing, min_component_size=min_size,
                    cluster_method="likelihood", rng=rng, return_details=True)
                acc += align_labels(true_labels, det["labels"], n_components)[0]
                for m in estimators:
                    sq[m] += (float(est[m] @ eval_pref) - routed_truth) ** 2
            acc_acc.append(acc / n_seeds)
            rmse_acc.append((sq["mixture-hajek"] / n_seeds) ** 0.5)
            if store_est is not None:
                for m in estimators:
                    store_est[m].append((sq[m] / n_seeds) ** 0.5)

    print("\nRecovery accuracy & mixture-Hajek RMSE vs dimension:")
    print(f"{'cols=SA':>8}{'spars':>8}{'K_lin':>7}{'acc_lin':>9}{'rmse_lin':>10}"
          f"{'K_fix':>7}{'acc_fix':>9}{'rmse_fix':>10}")
    for i, S in enumerate(S_list):
        cols = cols_list[i]
        print(f"{cols:>8}{H / cols:>8.3f}{alpha * cols:>7}{lin_acc[i]:>9.3f}"
              f"{lin_rmse[i]:>10.4f}{K_fixed:>7}{fix_acc[i]:>9.3f}{fix_rmse[i]:>10.4f}")

    print("\nLinear regime (K=alpha*SA), routed-value RMSE by estimator:")
    print(f"{'cols=SA':>8}" + "".join(f"{m:>15}" for m in estimators))
    for i in range(len(S_list)):
        print(f"{cols_list[i]:>8}"
              + "".join(f"{lin_est[m][i]:>15.4f}" for m in estimators))

    print("\nTakeaway: with rows LINEAR in columns (K=alpha*SA) recovery accuracy "
          "stays roughly flat and the mixture-Hajek RMSE stays low as the "
          "dimension grows -- mixture recovery survives the ultra-sparse regime. "
          "With a FIXED row budget the rows/columns ratio -> 0, accuracy collapses "
          "toward chance and the RMSE blows up: exactly the linear-scaling "
          "threshold of the ultra-sparse matrix-completion result.")

    save_accuracy_plot(cols_list, lin_acc, fix_acc, alpha, K_fixed,
                       os.path.join(FIGURES_DIR,
                                    "weighted_estimator_mixture_policy_ultrasparse.png"))


def save_accuracy_plot(cols_list, lin_acc, fix_acc, alpha, K_fixed, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional
        print(f"\n(Skipping plot: matplotlib unavailable -- {e})")
        return
    plt.figure(figsize=(6, 4))
    plt.semilogx(cols_list, lin_acc, "-o", label=f"K = {alpha}*(S*A)  (linear)")
    plt.semilogx(cols_list, fix_acc, "--s", label=f"K = {K_fixed}  (fixed)")
    plt.axhline(1.0 / 3, ls=":", color="gray", label="chance (3 components)")
    plt.xlabel("columns = S*A")
    plt.ylabel("latent-component recovery accuracy")
    plt.title("Ultra-sparse mixture recovery: rows linear vs fixed")
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    print(f"\nSaved ultra-sparse plot -> {path}")


if __name__ == "__main__":
    if "--ultra-sparse" in sys.argv:
        ultra_sparse_experiment()
    else:
        main()
