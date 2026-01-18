import random
import numpy as np
import matplotlib.pyplot as plt

from mdp import MOMDP
from mdp import MOUCBVI


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)

def softmax(scores, temperature=1.0):
    temp = max(temperature, 1e-8)
    scaled = scores / temp
    scaled = scaled - np.max(scaled)
    exps = np.exp(scaled)
    return exps / np.sum(exps)


class ConflictMOMDP(MOMDP):
    def __init__(self, S, A, H, d, conflict=0.0, noise_scale=0.05, seed=None):
        self.conflict = conflict
        self.noise_scale = noise_scale
        set_seed(seed)
        super(ConflictMOMDP, self).__init__(S, A, H, d)

    def _init_rewards(self, dist=None):
        if dist == "random":
            base = np.random.rand(self.H, self.S, self.A)
            anti = 1.0 - base
            rewards = np.zeros((self.d, self.H, self.S, self.A))
            order = np.random.permutation(self.d)
            for idx, j in enumerate(order):
                if idx < self.d // 2:
                    target = base
                else:
                    target = (1.0 - self.conflict) * base + self.conflict * anti
                if self.noise_scale > 0.0:
                    noise = self.noise_scale * (np.random.rand(self.H, self.S, self.A) - 0.5)
                    target = np.clip(target + noise, 0.0, 1.0)
                rewards[j] = target
            self.rewards = rewards


def sample_preferences(num, d, alpha=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if alpha is None:
        alpha = np.ones(d)
    return rng.dirichlet(alpha, size=num)

def sample_clustered_preferences(num, d, clusters=None, concentration=0.05, rng=None, return_ids=False):
    if rng is None:
        rng = np.random.default_rng()
    if clusters is None:
        clusters = d
    clusters = max(1, min(clusters, d))
    cluster_ids = rng.integers(0, clusters, size=num)
    prefs = np.zeros((num, d))
    for i, cluster_id in enumerate(cluster_ids):
        alpha = np.ones(d) * concentration
        alpha[cluster_id] = 1.0
        prefs[i] = rng.dirichlet(alpha)
    if return_ids:
        return prefs, cluster_ids
    return prefs


def scalarize_rewards(rewards, pref):
    pref = pref / np.sum(pref)
    return np.sum(rewards * pref.reshape((-1, 1, 1, 1)), axis=0)


def build_reward_sequence(
    momdp,
    num_episodes,
    pref_mode="dirichlet",
    pref_alpha=None,
    pref_clusters=None,
    pref_concentration=0.05,
    return_tasks=False,
    seed=None,
):
    rng = np.random.default_rng(seed)
    if pref_mode == "dirichlet":
        prefs = sample_preferences(num_episodes, momdp.d, alpha=pref_alpha, rng=rng)
        task_ids = None
    elif pref_mode == "clustered":
        prefs, task_ids = sample_clustered_preferences(
            num_episodes,
            momdp.d,
            clusters=pref_clusters,
            concentration=pref_concentration,
            rng=rng,
            return_ids=True,
        )
    else:
        raise ValueError("Unknown pref_mode: {}".format(pref_mode))
    rewards = [scalarize_rewards(momdp.rewards, pref) for pref in prefs]
    if return_tasks:
        return rewards, prefs, task_ids
    return rewards, prefs


def get_V_pi(reward, policy, transit):
    S = reward.shape[1]
    H = reward.shape[0]
    p_x = np.zeros(S)
    p_x[0] = 1.0
    value = 0.0
    for h in range(H):
        p_xa = policy[h] * p_x.reshape((S, 1))
        value += np.sum(p_xa * reward[h])
        p_y = np.sum(p_xa.reshape((S, -1, 1)) * transit, axis=(0, 1))
        p_x = p_y
    return value


def get_V_opt(reward, transit):
    S = reward.shape[1]
    H = reward.shape[0]
    V = np.zeros(S)
    pi = np.zeros((H, S, reward.shape[2]))
    for h in np.arange(H - 1, -1, -1):
        Q = reward[h] + np.sum(transit * V.reshape((1, 1, S)), axis=2)
        V = np.amax(Q, axis=1)
        pi[h] = np.eye(reward.shape[2])[np.argmax(Q, axis=1)]
    return V[0], pi


def regret_with_transits(rewards, policies, transits):
    regret = 0.0
    regrets = [0.0]
    for reward, policy, transit in zip(rewards, policies, transits):
        V_opt, _ = get_V_opt(reward, transit)
        V_pi = get_V_pi(reward, policy, transit)
        regret += (V_opt - V_pi)
        regrets.append(regret)
    return regrets


def regret_fixed_policy(rewards, policy, transits):
    regret = 0.0
    regrets = [0.0]
    for reward, transit in zip(rewards, transits):
        V_opt, _ = get_V_opt(reward, transit)
        V_pi = get_V_pi(reward, policy, transit)
        regret += (V_opt - V_pi)
        regrets.append(regret)
    return regrets


def run_ucbvi_on_rewards(momdp, rewards, coeff=1.0, task_ids=None):
    moucbvi = MOUCBVI(momdp)
    policies = []
    for idx, reward in enumerate(rewards):
        _, policy = moucbvi.get_Q_table(reward, coeff)
        task_id = None if task_ids is None else task_ids[idx]
        if task_id is None:
            _, trajectory = momdp.play(reward, policy)
        else:
            _, trajectory = momdp.play(reward, policy, task_id=task_id)
        moucbvi.update_history(trajectory)
        policies.append(policy)
    return policies


class UCBVIExpert(object):
    def __init__(self, momdp, histogram=None):
        super(UCBVIExpert, self).__init__()
        self.momdp = momdp
        self.S = self.momdp.S
        self.A = self.momdp.A
        self.H = self.momdp.H
        self.d = self.momdp.d
        if histogram is None:
            self.histogram = np.ones((self.S, self.A, self.S))
        else:
            self.histogram = histogram

    def update_history(self, episode_trajectory, weight=1.0):
        for (x, a, y) in episode_trajectory:
            self.histogram[x, a, y] += weight

    def get_empi_transit(self):
        empi_transit = np.copy(self.histogram)
        empi_transit /= np.sum(empi_transit, axis=2, keepdims=True)
        return empi_transit

    def get_bonus(self, coeff=1.0):
        bonus = np.sqrt(np.minimum(self.d, self.S) * self.H ** 2 / np.sum(self.histogram, axis=2))
        return bonus * coeff

    def get_Q_table(self, reward, coeff=1.0):
        bonus = self.get_bonus(coeff)
        empi_transit = self.get_empi_transit()
        pi = np.zeros((self.H, self.S, self.A))
        Q = np.zeros((self.H, self.S, self.A))
        V = np.zeros(self.S)
        for h in np.arange(self.H - 1, -1, -1):
            Q[h] = reward[h] + bonus + np.sum(empi_transit * V.reshape((1, 1, self.S)), axis=2)
            Q[h] = np.minimum(Q[h], self.H)
            V = np.amax(Q[h], axis=1)
            pi[h] = np.eye(self.A)[np.argmax(Q[h], axis=1)]
        return Q, pi

    def eval_policy_value(self, reward, policy):
        empi_transit = self.get_empi_transit()
        p_x = np.zeros(self.S)
        p_x[0] = 1.0
        value = 0.0
        for h in range(self.H):
            p_xa = policy[h] * p_x.reshape((self.S, 1))
            value += np.sum(p_xa * reward[h])
            p_y = np.sum(p_xa.reshape((self.S, self.A, 1)) * empi_transit, axis=(0, 1))
            p_x = p_y
        return value


class RoutedEnsembleUCBVI(object):
    def __init__(
        self,
        momdp,
        num_experts,
        eta=0.2,
        lambda_mu=0.0,
        route_temperature=0.5,
        min_resp=0.0,
        share_histogram=False,
        coeff_scales=None,
        select_by_value=True,
        routing_mode="score",
        random_alpha=1.0,
        init_centroids="dirichlet",
        update_self_only=False,
        self_update_scale=None,
        seed=None,
    ):
        super(RoutedEnsembleUCBVI, self).__init__()
        self.momdp = momdp
        self.S = self.momdp.S
        self.A = self.momdp.A
        self.H = self.momdp.H
        self.d = self.momdp.d
        self.K = num_experts
        self.eta = eta
        self.lambda_mu = lambda_mu
        self.route_temperature = route_temperature
        self.min_resp = min_resp
        self.share_histogram = share_histogram
        self.select_by_value = select_by_value
        self.routing_mode = routing_mode
        self.random_alpha = random_alpha
        self.update_self_only = update_self_only
        self.self_update_scale = self_update_scale
        self.rng = np.random.default_rng(seed)
        if init_centroids == "onehot":
            self.centroids = np.zeros((self.K, self.d))
            for k in range(self.K):
                self.centroids[k, k % self.d] = 1.0
        elif init_centroids == "dirichlet":
            self.centroids = self.rng.dirichlet(np.ones(self.d), size=self.K)
        else:
            raise ValueError("Unknown init_centroids: {}".format(init_centroids))
        if coeff_scales is None:
            coeff_scales = np.ones(self.K)
        self.coeff_scales = np.asarray(coeff_scales)
        if self.coeff_scales.shape[0] != self.K:
            raise ValueError("coeff_scales must have length num_experts")
        shared_histogram = None
        if self.share_histogram:
            shared_histogram = np.ones((self.S, self.A, self.S))
        self.experts = [UCBVIExpert(momdp, histogram=shared_histogram) for _ in range(self.K)]

    def _score_trajectory(self, Q, trajectory):
        score = 0.0
        for h, (x, a, _) in enumerate(trajectory):
            score += Q[h, x, a]
        return score

    def online_game(self, rewards, prefs, coeff=1.0, task_ids=None):
        Rs = []
        Pis = []
        for idx, (reward, pref) in enumerate(zip(rewards, prefs)):
            task_id = None if task_ids is None else task_ids[idx]
            Qs = []
            policies = []
            trajectories = []
            values = []
            for k, expert in enumerate(self.experts):
                Q, policy = expert.get_Q_table(reward, coeff * self.coeff_scales[k])
                if task_id is None:
                    _, trajectory = self.momdp.play(reward, policy)
                else:
                    _, trajectory = self.momdp.play(reward, policy, task_id=task_id)
                Qs.append(Q)
                policies.append(policy)
                trajectories.append(trajectory)
                values.append(expert.eval_policy_value(reward, policy))

            if self.update_self_only:
                resp = self._get_routing_resp(pref, Qs, trajectories)
                weight_scale = self.K if self.self_update_scale is None else self.self_update_scale
                for k in range(self.K):
                    self.experts[k].update_history(trajectories[k], weight=resp[k] * weight_scale)
                avg_resp = resp
            else:
                total_resp = np.zeros(self.K)
                for trajectory in trajectories:
                    resp = self._get_routing_resp(pref, Qs, trajectories, trajectory)
                    for k in range(self.K):
                        self.experts[k].update_history(trajectory, weight=resp[k])
                    total_resp += resp
                avg_resp = total_resp / max(len(trajectories), 1)
            for k in range(self.K):
                if avg_resp[k] > 0.0:
                    step = self.eta * avg_resp[k]
                    self.centroids[k] = (1.0 - step) * self.centroids[k] + step * pref

            if self.select_by_value:
                k_select = np.argmax(values)
            else:
                k_select = np.argmin(np.sum((self.centroids - pref) ** 2, axis=1))
            Pis.append(policies[k_select])
            Rs.append(reward)
        return Rs, Pis

    def _get_routing_resp(self, pref, Qs, trajectories, trajectory=None):
        if self.routing_mode == "score":
            if trajectory is None:
                scores = np.array([self._score_trajectory(Qs[k], trajectories[k]) for k in range(self.K)])
            else:
                scores = np.array([self._score_trajectory(Qs[k], trajectory) for k in range(self.K)])
            if self.lambda_mu > 0.0:
                for k in range(self.K):
                    diff = pref - self.centroids[k]
                    scores[k] -= self.lambda_mu * np.dot(diff, diff)
            resp = softmax(scores, temperature=self.route_temperature)
        elif self.routing_mode == "pref":
            scores = np.zeros(self.K)
            for k in range(self.K):
                diff = pref - self.centroids[k]
                scores[k] = -max(self.lambda_mu, 1e-6) * np.dot(diff, diff)
            resp = softmax(scores, temperature=self.route_temperature)
        elif self.routing_mode == "uniform":
            resp = np.ones(self.K) / self.K
        elif self.routing_mode == "random":
            alpha = np.ones(self.K) * max(self.random_alpha, 1e-6)
            resp = self.rng.dirichlet(alpha)
        else:
            raise ValueError("Unknown routing_mode: {}".format(self.routing_mode))
        if self.min_resp > 0.0:
            min_resp = min(self.min_resp, 1.0 / self.K - 1e-6)
            resp = (1.0 - min_resp * self.K) * resp + min_resp
            resp /= np.sum(resp)
        return resp


if __name__ == '__main__':
    S = 20
    A = 5
    H = 10
    d = 15
    K = 5000
    num_tasks = 5
    num_experts = 2
    conflict = 0.6
    seed = 7
    strict_single_expert = True
    use_warmup_preference_ensemble = False

    momdp = ConflictMOMDP(
        S,
        A,
        H,
        d,
        conflict=conflict,
        noise_scale=0.05,
        seed=seed,
    )
    rewards, prefs = build_reward_sequence(
        momdp,
        K,
        pref_mode="clustered",
        pref_clusters=num_tasks,
        pref_concentration=0.02,
        seed=seed + 1,
    )
    task_ids = None
    transits = [momdp.transit for _ in range(K)]

    algo_seed = seed + 123
    set_seed(algo_seed)
    policies_ucbvi = run_ucbvi_on_rewards(momdp, rewards, coeff=0.1, task_ids=task_ids)
    if strict_single_expert and num_experts == 1:
        coeff_scales = np.ones(num_experts)
    else:
        coeff_scales = np.linspace(0.6, 1.4, num_experts)

    ensemble = RoutedEnsembleUCBVI(
        momdp,
        num_experts,
        eta=0.3,
        lambda_mu=1.0,
        route_temperature=0.6,
        min_resp=0.02,
        share_histogram=True,
        coeff_scales=coeff_scales,
        select_by_value=True,
        routing_mode="score",
        init_centroids="dirichlet",
        update_self_only=False,
        seed=seed + 2,
    )
    if strict_single_expert and num_experts == 1:
        set_seed(algo_seed)
    rewards_ens, policies_ens = ensemble.online_game(rewards, prefs, coeff=0.1, task_ids=task_ids)

    uniform_ensemble = RoutedEnsembleUCBVI(
        momdp,
        num_experts,
        eta=0.3,
        lambda_mu=1.0,
        route_temperature=0.6,
        min_resp=0.02,
        share_histogram=True,
        select_by_value=False,
        routing_mode="uniform",
        init_centroids="dirichlet",
        update_self_only=False,
        seed=seed + 3,
    )
    rewards_uniform, policies_uniform = uniform_ensemble.online_game(rewards, prefs, coeff=0.1, task_ids=task_ids)
    regrets_uniform = regret_with_transits(rewards_uniform, policies_uniform, transits)

    random_ensemble = RoutedEnsembleUCBVI(
        momdp,
        num_experts,
        eta=0.3,
        lambda_mu=1.0,
        route_temperature=0.6,
        min_resp=0.02,
        share_histogram=True,
        select_by_value=False,
        routing_mode="random",
        init_centroids="dirichlet",
        update_self_only=False,
        seed=seed + 4,
    )
    rewards_random, policies_random = random_ensemble.online_game(rewards, prefs, coeff=0.1, task_ids=task_ids)
    regrets_random = regret_with_transits(rewards_random, policies_random, transits)

    regrets_ucbvi = regret_with_transits(rewards, policies_ucbvi, transits)
    regrets_ens = regret_with_transits(rewards_ens, policies_ens, transits)
    regrets_uniform = regret_with_transits(rewards_uniform, policies_uniform, transits)
    regrets_random = regret_with_transits(rewards_random, policies_random, transits)
    avg_reward = np.mean(np.stack(rewards, axis=0), axis=0)
    avg_transit = np.mean(np.stack(transits, axis=0), axis=0)
    _, pi_stationary = get_V_opt(avg_reward, avg_transit)
    regrets_stay = regret_fixed_policy(rewards, pi_stationary, transits)

    print(regrets_ucbvi[-10:])
    print(regrets_ens[-10:])
    print(regrets_uniform[-10:])
    print(regrets_random[-10:])
    print(regrets_stay[-10:])

    # save regrets to json
    import json
    with open("regrets.json", "w") as f:
        json.dump(regrets_ucbvi, f)
        json.dump(regrets_ens, f)
        json.dump(regrets_uniform, f)
        json.dump(regrets_random, f)
        json.dump(regrets_stay, f)

    plt.plot(range(K + 1), regrets_ucbvi, "-r")
    plt.plot(range(K + 1), regrets_ens, "--g")
    plt.plot(range(K + 1), regrets_uniform, "--y")
    plt.plot(range(K + 1), regrets_random, "--m")
    plt.plot(range(K + 1), regrets_stay, "--b")

    plt.xlabel("number of episodes", fontsize=15)
    plt.ylabel("total regret", fontsize=15)
    ensemble_label = "Warmup Preference Ensemble" if use_warmup_preference_ensemble else "Routed Ensemble"
    plt.legend(
        [
            "MO-UCBVI",
            ensemble_label,
            "Best stationary policy",
        ],
        fontsize=11,
    )

    plt.savefig("ensemble_mdp.pdf")
