import numpy as np
import matplotlib.pyplot as plt
from mdp import MOMDP
from mdp import MOUCBVI


class AltMinMOMDP(object):
    def __init__(
        self,
        momdp,
        num_experts,
        pref_set=None,
        episodes_per_iter=5,
        coeff=0.1,
        eta=0.3,
        agg_temperature=0.5,
        seed=None,
    ):
        super(AltMinMOMDP, self).__init__()
        self.momdp = momdp
        self.S = momdp.S
        self.A = momdp.A
        self.H = momdp.H
        self.d = momdp.d
        self.K = num_experts
        self.episodes_per_iter = episodes_per_iter
        self.coeff = coeff
        self.eta = eta
        self.agg_temperature = agg_temperature
        self.rng = np.random.default_rng(seed)

        if pref_set is None:
            pref_set = self.rng.dirichlet(np.ones(self.d), size=self.K)
        self.pref_set = np.asarray(pref_set, dtype=float)
        if self.pref_set.ndim != 2 or self.pref_set.shape[1] != self.d:
            raise ValueError("pref_set must have shape (num_prefs, d)")
        self.pref_set = self._project_simplex(self.pref_set)
        self.pref_experts = self._sample_expert_prefs()
        self.agg_weights = np.ones(self.K) / self.K
        self.experts = [MOUCBVI(momdp) for _ in range(self.K)]

    @staticmethod
    def _project_simplex(v):
        v = np.asarray(v, dtype=float)
        if v.ndim == 1:
            rows = v.reshape((1, -1))
        else:
            rows = v
        projected = np.zeros_like(rows)
        for idx, row in enumerate(rows):
            if np.sum(row) == 1.0 and np.all(row >= 0.0):
                projected[idx] = row
                continue
            u = np.sort(row)[::-1]
            cssv = np.cumsum(u)
            rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1.0))[0]
            if rho.size == 0:
                projected[idx] = np.ones_like(row) / max(len(row), 1)
                continue
            rho = rho[-1]
            theta = (cssv[rho] - 1.0) / float(rho + 1)
            projected[idx] = np.maximum(row - theta, 0.0)
        if v.ndim == 1:
            return projected[0]
        return projected

    @staticmethod
    def _reweight_rewards(rewards, pref):
        pref = np.asarray(pref, dtype=float)
        pref = pref / np.sum(pref)
        return np.sum(rewards * pref.reshape((-1, 1, 1, 1)), axis=0)

    def _sample_expert_prefs(self):
        num_prefs = self.pref_set.shape[0]
        replace = num_prefs < self.K
        indices = self.rng.choice(num_prefs, size=self.K, replace=replace)
        return self.pref_set[indices]

    @staticmethod
    def _mix_policies(policies, weights):
        weights = np.asarray(weights, dtype=float)
        weight_sum = np.sum(weights)
        if weight_sum <= 0.0:
            weights = np.ones(len(policies)) / max(len(policies), 1)
        else:
            weights = weights / weight_sum
        pi_ens = np.zeros_like(policies[0])
        for k, policy in enumerate(policies):
            pi_ens += weights[k] * policy
        return pi_ens

    def _trajectory_scalar_return(self, trajectory, reward):
        value = 0.0
        for h, (x, a, _) in enumerate(trajectory):
            value += reward[h, x, a]
        return value

    def _update_ensemble_policies(self):
        policies = []
        q_tables = []
        trajectories = []
        for k in range(self.K):
            reward_k = self._reweight_rewards(self.momdp.rewards, self.pref_experts[k])
            Q = None
            policy = None
            for _ in range(self.episodes_per_iter):
                Q, policy = self.experts[k].get_Q_table(reward_k, self.coeff)
                _, trajectory = self.momdp.play(reward_k, policy)
                self.experts[k].update_history(trajectory)
                trajectories.append(trajectory)
            policies.append(policy)
            q_tables.append(Q)
        return policies, q_tables, trajectories

    def _update_aggregation(self, trajectories):
        if not trajectories:
            self.agg_weights = np.ones(self.K) / self.K
            return
        rewards = [
            self._reweight_rewards(self.momdp.rewards, self.pref_experts[k])
            for k in range(self.K)
        ]
        counts = np.zeros(self.K, dtype=float)
        for trajectory in trajectories:
            returns = [
                self._trajectory_scalar_return(trajectory, rewards[k])
                for k in range(self.K)
            ]
            best_k = int(np.argmax(returns))
            counts[best_k] += 1.0
        self.agg_weights = self._project_simplex(counts)

    def step(self):
        self.pref_experts = self._sample_expert_prefs()
        policies, q_tables, trajectories = self._update_ensemble_policies()
        self._update_aggregation(trajectories)
        agg_policy = self._mix_policies(policies, self.agg_weights)
        return agg_policy, policies, q_tables

    def evaluate_regret(self, policy, pref=None):
        if pref is None:
            pref = np.mean(self.pref_set, axis=0)
        reward_true = self._reweight_rewards(self.momdp.rewards, pref)
        v_opt, _ = self.momdp._get_V_opt(reward_true)
        v_pi = self.momdp._get_V_pi(reward_true, policy)
        return v_opt - v_pi

if __name__ == "__main__":
    S = 20
    A = 5
    H = 10
    d = 15
    iters = 200
    num_experts = 3

    momdp = MOMDP(S, A, H, d)
    pref_set = np.random.default_rng(7).dirichlet(np.ones(d), size=50)
    learner = AltMinMOMDP(
        momdp,
        num_experts=num_experts,
        pref_set=pref_set,
        episodes_per_iter=3,
        coeff=0.1,
        seed=7,
    )

    regrets = []
    pref_errors = []
    for _ in range(iters):
        policy, _, _ = learner.step()
        regrets.append(learner.evaluate_regret(policy))
        pref_errors.append(0.0)

    print("pref set size:", learner.pref_set.shape[0])
    print("avg regret:", float(np.mean(regrets[-10:])))

    plt.plot(range(iters), regrets, "-r")
    plt.plot(range(iters), pref_errors, "--b")
    plt.xlabel("iteration", fontsize=13)
    plt.legend(["regret", "pref error"], fontsize=11)
    plt.savefig("plot/altmin_momdp.pdf")
