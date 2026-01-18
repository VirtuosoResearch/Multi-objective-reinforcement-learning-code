import numpy as np
from mdp import MOMDP
from mdp import MOUCBVI

def scalarize_rewards(rewards, pref):
    pref = np.asarray(pref, dtype=float)
    pref = pref / np.sum(pref)
    return np.sum(rewards * pref.reshape((-1, 1, 1, 1)), axis=0)


def project_simplex(v):
    v = np.asarray(v, dtype=float)
    if np.sum(v) == 1.0 and np.all(v >= 0.0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1.0))[0]
    if rho.size == 0:
        return np.ones_like(v) / max(len(v), 1)
    rho = rho[-1]
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    return w


class AltMinPreferenceLearner(object):
    def __init__(
        self,
        momdp,
        pref_true=None,
        pref_init=None,
        episodes_per_iter=5,
        coeff=0.1,
        ridge=1e-4,
        seed=None,
    ):
        super(AltMinPreferenceLearner, self).__init__()
        self.momdp = momdp
        self.S = momdp.S
        self.A = momdp.A
        self.H = momdp.H
        self.d = momdp.d
        self.episodes_per_iter = episodes_per_iter
        self.coeff = coeff
        self.ridge = ridge
        self.rng = np.random.default_rng(seed)
        if pref_true is None:
            pref_true = self.rng.random(self.d)
            pref_true /= np.sum(pref_true)
        if pref_init is None:
            pref_init = np.ones(self.d) / self.d
        self.pref_true = np.asarray(pref_true, dtype=float)
        self.pref_est = project_simplex(np.asarray(pref_init, dtype=float))
        self.moucbvi = MOUCBVI(momdp)
        self.feature_returns = []
        self.scalar_returns = []

    def _trajectory_returns(self, trajectory):
        vec_return = np.zeros(self.d)
        scalar_return = 0.0
        reward_true = scalarize_rewards(self.momdp.rewards, self.pref_true)
        for h, (x, a, _) in enumerate(trajectory):
            vec_return += self.momdp.rewards[:, h, x, a]
            scalar_return += reward_true[h, x, a]
        return vec_return, scalar_return

    def _update_pref(self):
        X = np.stack(self.feature_returns, axis=0)
        y = np.asarray(self.scalar_returns, dtype=float)
        gram = X.T @ X + self.ridge * np.eye(self.d)
        rhs = X.T @ y
        w = np.linalg.solve(gram, rhs)
        self.pref_est = project_simplex(w)

    def step(self):
        reward_est = scalarize_rewards(self.momdp.rewards, self.pref_est)
        policy = None
        for _ in range(self.episodes_per_iter):
            _, policy = self.moucbvi.get_Q_table(reward_est, self.coeff)
            _, trajectory = self.momdp.play(reward_est, policy)
            self.moucbvi.update_history(trajectory)
            vec_return, scalar_return = self._trajectory_returns(trajectory)
            self.feature_returns.append(vec_return)
            self.scalar_returns.append(scalar_return)
        self._update_pref()
        return policy

    def evaluate_regret(self, policy):
        reward_true = scalarize_rewards(self.momdp.rewards, self.pref_true)
        v_opt, _ = self.momdp._get_V_opt(reward_true)
        v_pi = self.momdp._get_V_pi(reward_true, policy)
        return v_opt - v_pi


if __name__ == "__main__":
    S = 20
    A = 5
    H = 10
    d = 15
    iters = 200

    momdp = MOMDP(S, A, H, d)
    learner = AltMinPreferenceLearner(momdp, episodes_per_iter=3, coeff=0.1, seed=7)

    regrets = []
    pref_errors = []
    for _ in range(iters):
        policy = learner.step()
        regrets.append(learner.evaluate_regret(policy))
        pref_errors.append(np.linalg.norm(learner.pref_est - learner.pref_true))

    print("final pref est:", learner.pref_est)
    print("final pref true:", learner.pref_true)
    print("avg regret:", float(np.mean(regrets[-10:])))

    import matplotlib.pyplot as plt

    plt.plot(range(iters), regrets, "-r")
    plt.plot(range(iters), pref_errors, "--b")
    plt.xlabel("iteration", fontsize=13)
    plt.legend(["regret", "pref error"], fontsize=11)
    plt.savefig("altmin_mdp.pdf")
