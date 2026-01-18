import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class ConditionedPolicy:
    """
    pi_theta(a=0|w) = sigmoid(theta0 + theta1*(w-0.5)), a in {0,1}
    """
    def __init__(self, theta=None):
        self.theta = np.zeros(2) if theta is None else np.array(theta, dtype=float)

    def p0(self, w):
        x = np.array([1.0, w - 0.5])
        return sigmoid(self.theta @ x)

    def prob(self, a, w):
        p0 = self.p0(w)
        return p0 if a == 0 else (1.0 - p0)

    def grad_log_prob(self, a, w):
        # grad wrt theta of log pi(a|w)
        x = np.array([1.0, w - 0.5])
        p0 = self.p0(w)
        if a == 0:
            return (1.0 - p0) * x
        else:
            return (-p0) * x

def oracle_action(w):
    return 0 if w > 0.5 else 1

def sample_reward_vec(a, sigma_r, rng):
    # r in R^2
    if a == 0:
        mean = np.array([1.0, 0.0])
    else:
        mean = np.array([0.0, 1.0])
    return mean + rng.normal(scale=sigma_r, size=2)

class EnsembleBehavior:
    """
    Two experts: expert0 always a=0, expert1 always a=1.
    Routing uses oracle threshold w>0.5, but flips with prob p_err.
    Then apply epsilon exploration floor: with prob eps choose uniform action.
    """
    def __init__(self, p_err=0.2, eps=0.02, rng=None):
        self.p_err = float(p_err)
        self.eps = float(eps)
        self.rng = np.random.default_rng() if rng is None else rng

    def mu_probs(self, w):
        # returns mu(a=0|w), mu(a=1|w)
        a_star = oracle_action(w)
        # intended expert action is a_star, but routing flips with p_err
        p_choose_star = 1.0 - self.p_err
        p_choose_wrong = self.p_err
        # before eps exploration: prob(action=a_star)=p_choose_star, else p_choose_wrong
        p0_noeps = p_choose_star if a_star == 0 else p_choose_wrong
        p1_noeps = 1.0 - p0_noeps
        # eps mixing with uniform
        p0 = (1.0 - self.eps) * p0_noeps + self.eps * 0.5
        p1 = (1.0 - self.eps) * p1_noeps + self.eps * 0.5
        return p0, p1

    def sample_action(self, w):
        p0, p1 = self.mu_probs(w)
        a = 0 if self.rng.random() < p0 else 1
        return a, (p0, p1)

def evaluate_policy(policy, sigma_r, n_eval=20000, seed=0):
    rng = np.random.default_rng(seed)
    regrets = []
    for _ in range(n_eval):
        w = rng.random()
        # sample action from target policy (on-policy eval)
        p0 = policy.p0(w)
        a = 0 if rng.random() < p0 else 1
        r = sample_reward_vec(a, sigma_r, rng)
        y = w * r[0] + (1.0 - w) * r[1]
        # oracle value (in expectation, ignoring noise) is max(w, 1-w)
        q_star = max(w, 1.0 - w)
        regrets.append(q_star - y)
    return float(np.mean(regrets))

def run(seed=0,
        p_err=0.35,
        eps=0.02,
        sigma_r=0.1,
        iters=400,
        batch_size=512,
        lr=0.2,
        n_bins=20,
        use_gamma_weight=True):
    rng = np.random.default_rng(seed)
    behavior = EnsembleBehavior(p_err=p_err, eps=eps, rng=rng)
    policy = ConditionedPolicy(theta=[0.0, 0.0])

    # bin-based observable conflict estimate for ORACLE policy vs behavior
    # We maintain per-bin running mean of W_star^2 where W_star = 1{a=a*}/mu(a*|w).
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_sum_w2 = np.zeros(n_bins, dtype=float)

    def bin_id(w):
        b = int(np.floor(w * n_bins))
        return min(max(b, 0), n_bins - 1)

    regrets = []
    gammas = []
    grad_norms = []

    for t in range(iters):
        # compute predictable weights h(w) from past gamma estimates (per bin)
        # gamma_hat_bin = mean(W_star^2)-1; g_hat_bin = 1/(1+gamma_hat_bin)
        g_hat_bin = np.ones(n_bins, dtype=float)
        for b in range(n_bins):
            if bin_counts[b] > 50:  # wait for enough samples
                m = bin_sum_w2[b] / bin_counts[b]
                gamma_hat = max(m - 1.0, 0.0)
                g_hat_bin[b] = 1.0 / (1.0 + gamma_hat)
            else:
                g_hat_bin[b] = 1.0

        # collect a batch from behavior
        ws = rng.random(batch_size)
        actions = np.zeros(batch_size, dtype=int)
        mu_ps = np.zeros((batch_size, 2), dtype=float)
        ys = np.zeros(batch_size, dtype=float)

        for i in range(batch_size):
            w = float(ws[i])
            a, (p0, p1) = behavior.sample_action(w)
            actions[i] = a
            mu_ps[i, 0] = p0
            mu_ps[i, 1] = p1
            r = sample_reward_vec(a, sigma_r, rng)
            ys[i] = w * r[0] + (1.0 - w) * r[1]

            # update observable conflict stats for oracle policy (uses logged mu + known a*(w))
            b = bin_id(w)
            a_star = oracle_action(w)
            mu_astar = mu_ps[i, a_star]
            w_star = (1.0 / mu_astar) if (a == a_star) else 0.0
            bin_counts[b] += 1
            bin_sum_w2[b] += w_star * w_star

        # compute per-sample h_i (predictable via bin lookup; depends on w only)
        if use_gamma_weight:
            hs = np.array([g_hat_bin[bin_id(float(w))] for w in ws], dtype=float)
        else:
            hs = np.ones(batch_size, dtype=float)

        # IPS policy gradient on weighted empirical objective
        # grad = (1/sum h) sum_i h_i * y_i * (pi_theta(a_i|w_i)/mu(a_i|w_i)) * grad log pi_theta(a_i|w_i)
        grad = np.zeros_like(policy.theta)
        denom = float(np.sum(hs)) + 1e-12
        w2_batch = []

        for i in range(batch_size):
            w = float(ws[i])
            a = int(actions[i])
            mu_a = float(mu_ps[i, a])
            pi_a = float(policy.prob(a, w))
            W = pi_a / (mu_a + 1e-12)
            grad += hs[i] * ys[i] * W * policy.grad_log_prob(a, w)
            w2_batch.append(W * W)

        grad /= denom
        policy.theta += lr * grad

        # logging: observable gamma proxy for current target policy on this batch (not oracle)
        gamma_batch = float(np.mean(w2_batch) - 1.0)
        gammas.append(max(gamma_batch, 0.0))
        grad_norms.append(float(np.linalg.norm(grad)))

        # evaluation regret (on-policy under current policy)
        regrets.append(evaluate_policy(policy, sigma_r, n_eval=6000, seed=seed + 999 + t))

    return {
        "regret": np.array(regrets),
        "gamma_proxy": np.array(gammas),
        "grad_norm": np.array(grad_norms),
    }

if __name__ == "__main__":
    # Sweep routing error to show conflict -> sample complexity
    seeds = [0, 1, 2]
    p_err_list = [0.0, 0.2, 0.4]

    for p_err in p_err_list:
        out_u = [run(seed=s, p_err=p_err, use_gamma_weight=False) for s in seeds]
        out_w = [run(seed=s, p_err=p_err, use_gamma_weight=True) for s in seeds]

        def mean_curve(key, outs):
            return np.mean([o[key] for o in outs], axis=0)

        reg_u = mean_curve("regret", out_u)
        reg_w = mean_curve("regret", out_w)
        gam_u = mean_curve("gamma_proxy", out_u)
        gam_w = mean_curve("gamma_proxy", out_w)

        plt.figure()
        plt.plot(reg_u, label="Uniform h=1")
        plt.plot(reg_w, label="Gamma-weighted h=1/(1+Gamma_hat)")
        plt.xlabel("Iteration")
        plt.ylabel("Eval regret (lower is better)")
        plt.title(f"Routing error p_err={p_err}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(gam_u, label="Uniform: gamma proxy")
        plt.plot(gam_w, label="Weighted: gamma proxy")
        plt.xlabel("Iteration")
        plt.ylabel("Batch gamma proxy = mean(W^2)-1")
        plt.title(f"Gamma proxy (target-vs-behavior) p_err={p_err}")
        plt.legend()
        plt.tight_layout()
        plt.show()
