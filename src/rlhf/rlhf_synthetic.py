import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


ACTION_NAMES = ("comply", "cautious", "refuse")
METHOD_SINGLE = "single_policy"
METHOD_UNIFORM = "uniform_ensemble"
METHOD_ROUTED = "routed_ensemble"


def build_features(contexts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights[:, None]
    return np.concatenate(
        [
            np.ones((len(contexts), 1)),
            contexts,
            w,
            1.0 - w,
            contexts[:, [0]] * w,
            contexts[:, [1]] * (1.0 - w),
        ],
        axis=1,
    )


def scalarize(weights: np.ndarray, helpfulness: np.ndarray, safety: np.ndarray) -> np.ndarray:
    return weights * helpfulness + (1.0 - weights) * safety


def hypervolume_2d(points: np.ndarray, reference=(0.0, 0.0)) -> float:
    if len(points) == 0:
        return 0.0

    ref_x, ref_y = reference
    filtered = points[(points[:, 0] > ref_x) & (points[:, 1] > ref_y)]
    if len(filtered) == 0:
        return 0.0

    order = np.argsort(filtered[:, 0])
    filtered = filtered[order]

    frontier = []
    best_y = -np.inf
    for x, y in filtered[::-1]:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    frontier = np.array(frontier[::-1], dtype=float)

    hv = 0.0
    prev_x = ref_x
    for x, y in frontier:
        hv += max(x - prev_x, 0.0) * max(y - ref_y, 0.0)
        prev_x = x
    return float(hv)


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(values[mask].mean())


@dataclass
class EvalBundle:
    utility: np.ndarray
    helpfulness: np.ndarray
    safety: np.ndarray
    oracle_utility: np.ndarray
    risk: np.ndarray


class SyntheticRLHFEnv:
    """
    A lightweight RLHF proxy:
    - contexts are prompt latents
    - actions are response styles
    - rewards are helpfulness + safety

    The hidden safety boundary is XOR-like in (x1, x2), so a single linear
    policy underfits while routed experts can specialize on different regions.
    """

    def __init__(self, reward_noise: float = 0.03):
        self.reward_noise = float(reward_noise)
        self.num_actions = len(ACTION_NAMES)
        self.centers = np.array(
            [
                [1.15, 1.10],
                [1.15, -1.10],
                [-1.15, 1.10],
                [-1.15, -1.10],
            ],
            dtype=float,
        )

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        cluster_ids = rng.integers(0, len(self.centers), size=batch_size)
        contexts = self.centers[cluster_ids] + rng.normal(0.0, 0.35, size=(batch_size, 2))
        weights = rng.uniform(0.0, 1.0, size=batch_size)
        risk = (contexts[:, 0] * contexts[:, 1] < 0.0).astype(np.int64)
        return contexts, weights, risk

    def reward_components(self, contexts: np.ndarray, actions: np.ndarray, rng=None, noisy: bool = True):
        risk = contexts[:, 0] * contexts[:, 1] < 0.0
        complexity = contexts[:, 0] > 0.0

        helpfulness = np.zeros(len(actions), dtype=float)
        safety = np.zeros(len(actions), dtype=float)

        comply = actions == 0
        cautious = actions == 1
        refuse = actions == 2

        benign = ~risk
        unsafe = risk

        helpfulness[benign & comply] = 0.86 + 0.08 * complexity[benign & comply]
        safety[benign & comply] = 0.56

        helpfulness[benign & cautious] = 0.70 + 0.04 * complexity[benign & cautious]
        safety[benign & cautious] = 0.82

        helpfulness[benign & refuse] = 0.12
        safety[benign & refuse] = 0.97

        helpfulness[unsafe & comply] = 0.88 - 0.05 * complexity[unsafe & comply]
        safety[unsafe & comply] = 0.03

        helpfulness[unsafe & cautious] = 0.55
        safety[unsafe & cautious] = 0.72

        helpfulness[unsafe & refuse] = 0.18
        safety[unsafe & refuse] = 0.98

        if noisy and rng is not None and self.reward_noise > 0.0:
            helpfulness = helpfulness + rng.normal(0.0, self.reward_noise, size=len(actions))
            safety = safety + rng.normal(0.0, self.reward_noise, size=len(actions))

        helpfulness = np.clip(helpfulness, 0.0, 1.0)
        safety = np.clip(safety, 0.0, 1.0)
        return helpfulness, safety

    def oracle(self, contexts: np.ndarray, weights: np.ndarray):
        action_grid = np.tile(np.arange(self.num_actions), (len(contexts), 1))
        flat_actions = action_grid.reshape(-1)
        repeated_contexts = np.repeat(contexts, self.num_actions, axis=0)
        helpfulness, safety = self.reward_components(repeated_contexts, flat_actions, noisy=False)

        helpfulness = helpfulness.reshape(len(contexts), self.num_actions)
        safety = safety.reshape(len(contexts), self.num_actions)
        utilities = scalarize(weights[:, None], helpfulness, safety)

        best_idx = np.argmax(utilities, axis=1)
        row_idx = np.arange(len(contexts))
        return (
            utilities[row_idx, best_idx],
            helpfulness[row_idx, best_idx],
            safety[row_idx, best_idx],
            best_idx,
        )


class LinearSoftmaxPolicy:
    def __init__(self, num_features: int, num_actions: int, rng: np.random.Generator, init_scale: float = 0.08):
        self.weights = rng.normal(0.0, init_scale, size=(num_actions, num_features))
        self.baseline = 0.0

    def action_probs(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights.T
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def sample(self, features: np.ndarray, rng: np.random.Generator):
        probs = self.action_probs(features)
        cdf = np.cumsum(probs, axis=1)
        draws = rng.random((len(features), 1))
        actions = (draws > cdf).sum(axis=1)
        return actions, probs

    def greedy(self, features: np.ndarray):
        probs = self.action_probs(features)
        return np.argmax(probs, axis=1), probs

    def update(self, features: np.ndarray, actions: np.ndarray, rewards: np.ndarray, probs: np.ndarray, learning_rate: float):
        if len(features) == 0:
            return

        self.baseline = 0.9 * self.baseline + 0.1 * float(rewards.mean())
        advantages = rewards - self.baseline

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(actions)), actions] = 1.0
        grad_logits = (one_hot - probs) * advantages[:, None]
        grad = grad_logits.T @ features / len(features)
        self.weights += learning_rate * grad


def train_method(method: str, env: SyntheticRLHFEnv, seed: int, num_experts: int, batch_size: int, train_steps: int, learning_rate: float):
    rng = np.random.default_rng(seed)
    num_features = build_features(np.zeros((1, 2)), np.zeros(1)).shape[1]
    num_policies = 1 if method == METHOD_SINGLE else num_experts
    policies = [LinearSoftmaxPolicy(num_features, env.num_actions, rng=rng) for _ in range(num_policies)]

    cumulative_regret = 0.0
    regret_curve = []

    for step in range(train_steps):
        contexts, weights, _ = env.sample_batch(batch_size, rng)
        features = build_features(contexts, weights)
        oracle_utility, _, _, _ = env.oracle(contexts, weights)

        if method == METHOD_SINGLE:
            actions, probs = policies[0].sample(features, rng)
            helpfulness, safety = env.reward_components(contexts, actions, rng=rng, noisy=True)
            achieved_utility = scalarize(weights, helpfulness, safety)
            policies[0].update(features, actions, achieved_utility, probs, learning_rate)
        else:
            all_actions = []
            all_probs = []
            all_utilities = []

            for policy in policies:
                actions, probs = policy.sample(features, rng)
                helpfulness, safety = env.reward_components(contexts, actions, rng=rng, noisy=True)
                utility = scalarize(weights, helpfulness, safety)
                all_actions.append(actions)
                all_probs.append(probs)
                all_utilities.append(utility)

            utility_matrix = np.stack(all_utilities, axis=1)
            if method == METHOD_UNIFORM:
                routed_idx = rng.integers(0, len(policies), size=batch_size)
            elif method == METHOD_ROUTED:
                routed_idx = np.argmax(utility_matrix, axis=1)
            else:
                raise ValueError("Unsupported method: %s" % method)

            achieved_utility = utility_matrix[np.arange(batch_size), routed_idx]

            for expert_idx, policy in enumerate(policies):
                mask = routed_idx == expert_idx
                policy.update(
                    features[mask],
                    all_actions[expert_idx][mask],
                    all_utilities[expert_idx][mask],
                    all_probs[expert_idx][mask],
                    learning_rate,
                )

        cumulative_regret += float(np.sum(oracle_utility - achieved_utility))
        regret_curve.append(cumulative_regret / ((step + 1) * batch_size))

    return policies, np.array(regret_curve, dtype=float)


def rollout_method(method: str, policies: List[LinearSoftmaxPolicy], env: SyntheticRLHFEnv, contexts: np.ndarray, weights: np.ndarray):
    features = build_features(contexts, weights)
    oracle_utility, _, _, _ = env.oracle(contexts, weights)

    if method == METHOD_SINGLE:
        actions, _ = policies[0].greedy(features)
        helpfulness, safety = env.reward_components(contexts, actions, noisy=False)
        utility = scalarize(weights, helpfulness, safety)
        return EvalBundle(utility, helpfulness, safety, oracle_utility, (contexts[:, 0] * contexts[:, 1] < 0.0))

    all_helpfulness = []
    all_safety = []
    all_utilities = []
    for policy in policies:
        actions, _ = policy.greedy(features)
        helpfulness, safety = env.reward_components(contexts, actions, noisy=False)
        utility = scalarize(weights, helpfulness, safety)
        all_helpfulness.append(helpfulness)
        all_safety.append(safety)
        all_utilities.append(utility)

    helpfulness_matrix = np.stack(all_helpfulness, axis=1)
    safety_matrix = np.stack(all_safety, axis=1)
    utility_matrix = np.stack(all_utilities, axis=1)

    if method == METHOD_UNIFORM:
        helpfulness = helpfulness_matrix.mean(axis=1)
        safety = safety_matrix.mean(axis=1)
        utility = utility_matrix.mean(axis=1)
    elif method == METHOD_ROUTED:
        routed_idx = np.argmax(utility_matrix, axis=1)
        row_idx = np.arange(len(contexts))
        helpfulness = helpfulness_matrix[row_idx, routed_idx]
        safety = safety_matrix[row_idx, routed_idx]
        utility = utility_matrix[row_idx, routed_idx]
    else:
        raise ValueError("Unsupported method: %s" % method)

    return EvalBundle(utility, helpfulness, safety, oracle_utility, (contexts[:, 0] * contexts[:, 1] < 0.0))


def evaluate_method(method: str, policies: List[LinearSoftmaxPolicy], env: SyntheticRLHFEnv, eval_contexts: np.ndarray, eval_weights: np.ndarray, frontier_weights: np.ndarray):
    mixed_eval = rollout_method(method, policies, env, eval_contexts, eval_weights)
    risk = mixed_eval.risk.astype(bool)
    benign = ~risk
    regret = mixed_eval.oracle_utility - mixed_eval.utility

    frontier_points = []
    frontier_utilities = []
    frontier_regrets = []
    for weight in frontier_weights:
        weights = np.full(len(eval_contexts), weight, dtype=float)
        bundle = rollout_method(method, policies, env, eval_contexts, weights)
        frontier_points.append([bundle.helpfulness.mean(), bundle.safety.mean()])
        frontier_utilities.append(bundle.utility.mean())
        frontier_regrets.append((bundle.oracle_utility - bundle.utility).mean())

    frontier_points = np.array(frontier_points, dtype=float)
    hypervolume = hypervolume_2d(frontier_points, reference=(0.0, 0.0))

    return {
        "utility": float(mixed_eval.utility.mean()),
        "regret": float(regret.mean()),
        "success_rate": float(np.mean(regret < 0.05)),
        "helpfulness": float(mixed_eval.helpfulness.mean()),
        "safety": float(mixed_eval.safety.mean()),
        "unsafe_safety": float(mixed_eval.safety[risk].mean()),
        "benign_helpfulness": float(mixed_eval.helpfulness[benign].mean()),
        "aligned_helpfulness": masked_mean(mixed_eval.helpfulness, benign & (eval_weights >= 0.60)),
        "aligned_safety": masked_mean(mixed_eval.safety, risk & (eval_weights <= 0.40)),
        "hypervolume": hypervolume,
        "frontier_utility": float(np.mean(frontier_utilities)),
        "frontier_regret": float(np.mean(frontier_regrets)),
        "per_example_utility": mixed_eval.utility,
    }


def summarize_metric(values: List[float]) -> str:
    arr = np.array(values, dtype=float)
    return "%.4f +/- %.4f" % (arr.mean(), arr.std(ddof=0))


def print_summary(all_results: Dict[str, List[dict]], pairwise_wins: Dict[str, List[float]]):
    metric_names = [
        "utility",
        "regret",
        "success_rate",
        "hypervolume",
        "frontier_utility",
        "frontier_regret",
        "final_train_regret",
        "aligned_helpfulness",
        "aligned_safety",
        "benign_helpfulness",
        "unsafe_safety",
    ]

    print("\n=== Aggregate Metrics Across Seeds ===")
    for method in [METHOD_SINGLE, METHOD_UNIFORM, METHOD_ROUTED]:
        print("\n[%s]" % method)
        for metric in metric_names:
            print("  %-18s: %s" % (metric, summarize_metric([run[metric] for run in all_results[method]])))

    print("\n=== Routed Ensemble Pairwise Win Rate ===")
    for baseline, wins in pairwise_wins.items():
        print("  routed vs %-16s: %.2f%%" % (baseline, 100.0 * np.mean(wins)))


def run_experiment(args):
    env = SyntheticRLHFEnv(reward_noise=args.reward_noise)
    frontier_weights = np.linspace(0.0, 1.0, args.frontier_points)

    all_results = {METHOD_SINGLE: [], METHOD_UNIFORM: [], METHOD_ROUTED: []}
    pairwise_wins = {METHOD_SINGLE: [], METHOD_UNIFORM: []}

    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        eval_rng = np.random.default_rng(10_000 + seed)
        eval_contexts, eval_weights, _ = env.sample_batch(args.eval_size, eval_rng)

        run_metrics = {}
        for method in [METHOD_SINGLE, METHOD_UNIFORM, METHOD_ROUTED]:
            policies, regret_curve = train_method(
                method=method,
                env=env,
                seed=seed,
                num_experts=args.num_experts,
                batch_size=args.batch_size,
                train_steps=args.train_steps,
                learning_rate=args.learning_rate,
            )
            metrics = evaluate_method(
                method=method,
                policies=policies,
                env=env,
                eval_contexts=eval_contexts,
                eval_weights=eval_weights,
                frontier_weights=frontier_weights,
            )
            metrics["final_train_regret"] = float(regret_curve[-1])
            all_results[method].append(metrics)
            run_metrics[method] = metrics

        routed_utility = run_metrics[METHOD_ROUTED]["per_example_utility"]
        for baseline in [METHOD_SINGLE, METHOD_UNIFORM]:
            baseline_utility = run_metrics[baseline]["per_example_utility"]
            pairwise_wins[baseline].append(float(np.mean(routed_utility > baseline_utility)))

        print(
            "Seed %d | utility: single=%.4f, uniform=%.4f, routed=%.4f"
            % (
                seed,
                run_metrics[METHOD_SINGLE]["utility"],
                run_metrics[METHOD_UNIFORM]["utility"],
                run_metrics[METHOD_ROUTED]["utility"],
            )
        )

    print_summary(all_results, pairwise_wins)

    routed_utility = np.mean([run["utility"] for run in all_results[METHOD_ROUTED]])
    single_utility = np.mean([run["utility"] for run in all_results[METHOD_SINGLE]])
    uniform_utility = np.mean([run["utility"] for run in all_results[METHOD_UNIFORM]])
    routed_regret = np.mean([run["regret"] for run in all_results[METHOD_ROUTED]])
    single_regret = np.mean([run["regret"] for run in all_results[METHOD_SINGLE]])
    uniform_regret = np.mean([run["regret"] for run in all_results[METHOD_UNIFORM]])

    passed = routed_utility > single_utility and routed_utility > uniform_utility
    passed = passed and routed_regret < single_regret and routed_regret < uniform_regret
    print("\n=== Verdict ===")
    if passed:
        print("Routed ensemble outperforms both the single policy and uniform routing on utility and regret.")
    else:
        print("Routed ensemble does not dominate both baselines under the current synthetic setup.")


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic RLHF validation for routed ensembles.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--eval-size", type=int, default=2048)
    parser.add_argument("--frontier-points", type=int, default=11)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--reward-noise", type=float, default=0.03)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
