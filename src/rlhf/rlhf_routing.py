import argparse
import itertools
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


METHOD_SINGLE = "single_policy"
METHOD_UNIFORM = "uniform_ensemble"
METHOD_ROUTED = "routed_ensemble"
VALID_OBJECTIVES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")


@dataclass
class ResponseGroup:
    prompt: str
    responses: List[str]
    rewards: torch.Tensor


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hypervolume_2d(points: List[Tuple[float, float]], reference=(0.0, 0.0)) -> float:
    if not points:
        return 0.0

    pts = torch.tensor(points, dtype=torch.float32)
    pts = pts[(pts[:, 0] > reference[0]) & (pts[:, 1] > reference[1])]
    if pts.numel() == 0:
        return 0.0

    order = torch.argsort(pts[:, 0])
    pts = pts[order]

    frontier = []
    best_y = float("-inf")
    for x, y in reversed(pts.tolist()):
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    frontier.reverse()

    hv = 0.0
    prev_x = reference[0]
    for x, y in frontier:
        hv += max(x - prev_x, 0.0) * max(y - reference[1], 0.0)
        prev_x = x
    return float(hv)


def anchor_weights(num_experts: int):
    if num_experts == 1:
        return [0.5]
    if num_experts == 2:
        return [0.0, 1.0]
    return torch.linspace(0.0, 1.0, steps=num_experts).tolist()


def normalize_score(value: float) -> float:
    return max(0.0, min(1.0, float(value) / 4.0))


def load_helpsteer_groups(
    objective_a: str,
    objective_b: str,
    train_groups: int,
    eval_groups: int,
    seed: int,
    offline: bool,
):
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    train_ds = load_dataset("nvidia/HelpSteer2", split="train")
    eval_ds = load_dataset("nvidia/HelpSteer2", split="validation")

    def build(ds):
        grouped = defaultdict(list)
        for row in ds:
            grouped[row["prompt"]].append(row)

        groups = []
        for prompt, rows in grouped.items():
            if len(rows) < 2:
                continue
            responses = []
            rewards = []
            for row in rows:
                responses.append(row["response"].strip())
                rewards.append(
                    [
                        normalize_score(row[objective_a]),
                        normalize_score(row[objective_b]),
                    ]
                )
            groups.append(ResponseGroup(prompt=prompt.strip(), responses=responses, rewards=torch.tensor(rewards, dtype=torch.float32)))

        rng = random.Random(seed)
        rng.shuffle(groups)
        return groups

    train_groups_data = build(train_ds)[:train_groups]
    eval_groups_data = build(eval_ds)[:eval_groups]
    return train_groups_data, eval_groups_data


def build_sequence(tokenizer, prompt: str, response: str, max_prompt_tokens: int, max_response_tokens: int):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids[-max_prompt_tokens:]
    response_ids = tokenizer(response, add_special_tokens=False).input_ids[:max_response_tokens]
    response_ids = response_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    return input_ids, labels


def pad_batch(batch_ids: Sequence[Sequence[int]], pad_value: int):
    max_len = max(len(item) for item in batch_ids)
    padded = []
    mask = []
    for item in batch_ids:
        pad_len = max_len - len(item)
        padded.append(list(item) + [pad_value] * pad_len)
        mask.append([1] * len(item) + [0] * pad_len)
    return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def collate_prompt_response_pairs(tokenizer, prompt_response_pairs: Sequence[Tuple[str, str]], max_prompt_tokens: int, max_response_tokens: int):
    input_id_list = []
    label_list = []
    for prompt, response in prompt_response_pairs:
        input_ids, labels = build_sequence(tokenizer, prompt, response, max_prompt_tokens, max_response_tokens)
        input_id_list.append(input_ids)
        label_list.append(labels)
    input_ids, attention_mask = pad_batch(input_id_list, tokenizer.pad_token_id)
    labels, _ = pad_batch(label_list, -100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def sequence_logps(model, input_ids, attention_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid_mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
    token_logps = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * valid_mask
    lengths = valid_mask.sum(dim=-1).clamp_min(1)
    return token_logps.sum(dim=-1) / lengths


def flatten_candidate_pairs(groups: Sequence[ResponseGroup]):
    flat_pairs = []
    metadata = []
    for group_idx, group in enumerate(groups):
        for response_idx, response in enumerate(group.responses):
            flat_pairs.append((group.prompt, response))
            metadata.append((group_idx, response_idx))
    return flat_pairs, metadata


def propose_responses(model, tokenizer, groups: Sequence[ResponseGroup], adapter_name: str, policy_weights: torch.Tensor, args, device: torch.device):
    model.set_adapter(adapter_name)
    flat_pairs, metadata = flatten_candidate_pairs(groups)
    batch = collate_prompt_response_pairs(tokenizer, flat_pairs, args.max_prompt_tokens, args.max_response_tokens)
    batch = move_batch(batch, device)

    with torch.no_grad():
        logps = sequence_logps(model, batch["input_ids"], batch["attention_mask"], batch["labels"]).detach().cpu()

    grouped_scores = defaultdict(list)
    for score, (group_idx, response_idx) in zip(logps.tolist(), metadata):
        reward_prior = float((groups[group_idx].rewards[response_idx].to(policy_weights.device) * policy_weights).sum().item())
        total_score = score + args.reward_prior_scale * reward_prior
        grouped_scores[group_idx].append((total_score, response_idx))

    selected_indices = []
    selected_rewards = []
    for group_idx, group in enumerate(groups):
        best_response_idx = max(grouped_scores[group_idx], key=lambda item: item[0])[1]
        selected_indices.append(best_response_idx)
        selected_rewards.append(group.rewards[best_response_idx])

    return {
        "selected_indices": selected_indices,
        "selected_rewards": torch.stack(selected_rewards, dim=0),
    }


def weighted_sft_loss(model, tokenizer, groups: Sequence[ResponseGroup], selected_indices: Sequence[int], reward_weights: torch.Tensor, args, device: torch.device):
    prompt_response_pairs = [
        (group.prompt, group.responses[selected_idx])
        for group, selected_idx in zip(groups, selected_indices)
    ]
    batch = collate_prompt_response_pairs(tokenizer, prompt_response_pairs, args.max_prompt_tokens, args.max_response_tokens)
    batch = move_batch(batch, device)
    seq_logp = sequence_logps(model, batch["input_ids"], batch["attention_mask"], batch["labels"])
    return -(reward_weights.detach() * seq_logp).mean()


def oracle_indices_for_weights(groups: Sequence[ResponseGroup], weights: torch.Tensor):
    selected = []
    for group in groups:
        utilities = (group.rewards.to(weights.device) * weights.view(1, 2)).sum(dim=1)
        selected.append(int(torch.argmax(utilities).item()))
    return selected


def iter_batches(groups: List[ResponseGroup], batch_size: int, shuffle: bool, seed: int):
    indices = list(range(len(groups)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield [groups[idx] for idx in batch_indices]


def lora_config(rank: int, alpha: int, dropout: float):
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )


def get_trainable_params(model, adapter_name: str):
    model.set_adapter(adapter_name)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def warmstart_ensemble(model, tokenizer, groups: List[ResponseGroup], adapter_names: List[str], args, device: torch.device):
    anchors = anchor_weights(len(adapter_names))
    optimizers = {
        adapter_name: torch.optim.AdamW(get_trainable_params(model, adapter_name), lr=args.learning_rate)
        for adapter_name in adapter_names
    }
    model.train()

    for adapter_name, anchor in zip(adapter_names, anchors):
        anchor_weights_tensor = torch.tensor([anchor, 1.0 - anchor], dtype=torch.float32, device=device)
        for epoch in range(args.warmstart_epochs):
            for batch_groups in iter_batches(groups, args.batch_size, True, args.seed + 5000 + epoch):
                selected_indices = oracle_indices_for_weights(batch_groups, anchor_weights_tensor)
                scalar_rewards = []
                for group, selected_idx in zip(batch_groups, selected_indices):
                    scalar_rewards.append((group.rewards[selected_idx].to(device) * anchor_weights_tensor).sum())
                scalar_rewards = torch.stack(scalar_rewards)

                model.set_adapter(adapter_name)
                optimizers[adapter_name].zero_grad(set_to_none=True)
                loss = weighted_sft_loss(
                    model,
                    tokenizer,
                    batch_groups,
                    selected_indices,
                    scalar_rewards,
                    args,
                    device,
                )
                loss.backward()
                optimizers[adapter_name].step()


def train_single(model, tokenizer, groups: List[ResponseGroup], args, device: torch.device):
    optimizer = torch.optim.AdamW(get_trainable_params(model, METHOD_SINGLE), lr=args.learning_rate)
    model.train()
    single_policy_weights = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)

    for epoch in range(args.epochs):
        generator = torch.Generator(device=device).manual_seed(args.seed + epoch)
        for step, batch_groups in enumerate(iter_batches(groups, args.batch_size, True, args.seed + epoch), start=1):
            w = torch.rand(len(batch_groups), generator=generator, device=device)
            weights = torch.stack([w, 1.0 - w], dim=1)

            proposal = propose_responses(model, tokenizer, batch_groups, METHOD_SINGLE, single_policy_weights, args, device)
            scalar_rewards = (proposal["selected_rewards"].to(device) * weights).sum(dim=1)

            optimizer.zero_grad(set_to_none=True)
            loss = weighted_sft_loss(
                model,
                tokenizer,
                batch_groups,
                proposal["selected_indices"],
                scalar_rewards,
                args,
                device,
            )
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0:
                print(f"[{METHOD_SINGLE}] epoch={epoch + 1} step={step} loss={loss.item():.4f}")


def train_ensemble(model, tokenizer, groups: List[ResponseGroup], adapter_names: List[str], method: str, args, device: torch.device):
    optimizers = {
        adapter_name: torch.optim.AdamW(get_trainable_params(model, adapter_name), lr=args.learning_rate)
        for adapter_name in adapter_names
    }
    route_counts = {adapter_name: 0 for adapter_name in adapter_names}
    anchors = anchor_weights(len(adapter_names))
    anchor_map = {
        adapter_name: torch.tensor([anchor, 1.0 - anchor], dtype=torch.float32, device=device)
        for adapter_name, anchor in zip(adapter_names, anchors)
    }
    model.train()

    for epoch in range(args.epochs):
        generator = torch.Generator(device=device).manual_seed(args.seed + epoch + 1000)
        for step, batch_groups in enumerate(iter_batches(groups, args.batch_size, True, args.seed + epoch), start=1):
            w = torch.rand(len(batch_groups), generator=generator, device=device)
            weights = torch.stack([w, 1.0 - w], dim=1)

            proposals = []
            for adapter_name in adapter_names:
                proposals.append(propose_responses(model, tokenizer, batch_groups, adapter_name, anchor_map[adapter_name], args, device))

            reward_matrix = torch.stack([proposal["selected_rewards"] for proposal in proposals], dim=1).to(device)
            utility_matrix = (reward_matrix * weights.unsqueeze(1)).sum(dim=2)

            if method == METHOD_UNIFORM:
                route_ids = torch.randint(len(adapter_names), (len(batch_groups),), generator=generator, device=device)
            elif method == METHOD_ROUTED:
                route_ids = torch.argmax(utility_matrix, dim=1)
            else:
                raise ValueError(f"Unsupported method: {method}")

            for expert_idx, adapter_name in enumerate(adapter_names):
                assigned = route_ids.eq(expert_idx)
                if not assigned.any():
                    continue

                route_counts[adapter_name] += int(assigned.sum().item())
                selected_groups = [batch_groups[idx] for idx in assigned.nonzero(as_tuple=False).squeeze(1).tolist()]
                selected_indices = [
                    proposals[expert_idx]["selected_indices"][idx]
                    for idx in assigned.nonzero(as_tuple=False).squeeze(1).tolist()
                ]
                selected_rewards = proposals[expert_idx]["selected_rewards"][assigned.cpu()].to(device)
                selected_weights = weights[assigned]
                scalar_rewards = (selected_rewards * selected_weights).sum(dim=1)

                model.set_adapter(adapter_name)
                optimizers[adapter_name].zero_grad(set_to_none=True)
                loss = weighted_sft_loss(
                    model,
                    tokenizer,
                    selected_groups,
                    selected_indices,
                    scalar_rewards,
                    args,
                    device,
                )
                loss.backward()
                optimizers[adapter_name].step()

            if step % args.log_every == 0:
                counts = ", ".join(f"{name}:{count}" for name, count in route_counts.items())
                print(f"[{method}] epoch={epoch + 1} step={step} routed=({counts})")

    return route_counts


def evaluate_method(method: str, model, tokenizer, groups: List[ResponseGroup], adapter_names: List[str], weight_grid: List[float], args, device: torch.device):
    frontier_points = []
    utilities = []
    regrets = []

    if method == METHOD_SINGLE:
        eval_weights = [torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)]
    else:
        eval_weights = [
            torch.tensor([anchor, 1.0 - anchor], dtype=torch.float32, device=device)
            for anchor in anchor_weights(len(adapter_names))
        ]

    per_adapter_rewards = []
    for adapter_name, policy_weights in zip(adapter_names, eval_weights):
        proposal = propose_responses(model, tokenizer, groups, adapter_name, policy_weights, args, device)
        per_adapter_rewards.append(proposal["selected_rewards"])
    reward_tensor = torch.stack(per_adapter_rewards, dim=1)

    for w_a in weight_grid:
        weights = torch.tensor([w_a, 1.0 - w_a], dtype=torch.float32)
        utility_matrix = (reward_tensor * weights.view(1, 1, 2)).sum(dim=2)

        oracle_rewards = []
        selected_rewards = []
        for group_idx, group in enumerate(groups):
            group_utilities = (group.rewards * weights.view(1, 2)).sum(dim=1)
            oracle_idx = int(torch.argmax(group_utilities).item())
            oracle_rewards.append(group.rewards[oracle_idx])

            if method == METHOD_SINGLE:
                chosen_reward = reward_tensor[group_idx, 0]
            elif method == METHOD_UNIFORM:
                chosen_reward = reward_tensor[group_idx].mean(dim=0)
            elif method == METHOD_ROUTED:
                chosen_expert = int(torch.argmax(utility_matrix[group_idx]).item())
                chosen_reward = reward_tensor[group_idx, chosen_expert]
            else:
                raise ValueError(f"Unsupported method: {method}")
            selected_rewards.append(chosen_reward)

        oracle_rewards = torch.stack(oracle_rewards, dim=0)
        selected_rewards = torch.stack(selected_rewards, dim=0)
        scalarized = (selected_rewards * weights.view(1, 2)).sum(dim=1)
        oracle_scalarized = (oracle_rewards * weights.view(1, 2)).sum(dim=1)
        utilities.append(float(scalarized.mean().item()))
        regrets.append(float((oracle_scalarized - scalarized).mean().item()))
        frontier_points.append((float(selected_rewards[:, 0].mean().item()), float(selected_rewards[:, 1].mean().item())))

    return {
        "utility": float(torch.tensor(utilities).mean().item()),
        "frontier_utility": float(max(utilities)),
        "regret": float(torch.tensor(regrets).mean().item()),
        "hypervolume": hypervolume_2d(frontier_points, reference=(0.0, 0.0)),
        "objective_a_reward": float(max(point[0] for point in frontier_points)),
        "objective_b_reward": float(max(point[1] for point in frontier_points)),
    }


def print_summary(results: Dict[str, Dict[str, float]], objective_a: str, objective_b: str):
    metrics = [
        "utility",
        "frontier_utility",
        "regret",
        "hypervolume",
        "objective_a_reward",
        "objective_b_reward",
    ]
    print(f"\n=== Routed RLHF Summary ({objective_a} vs {objective_b}) ===")
    for method in [METHOD_SINGLE, METHOD_UNIFORM, METHOD_ROUTED]:
        print(f"\n[{method}]")
        for metric in metrics:
            print(f"  {metric:18s}: {results[method][metric]:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="PDF-style routed ensemble validation in an RLHF scenario using HelpSteer2.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--objective-a", type=str, default="coherence", choices=VALID_OBJECTIVES)
    parser.add_argument("--objective-b", type=str, default="verbosity", choices=VALID_OBJECTIVES)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-groups", type=int, default=128)
    parser.add_argument("--eval-groups", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-response-tokens", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--frontier-points", type=int, default=11)
    parser.add_argument("--warmstart-epochs", type=int, default=1)
    parser.add_argument("--reward-prior-scale", type=float, default=6.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(
        f"Loading HelpSteer2 response groups for objectives ({args.objective_a}, {args.objective_b}) "
        f"with train_groups={args.train_groups}, eval_groups={args.eval_groups}"
    )
    train_groups, eval_groups = load_helpsteer_groups(
        args.objective_a,
        args.objective_b,
        args.train_groups,
        args.eval_groups,
        args.seed,
        args.offline,
    )
    print(f"Loaded {len(train_groups)} train groups / {len(eval_groups)} eval groups")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.offline)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {args.model_name} on {device}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        local_files_only=args.offline,
    )
    model = get_peft_model(
        base_model,
        lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout),
        adapter_name=METHOD_SINGLE,
    )

    uniform_adapters = [f"uniform_expert_{idx}" for idx in range(args.num_experts)]
    routed_adapters = [f"routed_expert_{idx}" for idx in range(args.num_experts)]
    for adapter_name in uniform_adapters + routed_adapters:
        model.add_adapter(adapter_name, lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout))

    model.config.use_cache = False
    model.to(device)

    print("\nTraining single policy")
    train_single(model, tokenizer, train_groups, args, device)

    print("\nTraining uniform ensemble")
    warmstart_ensemble(model, tokenizer, train_groups, uniform_adapters, args, device)
    uniform_counts = train_ensemble(model, tokenizer, train_groups, uniform_adapters, METHOD_UNIFORM, args, device)

    print("\nTraining routed ensemble")
    warmstart_ensemble(model, tokenizer, train_groups, routed_adapters, args, device)
    routed_counts = train_ensemble(model, tokenizer, train_groups, routed_adapters, METHOD_ROUTED, args, device)

    print(f"Uniform route counts: {uniform_counts}")
    print(f"Routed route counts: {routed_counts}")

    weight_grid = torch.linspace(0.0, 1.0, steps=args.frontier_points).tolist()
    results = {
        METHOD_SINGLE: evaluate_method(METHOD_SINGLE, model, tokenizer, eval_groups, [METHOD_SINGLE], weight_grid, args, device),
        METHOD_UNIFORM: evaluate_method(METHOD_UNIFORM, model, tokenizer, eval_groups, uniform_adapters, weight_grid, args, device),
        METHOD_ROUTED: evaluate_method(METHOD_ROUTED, model, tokenizer, eval_groups, routed_adapters, weight_grid, args, device),
    }

    print_summary(results, args.objective_a, args.objective_b)

    print("\n=== Verdict ===")
    routed = results[METHOD_ROUTED]
    single = results[METHOD_SINGLE]
    uniform = results[METHOD_UNIFORM]
    if (
        routed["utility"] > max(single["utility"], uniform["utility"])
        and routed["hypervolume"] > max(single["hypervolume"], uniform["hypervolume"])
        and routed["regret"] < min(single["regret"], uniform["regret"])
    ):
        print("Routed ensemble outperforms both the single policy and uniform ensemble on RLHF routing metrics.")
    else:
        print("Routed ensemble does not dominate both baselines under the current RLHF routing setup.")


if __name__ == "__main__":
    main()
