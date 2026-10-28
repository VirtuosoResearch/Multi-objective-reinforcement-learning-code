import numpy as np

from morl_baselines.common.performance_indicators import hypervolume


def routed_hypervolumes(records, ref_point, ensemble_size):
    ref_point = np.asarray(ref_point, dtype=float)
    validation = np.asarray([record["selection_returns"] for record in records], dtype=float)
    test = np.asarray([record["candidate_test_returns"] for record in records], dtype=float)
    if validation.shape != test.shape or validation.ndim != 3 or validation.shape[2:] != ref_point.shape:
        raise ValueError("Validation/test returns must have shape (preferences, candidates, reward_dim).")
    if test.shape[1] not in (ensemble_size, ensemble_size + 1):
        raise ValueError("Expected ensemble members and at most one independent single-policy baseline.")
    if not np.all(np.isfinite(validation)) or not np.all(np.isfinite(test)) or not np.all(np.isfinite(ref_point)):
        raise ValueError("HV requires finite reference coordinates and vector returns.")

    def hv(points):
        return float(hypervolume(ref_point, points))

    indices = np.asarray([record["ensemble_member"] for record in records])
    rows = np.arange(len(records))
    test_fronts = {"ensemble": test[rows, indices]}
    validation_fronts = {"ensemble": validation[rows, indices]}
    for j in range(ensemble_size):
        test_fronts[f"member_{j}"] = test[:, j]
        validation_fronts[f"member_{j}"] = validation[:, j]
    has_baseline = test.shape[1] == ensemble_size + 1
    if has_baseline:
        test_fronts["single_policy"] = test[:, -1]
        validation_fronts["single_policy"] = validation[:, -1]
    validation_hv = {name: hv(front) for name, front in validation_fronts.items()}
    test_hv = {name: hv(front) for name, front in test_fronts.items()}
    selected = max(validation_hv, key=validation_hv.get)
    if has_baseline and validation_hv["single_policy"] == validation_hv[selected]:
        selected = "single_policy"
    metrics = {
        "hypervolume": test_hv["ensemble"],
        "ensemble_hypervolume": test_hv["ensemble"],
        "ensemble_validation_hypervolume": validation_hv["ensemble"],
        "member_hypervolumes": [test_hv[f"member_{j}"] for j in range(ensemble_size)],
        "selected_policy": selected,
        "selected_hypervolume": test_hv[selected],
        "selected_validation_hypervolume": validation_hv[selected],
        "oracle_best_policy_hypervolume": max(test_hv.values()),
        "candidate_union_hypervolume": hv(test.reshape(-1, test.shape[-1])),
        "ref_point": ref_point.tolist(),
    }
    if has_baseline:
        metrics.update(
            {
                "single_policy_hypervolume": test_hv["single_policy"],
                "single_policy_validation_hypervolume": validation_hv["single_policy"],
                "ensemble_hv_minus_single_policy": test_hv["ensemble"] - test_hv["single_policy"],
                "selected_hv_minus_single_policy": test_hv[selected] - test_hv["single_policy"],
            }
        )
    return metrics
