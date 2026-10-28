## Multi-Objective Reinforcement Learning Using Routed Ensembles and Trajectory Attribution

Code for the paper [Multi-Objective Reinforcement Learning Using Routed Ensembles and Trajectory Attribution](https://arxiv.org/abs/2610.35733).

## Structure
- `src/morl`: routed Q-Pensieve ensembles on MO-Gymnasium MuJoCo (HalfCheetah, Hopper, Ant)
- `src/morl-baselines`: full MORL-Baselines library with ensemble examples (DST, Fruit Tree, MuJoCo)
- `src/mtrl`: routed ensembles on Meta-World MT10
- `src/rlhf`: routed ensembles for RLHF (HelpSteer2 and synthetic)
- `src/momdp`, `src/toyex`, `src/*weighted_estimator*.py`: tabular MOMDP experiments and weighted off-policy estimators
- `src/ksm`, `src/rl`: trajectory attribution method and single-objective RL experiments (see `src/rl/README.md`)
- `figures/`: plotting notebooks

## Run experiments
MO-Gymnasium (from the repository root):
```bash
cd src/morl && pip install -e .
python examples/qpensieve_routed.py --env-id mo-halfcheetah-v4
```
Meta-World (from the repository root):
```bash
cd src/mtrl && pip install -e .
python routed_ensemble_mt10.py
```

## Citation
If you use this code, please cite:
```bibtex
@inproceedings{zhang2026morl,
  title={Multi-Objective Reinforcement Learning Using Routed Ensembles and Trajectory Attribution},
  author={Zhang, Zhenshuo and Xiong, Ruoxuan and Zhang, Hongyang R.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
}
```
