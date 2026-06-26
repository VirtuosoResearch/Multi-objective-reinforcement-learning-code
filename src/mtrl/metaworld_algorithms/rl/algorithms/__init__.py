from metaworld_algorithms.config.rl import AlgorithmConfig

from .base import Algorithm, OffPolicyAlgorithm
from .mtsac import MTSAC, MTSACConfig
from .routed_ensemble import RoutedEnsembleMTSAC, RoutedEnsembleConfig
from .routed_ensemble_vmap import RoutedEnsembleMTSACVmap, RoutedEnsembleVmapConfig
from .ppo import PPOConfig, PPO
from .sac import SAC, SACConfig
from .mamltrpo import MAMLTRPO, MAMLTRPOConfig
from .rl2 import RL2, RL2Config


def get_algorithm_for_config(config: AlgorithmConfig) -> type[Algorithm]:
    if type(config) is MTSACConfig:
        print("Using MTSAC algorithm")
        return MTSAC
    elif type(config) is RoutedEnsembleConfig:
        print("Using RoutedEnsembleMTSAC algorithm")
        return RoutedEnsembleMTSAC
    elif type(config) is RoutedEnsembleVmapConfig:
        print("Using RoutedEnsembleMTSACVmap algorithm")
        return RoutedEnsembleMTSACVmap
    elif type(config) is PPOConfig:
        print("Using PPO algorithm")
        return PPO
    elif type(config) is SACConfig:
        print("Using SAC algorithm")
        return SAC
    elif type(config) is MAMLTRPOConfig:
        print("Using MAMLTRPO algorithm")
        return MAMLTRPO
    elif type(config) is RL2Config:
        print("Using RL2 algorithm")
        return RL2
    else:
        raise ValueError(f"Invalid config type: {type(config)}")


__all__ = [
    "Algorithm",
    "OffPolicyAlgorithm",
    "MTSAC",
    "RoutedEnsembleMTSAC",
    "RoutedEnsembleMTSACVmap",
    "PPO",
    "MTSACConfig",
    "RoutedEnsembleConfig",
    "RoutedEnsembleVmapConfig",
    "PPOConfig",
]
