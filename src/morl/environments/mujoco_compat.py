import inspect
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from gym import spaces
from gym.envs.mujoco import mujoco_env


def _infer_timestep(model_path):
    try:
        root = ET.parse(model_path).getroot()
        option = root.find("option")
        if option is None:
            return None
        timestep = option.attrib.get("timestep")
        if timestep is None:
            return None
        return float(timestep)
    except Exception:
        return None


def init_mujoco_env(env, model_path, frame_skip):
    """Initialize MujocoEnv across gym versions with an observation space."""
    expected_render_modes = ["human", "rgb_array", "depth_array"]
    if not hasattr(env, "metadata") or env.metadata is None:
        env.metadata = {}
    if env.metadata.get("render_modes") != expected_render_modes:
        env.metadata["render_modes"] = expected_render_modes
    timestep = _infer_timestep(model_path)
    if timestep is None:
        timestep = 0.01
    env.metadata["render_fps"] = int(np.round(1.0 / (timestep * frame_skip)))

    def ensure_sim():
        if hasattr(env, "sim"):
            return
        if hasattr(env, "data"):
            env.sim = SimpleNamespace(data=env.data)
    if "observation_space" in inspect.signature(mujoco_env.MujocoEnv.__init__).parameters:
        dummy_obs_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(
            env,
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=dummy_obs_space,
        )
        ensure_sim()
        env.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=env._get_obs().shape,
            dtype=np.float64,
        )
    else:
        mujoco_env.MujocoEnv.__init__(env, model_path=model_path, frame_skip=frame_skip)
        ensure_sim()
