from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym
from d2c.envs.external.isaacgym.utils.task_registry import task_registry
from d2c.envs.external.isaacgym.utils.helpers import get_args, set_seed
import torch

class IsaacGymEnv:
	"""
	The wrapper that exposes IsaacGym/legged-gym style VecEnv interface:
	- reset(...) -> obs (torch)
	- step(actions) -> obs, rewards, dones, infos (4 returns)
	- reset_idx(env_ids)
	- num_envs/num_obs/num_actions/device properties
	It also provides gymnasium-style spaces (metadata only) and make_env_space() for D2C config inference (without forcing 5-return gymnasium semantics).
	"""
	def __init__(self, config: Any, **kwargs: Any) -> None:
		self._cfg = config
		ext = config.model_config.env.external
		args = get_args()
		if hasattr(ext, "task") and ext.task is not None:
			args.task = ext.task
		if hasattr(ext, "headless") and ext.headless is not None:
			args.headless = ext.headless
		if hasattr(ext, "num_envs") and ext.num_envs is not None:
			args.num_envs = ext.num_envs
		if hasattr(ext, "resume") and ext.resume is not None:
			args.resume = ext.resume
		env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=None)
		self._env = env
		self._env_cfg = env_cfg
		self.num_envs = int(getattr(self._env, "num_envs"))
		self.num_obs = int(getattr(self._env, "num_obs"))
		self.num_actions = int(getattr(self._env, "num_actions"))
		self.device = getattr(self._env, "device", None)
		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32,
		)
		self.action_space = gym.spaces.Box(
			low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32,
		)

	def __getattr__(self, name: str):
		return getattr(self._env, name)

	def _get_obs_from_env(self):
		"""Recover obs from buffers while reset returns None."""
		if hasattr(self._env, "get_observations"):
			try:
				o = self._env.get_observations()
				if o is not None:
					return o
			except Exception:
				pass
		for name in ("obs_buf", "_obs_buf"):
			if hasattr(self._env, name):
				o = getattr(self._env, name)
				if o is not None:
					return o
		raise RuntimeError("Failed to recover observations from env (reset/reset_idx returned None).")

	def reset(
		self,
		*,
		seed: Optional[int] = None,
		return_info: bool = False,
		options: Optional[dict] = None,
		**kwargs: Any,
	):
		if seed is not None:
			set_seed(seed)
		obs = self._env.reset()
		if obs is None:
			obs = self._get_obs_from_env()
		if return_info:
			return obs, {}
		return obs

	def reset_idx(self, env_ids):
		if not hasattr(self._env, "reset_idx"):
			return self.reset()

		dev = self.device if self.device is not None else "cpu"
		if isinstance(env_ids, np.ndarray):
			env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=dev)
		elif torch.is_tensor(env_ids):
			env_ids_t = env_ids.to(device=dev, dtype=torch.long)
		else:
			env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=dev)

		self._env.reset_idx(env_ids_t)

		if hasattr(self._env, "compute_observations"):
			try:
				self._env.compute_observations()
			except Exception:
				pass

		if hasattr(self._env, "get_observations"):
			try:
				o = self._env.get_observations()
				if isinstance(o, (tuple, list)) and len(o) > 0:
					o = o[0]
				if o is not None:
					return o
			except Exception:
				pass

		if hasattr(self._env, "obs_buf") and self._env.obs_buf is not None:
			return self._env.obs_buf

		return self.reset()


	def step(self, actions, return_extras: bool = False):
		"""Wrapper for IsaacGym/LeggedGym VecEnv.

			Underlying env may return:
			- 7-tuple: (obs, reward, done, info, obs_history, commands, critic_obs)
			- 4-tuple: (obs, reward, done, info)
			- 5-tuple: (obs, reward, terminated, truncated, info)  (gymnasium)
		"""
		out = self._env.step(actions)

		if return_extras:
			return out

		# Pass-through if already 4-return
		if isinstance(out, tuple) and len(out) == 4:
			return out

		# Common LeggedGym VecEnv 7-return
		if isinstance(out, tuple) and len(out) >= 7:
			obs, rewards, dones, infos, obs_history, commands, critic_obs = out[:7]
			if infos is None or not isinstance(infos, dict):
				infos = {}
			infos = dict(infos)
			infos.update(
				{
					"obs_history": obs_history,
					"commands": commands,
					"critic_obs": critic_obs,
				}
			)
			return obs, rewards, dones, infos

		# Gymnasium style 5-return -> 4-return
		if isinstance(out, tuple) and len(out) == 5:
			obs, rewards, terminated, truncated, infos = out
			if infos is None or not isinstance(infos, dict):
				infos = {}
			infos = dict(infos)
			infos.setdefault("time_outs", truncated)
			done = terminated | truncated
			return obs, rewards, done, infos

		raise RuntimeError(f"Unexpected env.step() return: type={type(out)} len={len(out) if isinstance(out,(tuple,list)) else 'NA'}")

	@staticmethod
	def make_env_space(
		data_source: Optional[str] = None,
		env_name: Optional[str] = None,
		task: Optional[str] = None,
		**kwargs: Any,
	):
		if task is None:
			task = kwargs.get("task", None) or env_name
		if task is None:
			raise ValueError("IsaacGymEnv.make_env_space() requires task or env_name.")
		env_cfg, _ = task_registry.get_cfgs(task)
		num_obs = int(getattr(env_cfg.env, "num_observations"))
		num_actions = int(getattr(env_cfg.env, "num_actions"))
		obs_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32,
		)
		act_space = gym.spaces.Box(
			low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32,
		)
		return SimpleNamespace(observation=obs_space, action=act_space)