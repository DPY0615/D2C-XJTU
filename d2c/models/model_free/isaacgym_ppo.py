# d2c/models/model_free/isaacgym_ppo.py
"""
ISAACGYM PPO Agent (D2C style)

This file MUST define exactly two public classes:
  1) ISAACGYM_PPOAgent(BaseAgent)
  2) AgentModule(BaseAgentModule)

It uses the vendored legged-gym/rsl_rl-like PPO in:
  - d2c.networks_and_utils_for_agent.isaacgym_ppo_utils.ppo.PPO
"""

import collections
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.utils import utils

from d2c.networks_and_utils_for_agent.isaacgym_ppo_utils.actor_critic import ActorCritic
from d2c.networks_and_utils_for_agent.isaacgym_ppo_utils.mlp_encoder import MLP_Encoder
from d2c.networks_and_utils_for_agent.isaacgym_ppo_utils.ppo import PPO as RslPPO


def _as_tensor(x, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.ndim == 1 else x


def _to_scalar_tensor(x, device: torch.device) -> torch.Tensor:
    """Convert float/np scalar/torch scalar to torch tensor on device."""
    if torch.is_tensor(x):
        return x.to(device=device)
    return torch.tensor(float(x), device=device)


class ISAACGYM_PPOAgent(BaseAgent):
    """
    PPO agent for IsaacGym/LeggedGym VecEnv (D2C agent style),
    wired to the vendored PPO implementation.
    """
    _ACTION_CLIP = 1.0

    def __init__(
        self,
        total_timesteps: int = 1_000_000,
        num_envs: int = 1,
        num_steps: int = 24,
        update_epochs: int = 5,
        num_minibatches: int = 4,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        target_kl: Optional[float] = 0.01,
        anneal_lr: bool = False,
        num_group: int = 1,
        est_learning_rate: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        # hparams
        self._total_timesteps = int(total_timesteps)
        self._num_envs_cfg = int(num_envs)
        self._num_steps = int(num_steps)

        self._update_epochs = int(update_epochs)
        self._num_minibatches = int(num_minibatches)

        self._gae_lambda = float(gae_lambda)
        self._clip_coef = float(clip_coef)
        self._ent_coef = float(ent_coef)
        self._vf_coef = float(vf_coef)
        self._max_grad_norm = float(max_grad_norm)
        self._target_kl = None if target_kl is None else float(target_kl)
        self._anneal_lr = bool(anneal_lr)

        self._num_group = int(num_group)
        self._est_learning_rate = None if est_learning_rate is None else float(est_learning_rate)

        # runtime shapes (detected in _prepare_for_train)
        self._num_envs: int = 0
        self._obs_dim: int = 0
        self._critic_raw_dim: int = 0
        self._cmd_dim: int = 0
        self._obs_history_dim: int = 0

        self._enc_out: int = 16
        self._enc_hidden = [256, 128, 64]

        # runtime
        self._batch_size: int = 0
        self._num_iterations: int = 0
        self._current_iteration: int = 0

        self._obs: Optional[torch.Tensor] = None
        self._obs_history: Optional[torch.Tensor] = None
        self._commands: Optional[torch.Tensor] = None
        self._critic_raw: Optional[torch.Tensor] = None
        self._dones: Optional[torch.Tensor] = None

        self._actor_critic: Optional[ActorCritic] = None
        self._encoder: Optional[MLP_Encoder] = None
        self._ppo: Optional[RslPPO] = None

        super().__init__(**kwargs)

    # ---------------- D2C required hooks ---------------- #
    def _get_modules(self) -> utils.Flags:
        model_params_q, _ = self._model_params.q
        model_params_p = self._model_params.p[0]

        # placeholder dims (will be rebuilt in _prepare_for_train)
        ph_enc_in = 1
        ph_cmd_dim = 0

        def encoder_factory():
            return MLP_Encoder(
                num_input_dim=ph_enc_in,
                num_output_dim=self._enc_out,
                hidden_dims=list(self._enc_hidden),
                activation="elu",
                orthogonal_init=False,
                output_detach=False,
            )

        def actor_critic_factory():
            base_obs_dim = int(getattr(self._env, "num_obs", self._env.observation_space.shape[0]))
            num_actions = int(getattr(self._env, "num_actions", self._env.action_space.shape[0]))

            # actor input in vendored PPO: [enc_out, obs, commands]
            num_actor_obs = base_obs_dim + self._enc_out + ph_cmd_dim
            # critic input in vendored PPO.act(): [critic_raw, commands] (critic_take_latent default False)
            num_critic_obs = base_obs_dim + ph_cmd_dim

            return ActorCritic(
                num_actor_obs=num_actor_obs,
                num_critic_obs=num_critic_obs,
                num_actions=num_actions,
                actor_hidden_dims=list(model_params_p),
                critic_hidden_dims=list(model_params_q),
                activation="elu",
                orthogonal_init=False,
                init_noise_std=1.0,
            )

        return utils.Flags(
            actor_critic_factory=actor_critic_factory,
            encoder_factory=encoder_factory,
            device=self._device,
        )

    def _init_vars(self) -> None:
        self._num_envs = int(getattr(self._env, "num_envs", self._num_envs_cfg))
        self._batch_size = int(self._num_envs * self._num_steps)
        self._num_iterations = max(1, self._total_timesteps // max(1, self._batch_size))
        self._current_iteration = 0

        self._obs = None
        self._obs_history = None
        self._commands = None
        self._critic_raw = None
        self._dones = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)

    def _build_fns(self) -> None:
        self._agent_module = AgentModule(modules=self._modules)
        self._actor_critic = self._agent_module.actor_critic
        self._encoder = self._agent_module.encoder

        lr = float(self._optimizers.ac[1]) if hasattr(self, "_optimizers") else 1e-3

        self._ppo = RslPPO(
            num_group=self._num_envs,
            encoder=self._encoder,
            actor_critic=self._actor_critic,
            num_learning_epochs=int(self._update_epochs),
            num_mini_batches=int(self._num_minibatches),
            clip_param=float(self._clip_coef),
            gamma=float(self._discount),
            lam=float(self._gae_lambda),
            value_loss_coef=float(self._vf_coef),
            entropy_coef=float(self._ent_coef),
            learning_rate=lr,
            max_grad_norm=float(self._max_grad_norm),
            use_clipped_value_loss=True,
            schedule="fixed",
            desired_kl=0.02,
            early_stop=True,
            device=self._device,
        )

        # eps align
        if hasattr(self, "_optimizers") and len(self._optimizers.ac) >= 3:
            eps = float(self._optimizers.ac[2])
            for g in self._ppo.optimizer.param_groups:
                g["eps"] = eps

    def _build_optimizers(self) -> None:
        return

    def _build_test_policies(self) -> None:
        # IMPORTANT: evaluation should be deterministic.
        def _policy(obs):
            assert self._actor_critic is not None and self._encoder is not None
            obs_t = _ensure_2d(_as_tensor(obs, self._device, torch.float32))

            hist = obs_t.new_zeros((obs_t.shape[0], max(self._obs_history_dim, 1)))
            cmds = obs_t.new_zeros((obs_t.shape[0], self._cmd_dim)) if self._cmd_dim > 0 else obs_t.new_zeros((obs_t.shape[0], 0))

            enc_out = self._encoder.encode(hist)
            actor_in = torch.cat((enc_out, obs_t, cmds), dim=-1)

            act = self._actor_critic.act_inference(actor_in)
            act = torch.clamp(act, -self._ACTION_CLIP, self._ACTION_CLIP)
            return act

        self._test_policies["main"] = _policy

    # BaseAgent abstract stub
    def _optimize_step(self, batch: Dict) -> Dict:  # pylint: disable=unused-argument
        return {}

    # ---------------- env helpers ---------------- #
    def _env_reset(self, seed: Optional[int] = None):
        out = self._env.reset(seed=seed) if seed is not None else self._env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            return out[0]
        return out

    def _env_step_extras(self, actions: torch.Tensor):
        """
        Return (obs, rew, done, info, obs_history, commands, critic_raw).
        Works with your IsaacGymEnv wrapper which supports return_extras=True.
        """
        try:
            out = self._env.step(actions, return_extras=True)
        except TypeError:
            out = self._env.step(actions)

        if not isinstance(out, (tuple, list)):
            raise RuntimeError(f"env.step returned non-tuple: {type(out)}")

        if len(out) >= 7:
            obs, rew, done, info, obs_hist, commands, critic_raw = out[:7]
            info = dict(info) if isinstance(info, dict) else {}
            return obs, rew, done, info, obs_hist, commands, critic_raw

        if len(out) == 4:
            obs, rew, done, info = out
            info = dict(info) if isinstance(info, dict) else {}
            return obs, rew, done, info, info.get("obs_history", None), info.get("commands", None), info.get("critic_obs", None)

        if len(out) == 5:
            obs, rew, terminated, truncated, info = out
            done = (_as_tensor(terminated, self._device, torch.bool) | _as_tensor(truncated, self._device, torch.bool))
            info = dict(info) if isinstance(info, dict) else {}
            info.setdefault("time_outs", truncated)
            return obs, rew, done, info, info.get("obs_history", None), info.get("commands", None), info.get("critic_obs", None)

        raise RuntimeError(f"Unsupported env.step return len={len(out)}")

    def _rebuild_for_detected_shapes(self, obs_dim: int, critic_raw_dim: int, obs_hist_dim: int, cmd_dim: int) -> None:
        """Rebuild encoder/actor_critic/PPO with detected env dims."""
        model_params_q, _ = self._model_params.q
        model_params_p = self._model_params.p[0]
        num_actions = int(getattr(self._env, "num_actions", self._env.action_space.shape[0]))

        self._obs_dim = int(obs_dim)
        self._critic_raw_dim = int(critic_raw_dim)
        self._obs_history_dim = int(obs_hist_dim)
        self._cmd_dim = int(cmd_dim)

        # rebuild encoder
        self._encoder = MLP_Encoder(
            num_input_dim=self._obs_history_dim if self._obs_history_dim > 0 else 1,
            num_output_dim=self._enc_out,
            hidden_dims=list(self._enc_hidden),
            activation="elu",
            orthogonal_init=False,
            output_detach=False,
        ).to(self._device)

        # ActorCritic dims must match vendored PPO internal concatenations:
        # actor input: [enc_out, obs, commands]
        num_actor_obs = self._enc_out + self._obs_dim + self._cmd_dim
        # critic input in act(): [critic_raw, commands] (critic_take_latent default False)
        num_critic_obs = self._critic_raw_dim + self._cmd_dim

        self._actor_critic = ActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=list(model_params_p),
            critic_hidden_dims=list(model_params_q),
            activation="elu",
            orthogonal_init=False,
            init_noise_std=1.0,
        ).to(self._device)

        lr = float(self._optimizers.ac[1]) if hasattr(self, "_optimizers") else 1e-3

        self._ppo = RslPPO(
            num_group=self._num_envs,  
            encoder=self._encoder,
            actor_critic=self._actor_critic,
            num_learning_epochs=int(self._update_epochs),
            num_mini_batches=int(self._num_minibatches),
            clip_param=float(self._clip_coef),
            gamma=float(self._discount),
            lam=float(self._gae_lambda),
            value_loss_coef=float(self._vf_coef),
            entropy_coef=float(self._ent_coef),
            learning_rate=lr,
            max_grad_norm=float(self._max_grad_norm),
            use_clipped_value_loss=True,
            schedule="fixed",
            desired_kl=None,
            device=self._device,
        )

        if hasattr(self, "_optimizers") and len(self._optimizers.ac) >= 3:
            eps = float(self._optimizers.ac[2])
            for g in self._ppo.optimizer.param_groups:
                g["eps"] = eps

    # ---------------- trainer entry ---------------- #
    def _prepare_for_train(self, _train_steps: int, env_seed: int) -> int:
        self._total_timesteps = int(_train_steps)

        obs0 = self._env_reset(seed=env_seed)
        obs0_t = _ensure_2d(_as_tensor(obs0, self._device, torch.float32))

        self._num_envs = int(getattr(self._env, "num_envs", self._num_envs_cfg))
        self._batch_size = int(self._num_envs * self._num_steps)
        self._num_iterations = max(1, self._total_timesteps // max(1, self._batch_size))
        self._current_iteration = 0

        # one step with zero actions to detect extras dims
        num_actions = int(getattr(self._env, "num_actions", self._env.action_space.shape[0]))
        zero_act = obs0_t.new_zeros((self._num_envs, num_actions))
        o1, _r, _d, _info, h1, c1, cr1 = self._env_step_extras(zero_act)

        o1_t = _ensure_2d(_as_tensor(o1, self._device, torch.float32))

        h1_t = _ensure_2d(_as_tensor(h1, self._device, torch.float32)) if h1 is not None else o1_t.new_zeros((self._num_envs, 1))
        c1_t = _ensure_2d(_as_tensor(c1, self._device, torch.float32)) if c1 is not None else o1_t.new_zeros((self._num_envs, 0))
        cr1_t = _ensure_2d(_as_tensor(cr1, self._device, torch.float32)) if cr1 is not None else o1_t

        obs_dim = int(o1_t.shape[-1])
        obs_hist_dim = int(h1_t.shape[-1])
        cmd_dim = int(c1_t.shape[-1])
        critic_raw_dim = int(cr1_t.shape[-1])

        self._rebuild_for_detected_shapes(obs_dim, critic_raw_dim, obs_hist_dim, cmd_dim)
        assert self._ppo is not None

        # storage shapes:
        # actor_obs_shape: raw obs
        # critic_obs_shape: [critic_raw, commands]
        critic_full_dim = critic_raw_dim + cmd_dim

        self._ppo.init_storage(
            num_envs=self._num_envs,
            num_transitions_per_env=int(self._num_steps),
            actor_obs_shape=(obs_dim,),
            critic_obs_shape=(critic_full_dim,),
            obs_history_shape=(obs_hist_dim,),
            commands_shape=(cmd_dim,),
            action_shape=(num_actions,),
        )

        # set state
        self._obs = o1_t
        self._obs_history = h1_t
        self._commands = c1_t
        self._critic_raw = cr1_t
        self._dones = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)

        return self._num_iterations

    def train_step(self) -> None:
        assert self._ppo is not None
        assert self._encoder is not None
        assert self._actor_critic is not None
        assert self._obs is not None and self._obs_history is not None and self._commands is not None and self._critic_raw is not None

        # anneal lr if requested
        if self._anneal_lr and self._num_iterations > 0:
            frac = 1.0 - (self._current_iteration / float(self._num_iterations))
            lr_now = frac * float(self._optimizers.ac[1])
            for g in self._ppo.optimizer.param_groups:
                g["lr"] = lr_now

        dbg_nonfinite_action = 0.0
        dbg_action_clipped = 0.0

        rew_sum = 0.0
        done_sum = 0.0


        ep_infos = []  

        for _ in range(self._num_steps):
            actions = self._ppo.act(self._obs, self._obs_history, self._commands, self._critic_raw)

            if actions.abs().max().item() >= (self._ACTION_CLIP - 1e-6):
                dbg_action_clipped = 1.0
                
            nxt_obs, rew, done, info, nxt_hist, nxt_cmd, nxt_critic = self._env_step_extras(actions)

            nxt_obs_t = _ensure_2d(_as_tensor(nxt_obs, self._device, torch.float32))

            rew_t = _ensure_2d(_as_tensor(rew, self._device, torch.float32)).view(self._num_envs, 1)
            done_t = _ensure_2d(_as_tensor(done, self._device, torch.float32)).view(self._num_envs, 1)

            nxt_hist_t = _ensure_2d(_as_tensor(nxt_hist, self._device, torch.float32)) if nxt_hist is not None else self._obs_history.new_zeros((self._num_envs, self._obs_history.shape[-1]))
            nxt_cmd_t = _ensure_2d(_as_tensor(nxt_cmd, self._device, torch.float32)) if nxt_cmd is not None else self._commands.new_zeros((self._num_envs, self._commands.shape[-1]))
            nxt_critic_t = _ensure_2d(_as_tensor(nxt_critic, self._device, torch.float32)) if nxt_critic is not None else nxt_obs_t

            info = dict(info) if isinstance(info, dict) else {}
            self._ppo.process_env_step(rew_t, done_t, info, next_obs=nxt_obs_t)

            rew_sum += float(rew_t.mean().item())
            done_sum += float(done_t.mean().item())


            if "episode" in info:
                ep_infos.append(info["episode"])

            self._obs = nxt_obs_t
            self._obs_history = nxt_hist_t
            self._commands = nxt_cmd_t
            self._critic_raw = nxt_critic_t
            self._dones = done_t

        # compute_returns needs critic_obs_full ([critic_raw, commands])
        last_critic_full = torch.cat((self._critic_raw, self._commands), dim=-1) if self._cmd_dim > 0 else self._critic_raw
        self._ppo.compute_returns(last_critic_full)

        mean_v_loss, mean_extra_loss, mean_pi_loss, mean_kl = self._ppo.update()

        self._global_step += int(self._batch_size)
        self._current_iteration += 1

        # ============ diagnostics ============
        with torch.no_grad():
            explained_var = torch.tensor(float("nan"), device=self._device)

            returns = getattr(getattr(self._ppo, "storage", None), "returns", None)
            values = getattr(getattr(self._ppo, "storage", None), "values", None)
            if returns is not None and values is not None:
                y_true = returns.flatten()
                y_pred = values.flatten()
                var_y = torch.var(y_true)
                if torch.isfinite(var_y) and var_y > 1e-12:
                    explained_var = 1.0 - torch.var(y_true - y_pred) / var_y


        act_t = None
        if hasattr(self._ppo, "transition") and self._ppo.transition is not None:
            act_t = getattr(self._ppo.transition, "actions", None)


        reward_all_zero = 0.0
        if returns is not None:
            reward_all_zero = float((torch.abs(returns).max() < 1e-12).item())

        if torch.is_tensor(act_t):
            act_safe = torch.nan_to_num(act_t, nan=0.0, posinf=0.0, neginf=0.0)
            action_mean = act_safe.mean()
            action_std = act_safe.std()
            action_absmax = act_safe.abs().max()
        else:
            # fall back: we don't have transition actions (or it's None)
            action_mean = torch.tensor(float("nan"), device=self._device)
            action_std = torch.tensor(float("nan"), device=self._device)
            action_absmax = torch.tensor(float("nan"), device=self._device)

        # ============ logging ============
        self._train_info = collections.OrderedDict()
        self._train_info["global_step"] = torch.tensor(self._global_step, device=self._device)

        self._train_info["actor_loss"] = _to_scalar_tensor(mean_pi_loss, self._device)
        self._train_info["v_loss"] = _to_scalar_tensor(mean_v_loss, self._device)
        self._train_info["estimator_loss"] = _to_scalar_tensor(mean_extra_loss, self._device)
        self._train_info["approx_kl"] = _to_scalar_tensor(mean_kl, self._device)
        self._train_info["explained_variance"] = explained_var
        self._train_info["learning_rate"] = torch.tensor(
            self._ppo.learning_rate, device=self._device
        )

        if ep_infos:
            ep_rew_total = 0.0
            ep_count = 0
            ep_reward_components = collections.defaultdict(float)
            for ep in ep_infos:
                if isinstance(ep, dict):
                    for k, v in ep.items():
                        val = float(v.item()) if torch.is_tensor(v) else float(v)
                        ep_reward_components[k] += val
                    ep_count += 1
            if ep_count > 0:
                for k in ep_reward_components:
                    ep_reward_components[k] /= ep_count
                for k, v in ep_reward_components.items():
                    if k.startswith("rew_"):
                        ep_rew_total += v
                    self._train_info[f"ep/{k}"] = torch.tensor(v, device=self._device)
                self._train_info["ep/rew_mean"] = torch.tensor(ep_rew_total, device=self._device)

        env_inner = getattr(self._env, "_env", self._env)
        if hasattr(env_inner, "episode_length_buf"):
            ep_len = float(env_inner.episode_length_buf.float().mean().item())
            self._train_info["ep/len_mean"] = torch.tensor(ep_len, device=self._device)

    def save(self, ckpt_name: str) -> None:
        return

    def restore(self, ckpt_name: str) -> None:
        return


class AgentModule(BaseAgentModule):
    def _build_modules(self) -> None:
        device = self._net_modules.device
        self._actor_critic = self._net_modules.actor_critic_factory().to(device)
        self._encoder = self._net_modules.encoder_factory().to(device)

    @property
    def actor_critic(self) -> nn.Module:
        return self._actor_critic

    @property
    def encoder(self) -> nn.Module:
        return self._encoder