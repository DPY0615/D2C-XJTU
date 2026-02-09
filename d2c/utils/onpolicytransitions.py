# utils/onpolicytransitions.py
from typing import Dict, Any
import torch


class OnPolicyTransitions:
    """
    Rollout storage (T, N, ...)
    plus timeout-aware terminal value for correct bootstrapping (leggedgym/rsl_rl style).
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape,
        action_shape,
        device="cpu",
    ):
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)

        self.obs = torch.zeros((self.num_steps, self.num_envs) + tuple(obs_shape), device=self.device, dtype=torch.float32)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + tuple(action_shape), device=self.device, dtype=torch.float32)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs), device=self.device, dtype=torch.float32)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), device=self.device, dtype=torch.float32)

        # "done for next state" buffer you already use (float 0/1)
        self.dones = torch.zeros((self.num_steps, self.num_envs), device=self.device, dtype=torch.float32)

        # additionally store terminated/truncated explicitly (float 0/1)
        self.terminated = torch.zeros((self.num_steps, self.num_envs), device=self.device, dtype=torch.float32)
        self.truncated = torch.zeros((self.num_steps, self.num_envs), device=self.device, dtype=torch.float32)

        # value estimates V(s_t)
        self.values = torch.zeros((self.num_steps, self.num_envs), device=self.device, dtype=torch.float32)

        # timeout-aware bootstrapping:
        # timeouts[t]=1 means step t ended due to timeout/truncation
        self.timeouts = torch.zeros((self.num_steps, self.num_envs), device=self.device, dtype=torch.float32)

        # terminal_values[t] = V(s_{t+1}^{terminal}) for those timeout envs (else 0)
        self.terminal_values = torch.zeros((self.num_steps, self.num_envs), device=self.device, dtype=torch.float32)

    def get_batch(self) -> Dict[str, Any]:
        return {
            "s1": self.obs,
            "a1": self.actions,
            "logprob": self.logprobs,
            "reward": self.rewards,
            "done": self.dones,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "value": self.values,
            "timeouts": self.timeouts,
            "terminal_value": self.terminal_values,
        }
