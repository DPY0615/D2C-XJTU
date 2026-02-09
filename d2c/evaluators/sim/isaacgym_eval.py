"""
IsaacGym Evaluator for D2C.

设计思路
========
IsaacGym VecEnv 的特点：所有 num_envs 个环境并行运行，单个 env 结束时自动
reset，每次 reset 时 env 会把该 episode 的奖励统计写进 extras["episode"]。

原始 legged_gym（rsl_rl on_policy_runner）的做法是：在训练 rollout 中直接
收集 extras["episode"]，然后打印 mean reward / mean episode length，不做
单独的 eval 循环。

旧的 IsaacGymEval 继承自 BMEval，试图在同一个 env 上重新跑 N 个 episode 来
评估。这有两个问题：
  1. 没有 torch.no_grad()、没有释放中间 tensor → OOM
  2. IsaacGym VecEnv 不支持"跑完一个 episode 就 break"的模式

新方案
------
完全不做独立的 eval rollout。改为：

  1. agent.train_step() 中每一步把 extras["episode"] 收集起来
  2. agent._train_info 中包含 ep/rew_mean, ep/len_mean 等统计
  3. IsaacGymEval.eval() 直接读 agent._train_info 并记录到 tensorboard
  4. 零额外显存开销
"""

import os
import logging
import collections
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple, List, Any

from d2c.evaluators.base import BaseEval
from d2c.utils import utils
from d2c.utils.logger import write_summary_tensorboard


class IsaacGymEval(BaseEval):
    """IsaacGym evaluator that reads episode stats from training rollout.

    不做任何额外的环境交互，完全从 agent._train_info 中的
    ep/rew_mean, ep/len_mean 等 key 读取已在 train_step 中收集好的
    episode 统计。

    :param str result_dir: tensorboard log 目录
    :param agent: ISAACGYM_PPOAgent 实例
    :param env: IsaacGymEnv 实例（仅用于兼容接口，不会被调用 step）
    """

    def __init__(
        self,
        result_dir: str,
        agent: Any,
        env: Any = None,
        **kwargs: Any,
    ) -> None:
        self._agent = agent
        self._eval_summary_dir = result_dir
        self._eval_summary_writers = self._build_writer()
        self._eval_r_results = []

    # ------------------------------------------------------------------ #
    # 核心 eval 方法
    # ------------------------------------------------------------------ #
    def eval(self, step: int) -> Dict:
        """读 agent._train_info 中的 episode 统计并写入 tensorboard。"""
        return_info = {}
        info = collections.OrderedDict()
        ti = getattr(self._agent, "_train_info", {})

        # 从 train_info 收集 ep/* 统计
        for k, v in ti.items():
            if k.startswith("ep/"):
                val = float(v.item()) if torch.is_tensor(v) else float(v)
                info[k] = val

        # 如果 train_info 没有 ep/* key（agent 还没跑 train_step），
        # 给一个默认值
        if not info:
            info["ep/rew_mean"] = 0.0
            info["ep/len_mean"] = 0.0

        # 记录
        ep_rew = info.get("ep/rew_mean", 0.0)
        ep_len = info.get("ep/len_mean", 0.0)
        self._eval_r_results.append([step, ep_rew])

        # 打印
        logging.info(
            f"[Eval] step={step}  ep_rew_mean={ep_rew:.4f}  ep_len_mean={ep_len:.1f}"
        )

        # 写 tensorboard
        for policy_key, writer in self._eval_summary_writers.items():
            write_summary_tensorboard(writer, step, info)
            for k, v in info.items():
                return_info[policy_key + "-" + k] = v

        return return_info

    def save_eval_results(self) -> None:
        if self._eval_r_results:
            results_file = os.path.join(self._eval_summary_dir, "results_reward.npy")
            np.save(results_file, np.array(self._eval_r_results))
            logging.info(f"Eval results saved to {results_file}")
        for writer in self._eval_summary_writers.values():
            writer.close()

    def _build_writer(self) -> Dict[str, SummaryWriter]:
        writers = collections.OrderedDict()
        policies = getattr(self._agent, "test_policies", {"main": None})
        for key in policies.keys():
            writers[key] = SummaryWriter(
                os.path.join(self._eval_summary_dir, key)
            )
        return writers
    
    def _eval_policies(self, step: int) -> Dict:
        """
        D2C BaseEval 要求的抽象方法。

        IsaacGym 情况下不做 per-policy rollout,
        直接复用 eval() 的结果即可。
        """
        return self.eval(step)