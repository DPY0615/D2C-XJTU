import os
import time
import logging
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from d2c.trainers.base import BaseTrainer
from d2c.utils import utils

class IsaacGymOnPolicyTrainer(BaseTrainer):
    def __init__(self, agent: Any, train_data: Any, config: Any, env: Any = None, evaluator: Any = None) -> None:
        super().__init__(agent, env, train_data, config)

        self._train_iters = self._train_cfg.total_train_steps  

        self._summary_freq = getattr(self._train_cfg, "on_policy_summary_freq", self._train_cfg.summary_freq)
        self._print_freq   = getattr(self._train_cfg, "on_policy_print_freq",   self._train_cfg.print_freq)
        self._save_freq    = getattr(self._train_cfg, "on_policy_save_freq",    self._train_cfg.save_freq)
        self._eval_freq    = getattr(self._train_cfg, "on_policy_eval_freq",    self._train_cfg.eval_freq)

        self._evaluator = evaluator

    def _train_behavior(self) -> None:
        pass

    def _train_dynamics(self) -> None:
        pass

    def _train_q(self) -> None:
        pass

    def _train_vae_s(self) -> None:
        pass
    
    def _train_agent(self) -> None:
        agent_ckpt_dir = self._train_cfg.agent_ckpt_dir
        utils.maybe_makedirs(os.path.dirname(agent_ckpt_dir))
        train_summary_dir = agent_ckpt_dir + "_train_log"
        writer = SummaryWriter(train_summary_dir)

        total_iterations = self._agent._prepare_for_train(
            self._train_cfg.total_train_steps,
            self._train_cfg.seed,
        )

        t0 = time.time()
        for it in range(1, total_iterations + 1):
            self._agent._current_iteration = it
            self._agent._total_iterations = total_iterations

            self._agent.train_step()

            # ====== 终端打印训练指标 ======
            if it % self._print_freq == 0 or it == total_iterations:
                ti = getattr(self._agent, "_train_info", {})
                elapsed = time.time() - t0
                fps = int(self._agent.global_step / max(elapsed, 1e-6))

                # 提取关键指标
                def _g(key, default="N/A"):
                    v = ti.get(key, None)
                    if v is None:
                        return default
                    return f"{float(v.item()) if hasattr(v, 'item') else float(v):.4f}"

                ep_rew = _g("ep/rew_mean")
                ep_len = _g("ep/len_mean")
                pi_loss = _g("actor_loss")
                v_loss = _g("v_loss")
                kl = _g("approx_kl")
                cur_lr = _g("learning_rate")

                print(
                    f"[Iter {it}/{total_iterations}]"
                    f"  step={self._agent.global_step}"
                    f"  ep_rew={ep_rew}"
                    f"  ep_len={ep_len}"
                    f"  pi_loss={pi_loss}"
                    f"  v_loss={v_loss}"
                    f"  kl={kl}"
                    f"  lr={cur_lr}"
                    f"  fps={fps}"
                    f"  time={elapsed:.0f}s"
                )
                self._agent.print_train_info()

            if it % self._summary_freq == 0 or it == total_iterations:
                self._agent.write_train_summary(writer)
            if it % self._eval_freq == 0 or it == total_iterations:
                if self._evaluator is not None:
                    try:
                        self._evaluator.eval(self._agent.global_step)
                    except Exception:
                        logging.exception("Eval failed with exception:")
            if it % self._save_freq == 0 or it == total_iterations:
                self._agent.save(agent_ckpt_dir)

        writer.close()
        logging.info("Training finished, time cost %.4gs.", time.time() - t0)

    def train(self) -> None:
        self._train_agent()