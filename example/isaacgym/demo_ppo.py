import sys
import datetime
sys.path.append("../../")
import isaacgym 
import torch
import logging
from dataclasses import dataclass
import numpy as np
import tyro
import os
from d2c.models import make_agent
from d2c.trainers import IsaacGymOnPolicyTrainer
from d2c.envs import benchmark_env
from d2c.evaluators import isaacgym_eval
from example.isaacgym.config import make_config
import wandb

# register isaacgym tasks
from d2c.envs.external.isaacgym.envs import register_tasks
register_tasks()


# If you live in China mainland and want to use wandb, you can use this wandb mirror to solve the problem of network.
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"

logging.basicConfig(level=logging.INFO)
nowTime = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")


@dataclass
class Args:
    # env & task
    env_name: str = "pointfoot" 
    task: str = "pointfoot_flat"
    resume: bool = False
    headless: bool = True
    num_envs: int = 1024
    # training
    total_train_steps: int = 100000000
    seed: int = -1 
    agent_ckpt_name: str = "ppo_isaacgym"

    # eval
    n_eval_episodes_max_step: int = 1000

    # logging
    wandb_project: str = "isaacgym_pointfoot"


def main(args: Args):
    # seed
    seed = int(np.random.randint(0, 100)) if args.seed is None or args.seed < 0 else int(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prefix = "env.external."

    command_args = {
        prefix + "benchmark_name": "isaacgym",
        prefix + "data_source": "real_world",
        prefix + "env_name": args.env_name,
        prefix + "data_name": "",  
        prefix + "state_normalize": False,
        prefix + "score_normalize": False,
    }

    command_args.update({
        "model.model_name": "isaacgym_ppo",
        "train.data_loader_name": None,
        "train.device": device,
        "train.seed": seed,
        "train.total_train_steps": int(args.total_train_steps),
        "train.agent_ckpt_name": args.agent_ckpt_name,
        "model.isaacgym_ppo.hyper_params.num_envs": int(args.num_envs),
    })

    command_args.update({
        prefix + "task": args.task,
        prefix + "resume": bool(args.resume),
        prefix + "headless": bool(args.headless),
        prefix + "num_envs": int(args.num_envs),
        "eval.n_eval_episodes_max_step": int(args.n_eval_episodes_max_step),
    })

    wandb_cfg = {
        "project": args.wandb_project,
        "name": (
            f"{command_args['env.external.benchmark_name']}_"
            f"{command_args['env.external.task']}_"
            f"seed={command_args['train.seed']}_{nowTime}"
        ),
        "reinit": False,
        "mode": "online",
    }
    command_args.update({"train.wandb": wandb_cfg})

    config = make_config(command_args)

    s_norm = {"obs_shift": None, "obs_scale": None}
    train_env = benchmark_env(config, **s_norm)

    print(
        "\n-------------Env name: {}, task: {}, model_name: {}-------------".format(
            config.model_config.env.external.env_name,
            config.model_config.env.external.task,
            config.model_config.model.model_name,
        )
    )

    agent = make_agent(config=config, env=train_env, data=None)
    # print("agent cfg:", getattr(agent, "cfg", None))
    # print("actor:", agent.actor if hasattr(agent, "actor") else "no actor attr")
    # print(config.model_config.model)
    
    wandb_cfg = getattr(config.model_config.train, "wandb", None)
    if wandb_cfg is not None:
        wandb.init(**wandb_cfg)

        
    evaluator = isaacgym_eval(agent=agent, env=train_env, config=config)
    trainer = IsaacGymOnPolicyTrainer(agent=agent, train_data=None, config=config, env=train_env, evaluator=evaluator)
    trainer.train()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
