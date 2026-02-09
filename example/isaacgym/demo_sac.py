# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2025 Xi'an Jiaotong University, Jiayi Xie

import sys
import datetime
sys.path.append('../../')
import isaacgym
import torch
import logging
# from d2c.trainers import Trainer
from d2c.models import make_agent
from d2c.trainers import OffPolicyTrainer
from d2c.envs import benchmark_env, LeaEnv
from d2c.data import Data
from d2c.evaluators import isaacgym_eval
from d2c.utils.replaybuffer import ReplayBuffer
from example.isaacgym.config import make_config

# If you live in China mainland and want to use wandb, you can use this wandb mirror to solve the problem of network.
import os
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"

logging.basicConfig(level=logging.INFO)
nowTime = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')

import numpy as np
import random
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    env_name: str = 'pointfoot'
    # data_name: str = 'pointfoot_flat_dataset'
    data_name: str = 'halfcheetah_medium_replay-v2'
    task: str = 'pointfoot_flat'
    resume: bool = False
    # experiment_name: str = None
    # run_name: str = None
    # load_run: str = None
    # checkpoint: int = None
    headless: bool = True
    # horovod: bool = False
    # rl_device: str = 'cuda:0'
    num_envs: int = 20
    # seed: int = None
    total_train_steps: int = 1000000
    n_eval_episodes_max_step: int = 50

def main(args: Args):
    seed = np.random.randint(0, 100) 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prefix = 'env.external.'
    command_args = {
        prefix + 'benchmark_name': 'isaacgym',
        prefix + 'data_source': 'real_world',
        prefix + 'env_name': 'pointfoot',
        # prefix + 'data_name': 'pointfoot_flat_dataset',
        prefix + 'data_name': 'halfcheetah_medium_replay-v2',
        prefix + 'state_normalize': False,
        prefix + 'score_normalize': False,
    }
    command_args.update({
        'model.model_name': 'sac',
        'train.data_loader_name': None,
        'train.device': device,
        'train.seed': seed,
        'train.total_train_steps': 1000000,
        'train.batch_size': 256,
        'train.agent_ckpt_name': '1211'
    })
    command_args.update({
        prefix + 'env_name': args.env_name,
        prefix + 'data_name': args.data_name,
        prefix + 'task': args.task,
        prefix + 'resume': args.resume,
        prefix + 'headless': args.headless,
        prefix + 'num_envs': args.num_envs,
        'eval.n_eval_episodes_max_step': args.n_eval_episodes_max_step,
        'train.total_train_steps': args.total_train_steps,
    })
    wandb = {
        'project': 'isaacgym_pointfoot',
        'name': command_args['env.external.benchmark_name']+'_'+command_args['env.external.task']+'_seed='+str(command_args['train.seed'])+'_'+nowTime,
        'reinit': False,
        'mode': 'online'
    }
    command_args.update({'train.wandb': wandb})

    config = make_config(command_args)
    # real_dataset = Data(config)
    # s_norm = dict(zip(['obs_shift', 'obs_scale'], real_dataset.state_shift_scale))
    # data = real_dataset.data

    s_norm = {'obs_shift': None, 'obs_scale': None}

    train_env = benchmark_env(config, **s_norm)
    print("\n-------------Env name: {}, task: {}, model_name: {}-------------".format(config.model_config.env.external.env_name, config.model_config.env.external.task, config.model_config.model.model_name))

    agent = make_agent(config=config, env=train_env)
    agent._empty_dataset = ReplayBuffer(
        state_dim=train_env._env.num_obs,
        action_dim=train_env._env.num_actions,
        max_size=config.model_config.train.model_buffer_size,
        device=config.model_config.train.device,
    )
    evaluator = isaacgym_eval(agent=agent, env=train_env, config=config)
    trainer = OffPolicyTrainer(agent=agent, train_data=agent._empty_dataset, config=config, env=train_env, evaluator=evaluator)
    trainer.train()


if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)
