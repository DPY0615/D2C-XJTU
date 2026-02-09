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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
from datetime import datetime
from typing import Tuple
import numpy as np
import json
from shutil import copyfile
import ntpath
from typing import Any, Optional, Tuple
from d2c.envs.external.isaacgym.envs.vec_env import VecEnv
# from d2c.envs.external.isaacgym.algorithm.on_policy_runner import OnPolicyRunner

from d2c.envs.external.isaacgym import (
    ISAAC_GYM_ROOT_DIR,
    ISAAC_GYM_ENVS_DIR,
)
from .helpers import (
    get_args,
    update_cfg_from_args,
    class_to_dict,
    get_load_path,
    set_seed,
    parse_sim_params,
)

class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(
        self,
        name: str,
        task_class: VecEnv,
        env_cfg,
        train_cfg,
    ):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name):
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    # def save_cfgs(self, name):
    #     os.makedirs(self.log_dir)

    #     json_str = json.dumps(class_to_dict(self.env_cfgs[name]), indent=4)
    #     file = os.path.join(self.log_dir, "env_cfg.json")
    #     with open(file, "w") as json_file:
    #         json_file.write(json_str)

    #     json_str = json.dumps(class_to_dict(self.train_cfgs[name]), indent=4)
    #     file = os.path.join(self.log_dir, "train_cfg.json")
    #     with open(file, "w") as json_file:
    #         json_file.write(json_str)

    #     save_items = [
    #         os.path.join(ISAAC_GYM_ENVS_DIR + "/{}/".format(name) + "{}.py".format(name)),
    #         os.path.join(ISAAC_GYM_ENVS_DIR + "/{}/".format(name) + "{}_config.py".format(name))
    #     ]
    #     if save_items is not None:
    #         for save_item in save_items:
    #             base_file_name = ntpath.basename(save_item)
    #             copyfile(save_item, self.log_dir + "/" + base_file_name)
    
    def save_cfgs(self, name, log_dir=None):
        log_dir = self.log_dir if log_dir is None else log_dir
        if log_dir is None:
            return

        os.makedirs(log_dir, exist_ok=True)

        json_str = json.dumps(class_to_dict(self.env_cfgs[name]), indent=4)
        with open(os.path.join(log_dir, "env_cfg.json"), "w") as f:
            f.write(json_str)

        json_str = json.dumps(class_to_dict(self.train_cfgs[name]), indent=4)
        with open(os.path.join(log_dir, "train_cfg.json"), "w") as f:
            f.write(json_str)

        save_items = [
            os.path.join(ISAAC_GYM_ENVS_DIR, f"{name}", f"{name}.py"),
            os.path.join(ISAAC_GYM_ENVS_DIR, f"{name}", f"{name}_config.py"),
        ]
        for save_item in save_items:
            base = ntpath.basename(save_item)
            copyfile(save_item, os.path.join(log_dir, base))


    def make_env(self, name, args=None, env_cfg=None):
        """Creates an environment either from a registered name or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name'

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        # set_seed(env_cfg.seed)
        seed = getattr(env_cfg, "seed", None)
        if seed is not None:
            set_seed(seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless,
        )
        return env, env_cfg

    # def make_alg_runner(
    #     self, env, name=None, args=None, train_cfg=None, log_root="default"
    # ): # -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
    #     """Creates the training algorithm  either from a registered namme or from the provided config file.

    #     Args:
    #         env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
    #         name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
    #         args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
    #         train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
    #         log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example).
    #                                   Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_wheel_legged_gym>/logs/<experiment_name>.

    #     Raises:
    #         ValueError: Error if neither 'name' or 'train_cfg' are provided
    #         Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

    #     Returns:
    #         PPO: The created algorithm
    #         Dict: the corresponding config file
    #     """
    #     # if no args passed get command line arguments
    #     if args is None:
    #         args = get_args()
    #     # if config files are passed use them, otherwise load from the name
    #     if train_cfg is None:
    #         if name is None:
    #             raise ValueError("Either 'name' or 'train_cfg' must be not None")
    #         # load config files
    #         _, train_cfg = self.get_cfgs(name)
    #     else:
    #         if name is not None:
    #             print(f"'train_cfg' provided -> Ignoring 'name={name}'")
    #     # override cfg from args (if specified)
    #     _, train_cfg = update_cfg_from_args(None, train_cfg, args)

    #     if log_root == "default":
    #         log_root = os.path.join(
    #             ISAAC_GYM_ROOT_DIR,
    #             "example",
    #             "isaacgym",
    #             "logs",
    #             args.task,
    #             train_cfg.runner.experiment_name,
    #         )
    #         self.log_dir = os.path.join(
    #             log_root,
    #             datetime.now().strftime("%b%d_%H-%M-%S")
    #             + "_"
    #             + train_cfg.runner.run_name
    #             + args.exptid,
    #         )
    #     elif log_root is None:
    #         self.log_dir = None
    #     else:
    #         self.log_dir = os.path.join(
    #             log_root,
    #             datetime.now().strftime("%b%d_%H-%M-%S")
    #             + "_"
    #             + train_cfg.runner.run_name
    #             + args.exptid,
    #         )
    #     train_cfg.runner.exptid = args.exptid
    #     train_cfg_dict = class_to_dict(train_cfg)
    #     runner = OnPolicyRunner(
    #         env, train_cfg_dict, self.log_dir, device=args.rl_device
    #     )
    #     # save resume path before creating a new log_dir
    #     resume = train_cfg.runner.resume
    #     if resume:
    #         # load previously trained model
    #         resume_path = get_load_path(
    #             log_root,
    #             load_run=train_cfg.runner.load_run,
    #             checkpoint=train_cfg.runner.checkpoint,
    #         )
    #         print(f"Loading model from: {resume_path}")
    #         runner.load(resume_path)
    #     return runner, train_cfg
    
    # def make_alg_runner_cfgs(
    #     self, env, name=None, args=None, train_cfg=None, log_root="default"
    # ): # -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
    #     """Creates the training algorithm  either from a registered namme or from the provided config file.

    #     Args:
    #         env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
    #         name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
    #         args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
    #         train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
    #         log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example).
    #                                   Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_wheel_legged_gym>/logs/<experiment_name>.

    #     Raises:
    #         ValueError: Error if neither 'name' or 'train_cfg' are provided
    #         Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

    #     Returns:
    #         PPO: The created algorithm
    #         Dict: the corresponding config file
    #     """
    #     # if no args passed get command line arguments
    #     if args is None:
    #         args = get_args()
    #     # if config files are passed use them, otherwise load from the name
    #     if train_cfg is None:
    #         if name is None:
    #             raise ValueError("Either 'name' or 'train_cfg' must be not None")
    #         # load config files
    #         _, train_cfg = self.get_cfgs(name)
    #     else:
    #         if name is not None:
    #             print(f"'train_cfg' provided -> Ignoring 'name={name}'")
    #     # override cfg from args (if specified)
    #     _, train_cfg = update_cfg_from_args(None, train_cfg, args)

    #     if log_root == "default":
    #         log_root = os.path.join(
    #             ISAAC_GYM_ROOT_DIR,
    #             "example",
    #             "isaacgym",
    #             "logs",
    #             args.task,
    #             train_cfg.runner.experiment_name,
    #         )
    #         self.log_dir = os.path.join(
    #             log_root,
    #             datetime.now().strftime("%b%d_%H-%M-%S")
    #             + "_"
    #             + train_cfg.runner.run_name
    #             + args.exptid,
    #         )
    #     elif log_root is None:
    #         self.log_dir = None
    #     else:
    #         self.log_dir = os.path.join(
    #             log_root,
    #             datetime.now().strftime("%b%d_%H-%M-%S")
    #             + "_"
    #             + train_cfg.runner.run_name
    #             + args.exptid,
    #         )
    #     train_cfg.runner.exptid = args.exptid
    #     train_cfg_dict = class_to_dict(train_cfg)
    #     return train_cfg

def make_alg_runner_cfgs(
        self,
        env: Optional[Any] = None,
        name: Optional[str] = None,
        args: Optional[Any] = None,
        train_cfg: Optional[Any] = None,
        log_root: Optional[str] = "default",
    ) -> Tuple[Any, Optional[str], Optional[str]]:
        """Prepare training config and logging directory for an on-policy run.

        This function is a **config/logging helper** extracted from the original
        `make_alg_runner()` implementation. It does **NOT** create any algorithm/runner
        instance (e.g., `OnPolicyRunner`). It only:
          - resolves `train_cfg` (from registry if not provided),
          - overrides cfg fields from IsaacGym CLI args,
          - computes `log_root` / `log_dir` and stores `self.log_dir`,
          - sets `train_cfg.runner.exptid`.

        Note:
            The `env` argument is kept for backward compatibility with older call sites,
            but it is unused. You may safely stop passing it.

        Args:
            env (isaacgym.VecTaskPython, optional):
                The environment to train. **Unused** here. Kept only for API compatibility.
            name (str, optional):
                Name of a registered task. Required if `train_cfg` is None.
            args (Args, optional):
                Isaac Gym command-line arguments. If None, `get_args()` will be called.
            train_cfg (Any, optional):
                Training config object. If None, it will be loaded from the registry by `name`.
            log_root (str | None, optional):
                Logging root directory.
                - "default": use `<ISAAC_GYM_ROOT_DIR>/example/isaacgym/logs/<task>/<experiment_name>`
                - None: disable logging (log_dir will be None)
                - other string: use as custom root dir.
                Logs (if enabled) will be saved in `<log_root>/<date_time>_<run_name><exptid>`.

        Raises:
            ValueError:
                If both `name` and `train_cfg` are None.

        Returns:
            train_cfg (Any):
                The resolved training config after applying CLI overrides.
            log_dir (str | None):
                The run directory for this experiment (timestamp + run_name + exptid),
                or None if logging disabled.
            log_root (str | None):
                The root directory actually used, or None if logging disabled.
        """
        if args is None:
            args = get_args()

        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")

        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root == "default":
            log_root = os.path.join(
                ISAAC_GYM_ROOT_DIR,
                "example",
                "isaacgym",
                "logs",
                args.task,
                train_cfg.runner.experiment_name,
            )

        if log_root is None:
            self.log_dir = None
        else:
            self.log_dir = os.path.join(
                log_root,
                datetime.now().strftime("%b%d_%H-%M-%S")
                + "_"
                + train_cfg.runner.run_name
                + args.exptid,
            )

        train_cfg.runner.exptid = args.exptid
        return train_cfg, self.log_dir, log_root

# make global task registry
task_registry = TaskRegistry()
