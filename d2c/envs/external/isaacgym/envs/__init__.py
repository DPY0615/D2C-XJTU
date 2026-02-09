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
# Copyright (c) 2025 Xi'an Jiaotong University, Jiayi Xie

from d2c.envs.external.isaacgym import (
    ISAAC_GYM_ROOT_DIR,
    ISAAC_GYM_ENVS_DIR,
)

import os, sys
# from d2c.envs.external.isaacgym.utils import task_registry

# robot_type = os.getenv("ROBOT_TYPE")

def register_tasks():
    from d2c.envs.external.isaacgym.utils import task_registry
    
    robot_type = os.getenv("ROBOT_TYPE")
    
    if not robot_type:
        print("\033[1m\033[31mError: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.\033[0m")
        sys.exit(1)

    if robot_type.startswith("PF"):
        if robot_type in ["PF_TRON1A", "PF_P441A", "PF_P441B", "PF_P441C", "PF_P441C2"]:
            from d2c.envs.external.isaacgym.envs.pointfoot_flat.pointfoot_flat import BipedPF
            from d2c.envs.external.isaacgym.envs.pointfoot_flat.pointfoot_flat_config import BipedCfgPF, BipedCfgPPOPF
            # from d2c.envs.external.isaacgym.envs.pointfoot_rough.pointfoot_rough import BipedPFRough
            # from d2c.envs.external.isaacgym.envs.pointfoot_rough.pointfoot_rough_config import BipedCfgPFRough, BipedCfgPPOPFRough
            task_registry.register("pointfoot_flat", BipedPF, BipedCfgPF(), BipedCfgPPOPF())
            # task_registry.register("pointfoot_rough", BipedPFRough, BipedCfgPFRough(), BipedCfgPPOPFRough())
    #     else:
    #         print("\033[1m\033[31mError: Input ROBOT_TYPE={}".format(robot_type), 
    #         "is not among valid robot types PF_TRON1A, PF_P441A, PF_P441B, PF_P441C, PF_P441C2.\033[0m")
    #         sys.exit(1)

    # elif robot_type == "SF_TRON1A":
    #     from d2c.envs.external.isaacgym.envs.solefoot_flat.solefoot_flat import BipedSF
    #     from d2c.envs.external.isaacgym.envs.solefoot_flat.solefoot_flat_config import BipedCfgSF, BipedCfgPPOSF
    #     from d2c.envs.external.isaacgym.envs.solefoot_rough.solefoot_rough import BipedSFRough
    #     from d2c.envs.external.isaacgym.envs.solefoot_rough.solefoot_rough_config import BipedCfgSFRough, BipedCfgPPOSFRough
    #     task_registry.register("pointfoot_flat", BipedSF, BipedCfgSF(), BipedCfgPPOSF())
    #     task_registry.register("pointfoot_rough", BipedSFRough, BipedCfgSFRough(), BipedCfgPPOSFRough())

    # elif robot_type == "WF_TRON1A":
    #     from d2c.envs.external.isaacgym.envs.wheelfoot_flat.wheelfoot_flat import BipedWF
    #     from d2c.envs.external.isaacgym.envs.wheelfoot_flat.wheelfoot_flat_config import BipedCfgWF, BipedCfgPPOWF
    #     from d2c.envs.external.isaacgym.envs.wheelfoot_rough.wheelfoot_rough import BipedWFRough
    #     from d2c.envs.external.isaacgym.envs.wheelfoot_rough.wheelfoot_rough_config import BipedCfgWFRough, BipedCfgPPOWFRough
    #     task_registry.register("pointfoot_flat", BipedWF, BipedCfgWF(), BipedCfgPPOWF())
    #     task_registry.register("pointfoot_rough", BipedWFRough, BipedCfgWFRough(), BipedCfgPPOWFRough())
    elif robot_type == "a1":
        from d2c.envs.external.isaacgym.envs.base.legged_robot import LeggedRobot
        from d2c.envs.external.isaacgym.envs.a1.a1_config import A1Cfg, A1CfgPPO
        task_registry.register("a1", LeggedRobot, A1Cfg(), A1CfgPPO())
        return

    else:
        print("\033[1m\033[31mError: Input ROBOT_TYPE={}".format(robot_type), 
            "is not among valid robot types a1, PF_P441A, PF_P441B, PF_P441C, PF_P441C2, PF_TRON1A, WF_TRON1A and SF_TRON1A.\033[0m")
        sys.exit(1)
