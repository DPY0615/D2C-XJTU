from d2c.utils.config import ConfigBuilder
from d2c.utils.utils import Flags
import importlib
import numpy as np

class IsaacGymConfigBuilder(ConfigBuilder):
    def _get_env_info(self) -> Flags:
        if self._exp_type == 'benchmark':
            try:
                self._env_ext = self._model_cfg.env.external
                # sys.path.append('../../example/benchmark/')
                import_path = '.'.join(('example.isaacgym.data', self._env_ext.benchmark_name, self._env_ext.data_source))
                module = importlib.import_module(import_path)
                domain = self._env_ext.env_name.split('-')[0]
                env_info = Flags(
                    norm_min=getattr(module, (domain + '_random_score').upper()),
                    norm_max=getattr(module, (domain + '_expert_score').upper()),
                    state_info=getattr(module, (domain + '_state').upper()),
                    action_info=getattr(module, (domain + '_action').upper()),
                )
            except:
                benchmark_name = self._env_ext.benchmark_name
                data_source = self._env_ext.data_source
                env_name = self._env_ext.env_name
                kwargs = dict()
                if 'combined_challenge' in self._env_ext:
                    kwargs.update({'combined_challenge': self._env_ext.combined_challenge})
                state_info, action_info = self._get_env_space(
                    benchmark_name,
                    data_source,
                    env_name,
                    **kwargs,
                )
                env_info = Flags(
                    norm_min=None,
                    norm_max=None,
                    state_info=state_info,
                    action_info=action_info,
                )
        elif self._exp_type == 'application':
            if self._app_cfg.state_dim is None:
                state_dim = len(self._app_cfg.state_indices)
            else:
                state_dim = self._app_cfg.state_dim
            if self._app_cfg.state_scaler == 'min_max':
                state_min = 0.0
                state_max = 1.0
            else:
                state_min, state_max = -np.inf, np.inf
            if self._app_cfg.action_dim is None:
                action_dim = len(self._app_cfg.action_indices)
            else:
                action_dim = self._app_cfg.action_dim
            if self._app_cfg.action_scaler == 'min_max':
                action_min = 0.0
                action_max = 1.0
            else:
                action_min, action_max = -np.inf, np.inf
            env_info = Flags(
                norm_min=None,
                norm_max=None,
                state_info=(state_dim, state_min, state_max),
                action_info=(action_dim, action_min, action_max),
            )
        else:
            raise ValueError(f'The value of the parameter experiment_type is wrong!')
        return env_info