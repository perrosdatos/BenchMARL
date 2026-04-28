import copy
import torch
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite, UnboundedContinuous, Categorical, Bounded
from torchrl.envs import EnvBase

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from .lux_env import LuxTorchRLEnv
from .lux_with_sap import LuxSapTorchRLEnv

class LuxClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        return lambda: LuxTorchRLEnv(
            batch_size=num_envs,
            device=device,
            seed=seed,
            **config,
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"] * self.config.get("match_count", 5)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        return {"agents": [f"unit_{i}" for i in range(self.config.get("max_units", 16))]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        obs_spec = env.full_observation_spec_unbatched.clone()
        return Composite({
            "agents": Composite({
                "action_mask": obs_spec["agents", "action_mask"]
            }, shape=obs_spec["agents"].shape)
        }, shape=obs_spec.shape)

    def observation_spec(self, env: EnvBase) -> Composite:
        # BenchMARL uses observation_spec of the inner env
        return env.full_observation_spec_unbatched.clone()

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return env.full_observation_spec_unbatched.clone().empty() # Not strictly required

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched.clone()

    @staticmethod
    def render_callback(experiment, env: EnvBase, data) -> "torch.Tensor":
        try:
            return env.render(mode="rgb_array")
        except TypeError:
            return env.render()

    @staticmethod
    def log_info(batch) -> Dict[str, float]:
        to_log = {}
        # batch represents the episodic rollout TensorDict
        if "next" in batch.keys() and "info" in batch.get("next").keys():
            info_td = batch.get(("next", "info"))
            done = batch.get(("next", "done"))
            if done.any():
                if "agent_points" in info_td.keys() and "opponent_points" in info_td.keys():
                    agent_pts = info_td.get("agent_points")[done]
                    opp_pts = info_td.get("opponent_points")[done]
                    if agent_pts.numel() > 0:
                        to_log["collection/end_episode_agent_points"] = agent_pts.to(torch.float).mean().item()
                        to_log["collection/end_episode_opponent_points"] = opp_pts.to(torch.float).mean().item()
                        to_log["collection/end_episode_win_margin"] = (agent_pts - opp_pts).to(torch.float).mean().item()
                
                # Check for any dynamic reward components injected with "rc_"
                for key in info_td.keys():
                    if key.startswith("rc_"):
                        comp_name = key[3:] # Remove "rc_"
                        comp_vals = info_td.get(key)[done]
                        if comp_vals.numel() > 0:
                            to_log[f"reward_components/{comp_name}"] = comp_vals.to(torch.float).mean().item()
                            
        return to_log

    @staticmethod
    def env_name() -> str:
        return "lux"


class LuxTask(Task):
    """Enum for Lux tasks."""

    MATCH = None
    MATCH_V2 = None

    @staticmethod
    def associated_class():
        return LuxClass


class LuxSapClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        return lambda: LuxSapTorchRLEnv(
            batch_size=num_envs,
            device=device,
            seed=seed,
            **config,
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"] * self.config.get("match_count", 5)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        return {"agents": [f"unit_{i}" for i in range(self.config.get("max_units", 16))]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        obs_spec = env.full_observation_spec_unbatched.clone()
        return Composite({
            "agents": Composite({
                "action_mask": obs_spec["agents", "action_mask"]
            }, shape=obs_spec["agents"].shape)
        }, shape=obs_spec.shape)

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.full_observation_spec_unbatched.clone()

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return env.full_observation_spec_unbatched.clone().empty()

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched.clone()

    @staticmethod
    def render_callback(experiment, env: EnvBase, data) -> "torch.Tensor":
        try:
            return env.render(mode="rgb_array")
        except TypeError:
            return env.render()

    @staticmethod
    def log_info(batch) -> Dict[str, float]:
        to_log = {}
        if "next" in batch.keys() and "info" in batch.get("next").keys():
            info_td = batch.get(("next", "info"))
            done = batch.get(("next", "done"))
            if done.any():
                if "agent_points" in info_td.keys() and "opponent_points" in info_td.keys():
                    agent_pts = info_td.get("agent_points")[done]
                    opp_pts = info_td.get("opponent_points")[done]
                    if agent_pts.numel() > 0:
                        to_log["collection/end_episode_agent_points"] = agent_pts.to(torch.float).mean().item()
                        to_log["collection/end_episode_opponent_points"] = opp_pts.to(torch.float).mean().item()
                        to_log["collection/end_episode_win_margin"] = (agent_pts - opp_pts).to(torch.float).mean().item()
                
                for key in info_td.keys():
                    if key.startswith("rc_"):
                        comp_name = key[3:]
                        comp_vals = info_td.get(key)[done]
                        if comp_vals.numel() > 0:
                            to_log[f"reward_components/{comp_name}"] = comp_vals.to(torch.float).mean().item()
                            
        return to_log

    @staticmethod
    def env_name() -> str:
        return "luxsap"


class LuxSapTask(Task):
    """Enum for Lux Sap tasks."""

    COMBAT = None

    @staticmethod
    def associated_class():
        return LuxSapClass
