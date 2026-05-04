import jax
import jax.numpy as jnp
import numpy as np
import torch
from typing import Optional, Dict, Any, List

from torchrl.envs import EnvBase
from torchrl.data import (
    Composite, Bounded, UnboundedContinuous,
    Categorical
)
from tensordict import TensorDict

from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams
from pygame import surfarray

from benchmarl.environments.lux.reward_exploration import compute_shaped_rewards


class LuxTorchRLEnv(EnvBase):
    def __init__(self, batch_size: int, device="cpu", seed: Optional[int] = None, **kwargs):
        if not isinstance(batch_size, int):
            batch_size = batch_size[0] if len(batch_size) > 0 else 1
        batch_size = max(1, int(batch_size))
        super().__init__(device=device, batch_size=torch.Size([batch_size]))
        
        self.max_steps = kwargs.get("max_steps", 150)
        self.match_count = kwargs.get("match_count", 5)
        self.num_teams = kwargs.get("num_teams", 2)
        self.max_units = kwargs.get("max_units", 16)
        self.map_width = kwargs.get("map_width", 24)
        self.map_height = kwargs.get("map_height", 24)
        self.reward_scaling = kwargs.get("reward_scaling", 0.1)
        self.reward_version = kwargs.get("reward_version", "v1")
        
        self.env_params = EnvParams(
            max_steps_in_match=self.max_steps - 1, 
            match_count_per_episode=self.match_count
        )
        self.raw_env = LuxAIS3Env(fixed_env_params=self.env_params)
        
        # JAX specific batch maps
        self._batch_reset = jax.jit(jax.vmap(self.raw_env.reset, in_axes=(0, None)))
        self._batch_step = jax.jit(jax.vmap(self.raw_env.step, in_axes=(0, 0, 0, None)))
        
        self._base_seed = seed if seed is not None else 42
        self.rng = jax.random.PRNGKey(self._base_seed)
        
        # This will hold current team for EACH environment in the batch (batch_size,)
        self.team_ids = None

        self._make_specs()
        
    def _make_specs(self):
        # 1. Action specification
        # MAPPO requires: {"agents": {"action": Discrete(5) for each agent}}
        agents_action_spec = Composite({
            "action": Categorical(
                n=5,
                shape=torch.Size([*self.batch_size, self.max_units]),
                device=self.device
            )
        }, shape=torch.Size([*self.batch_size, self.max_units]))
        
        self.action_spec = Composite({"agents": agents_action_spec}, shape=self.batch_size)
        
        # 2. Observation specification
        # 9 channels: 8 from original + 1 for active unit
        agents_obs_spec = Composite({
            "observation": Bounded(
                low=-1.0, high=1.0, # Approximate bounds
                shape=torch.Size([*self.batch_size, self.max_units, 16, self.map_width, self.map_height]),
                dtype=torch.float32,
                device=self.device
            ),
            "action_mask": Bounded(
                low=0, high=1,
                shape=torch.Size([*self.batch_size, self.max_units, 5]),
                dtype=torch.bool,
                device=self.device
            )
        }, shape=torch.Size([*self.batch_size, self.max_units]))
        self.rc_keys = [
            "fog_discovery", "novelty_bonus", "dispersion_bonus", 
            "relic_proximity", "relic_discovery", "energy_gain", 
            "collision_penalty", "stagnation_penalty", "local_point_generation",
            "relic_farming", "overcrowding_penalty", "total_reward"
        ]
        info_dict_spec = {
            "agent_points": UnboundedContinuous(shape=torch.Size([*self.batch_size, 1]), device=self.device),
            "opponent_points": UnboundedContinuous(shape=torch.Size([*self.batch_size, 1]), device=self.device)
        }
        for k in self.rc_keys:
            info_dict_spec[f"rc_{k}"] = UnboundedContinuous(shape=torch.Size([*self.batch_size, 1]), device=self.device)
            
        self.observation_spec = Composite({
            "agents": agents_obs_spec,
            "info": Composite(info_dict_spec, shape=self.batch_size)
        }, shape=self.batch_size)
        
        self.reward_spec = Composite({
            "agents": Composite({
                "reward": UnboundedContinuous(
                    shape=torch.Size([*self.batch_size, self.max_units, 1]),
                    device=self.device
                )
            }, shape=torch.Size([*self.batch_size, self.max_units]))
        }, shape=self.batch_size)
        
        # 4. Done specification
        done_spec = Bounded(
            low=0, high=1, shape=torch.Size([*self.batch_size, 1]), dtype=torch.bool, device=self.device
        )
        agents_done_spec = Composite({
             "done": Bounded(low=0, high=1, shape=torch.Size([*self.batch_size, self.max_units, 1]), dtype=torch.bool, device=self.device),
             "terminated": Bounded(low=0, high=1, shape=torch.Size([*self.batch_size, self.max_units, 1]), dtype=torch.bool, device=self.device),
             "truncated": Bounded(low=0, high=1, shape=torch.Size([*self.batch_size, self.max_units, 1]), dtype=torch.bool, device=self.device),
        }, shape=torch.Size([*self.batch_size, self.max_units]))
        
        self.done_spec = Composite({
            "done": done_spec,
            "terminated": done_spec.clone(),
            "truncated": done_spec.clone(),
            "agents": agents_done_spec
        }, shape=self.batch_size)
        
    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._base_seed = seed
            self.rng = jax.random.PRNGKey(seed)

    def _get_v(self, o, k):
        try:
            return getattr(o, k)
        except:
            if isinstance(o, dict):
                return o.get(k)
            return None

    def _build_pseudo_obs(self, jax_obs, player_id: int, b: int):
        p_key = f"player_{player_id}"
        p_obs = jax_obs[p_key]
        _v = self._get_v
        return {
            "units_mask": np.asarray(_v(p_obs, "units_mask"))[b].tolist(),
            "units": {
                "position": np.asarray(_v(_v(p_obs, "units"), "position"))[b].tolist(),
                "energy": np.asarray(_v(_v(p_obs, "units"), "energy"))[b].tolist()
            },
            "relic_nodes": np.asarray(_v(p_obs, "relic_nodes"))[b].tolist(),
            "relic_nodes_mask": np.asarray(_v(p_obs, "relic_nodes_mask"))[b].tolist(),
            "team_points": np.asarray(_v(p_obs, "team_points"))[b].tolist()
        }

    def _build_spatial_observation(self, jax_obs, team_ids: np.ndarray, units_pos: np.ndarray) -> np.ndarray:
        # We compute the exact same spatial grid as SB3, then slice for each agent
        # team_ids: (B,)
        b_size = self.batch_size[0]
        # (B, 8, W, H)
        grid = np.zeros((b_size, 8, self.map_width, self.map_height), dtype=np.float32)
        b_idx = np.arange(b_size)
        
        # Map states are globally shared but Fog of War is player-specific.
        # We must compose the environment fully symmetrically to respect JAX Multi-Agent perspectives.
        
        # Simplified tile map extraction from global map features
        mf_p0 = self._get_v(jax_obs["player_0"], "map_features")
        mf_p1 = self._get_v(jax_obs["player_1"], "map_features")
        
        tt0 = np.asarray(self._get_v(mf_p0, "tile_type"), np.int32)
        tt1 = np.asarray(self._get_v(mf_p1, "tile_type"), np.int32)
        tile_type = np.where((team_ids == 0)[:, None, None], tt0, tt1)
        
        grid[:, 4, :, :] = (tile_type == 1).astype(np.float32) # Nebula
        grid[:, 5, :, :] = (tile_type == 2).astype(np.float32) # Asteroid
        
        rn0 = np.asarray(self._get_v(jax_obs["player_0"], "relic_nodes"), np.int32)
        rmask0 = np.asarray(self._get_v(jax_obs["player_0"], "relic_nodes_mask"), bool)
        rn1 = np.asarray(self._get_v(jax_obs["player_1"], "relic_nodes"), np.int32)
        rmask1 = np.asarray(self._get_v(jax_obs["player_1"], "relic_nodes_mask"), bool)
        
        for b in range(b_size):
            p0_valid = rmask0[b]
            for pos in rn0[b][p0_valid]:
                if (team_ids == 0)[b]:
                    grid[b, 7, pos[0], pos[1]] = 1.0
            p1_valid = rmask1[b]
            for pos in rn1[b][p1_valid]:
                if (team_ids == 1)[b]:
                    grid[b, 7, pos[0], pos[1]] = 1.0

        # Sensors & Energy global maps
        sm0 = np.asarray(self._get_v(jax_obs["player_0"], "sensor_mask"), bool)
        sm1 = np.asarray(self._get_v(jax_obs["player_1"], "sensor_mask"), bool)
        p_sensor = np.where((team_ids == 0)[:, None, None], sm0, sm1)

        
        me0 = np.asarray(self._get_v(mf_p0, "energy"), np.float32)
        me1 = np.asarray(self._get_v(mf_p1, "energy"), np.float32)
        grid[:, 3, :, :] = np.where((team_ids == 0)[:, None, None], me0, me1) / 400.0
        
        grid[:, 6, :, :] = p_sensor.astype(np.float32)

        # Units representation absolute indices
        u0_p0 = self._get_v(jax_obs["player_0"], "units")
        u0_p1 = self._get_v(jax_obs["player_1"], "units")
        pos0 = np.asarray(self._get_v(u0_p0, "position"), np.int32)
        pos1 = np.asarray(self._get_v(u0_p1, "position"), np.int32)
        pos = np.where((team_ids == 0)[:, None, None, None], pos0, pos1)
        
        en0 = np.asarray(self._get_v(u0_p0, "energy"), np.float32)
        en1 = np.asarray(self._get_v(u0_p1, "energy"), np.float32)
        en = np.where((team_ids == 0)[:, None, None] if en0.ndim == 3 else (team_ids == 0)[:, None, None, None], en0, en1)
        
        m0 = np.asarray(self._get_v(jax_obs["player_0"], "units_mask"), bool)
        m1 = np.asarray(self._get_v(jax_obs["player_1"], "units_mask"), bool)
        m = np.where((team_ids == 0)[:, None, None], m0, m1)
        
        # Flatten for vectorized grid population
        # JAX arrays are absolute: r_team matches team_ids
        for r_team in [0, 1]:
            for u in range(self.max_units):
                valid = m[:, r_team, u]
                x_vals = pos[:, r_team, u, 0]
                y_vals = pos[:, r_team, u, 1]
                e_vals = en[:, r_team, u] if en.ndim == 3 else en[:, r_team, u, 0]
                
                my_mask = valid & (team_ids == r_team)
                op_mask = valid & (team_ids != r_team)
                
                grid[my_mask, 0, x_vals[my_mask], y_vals[my_mask]] = 1.0
                grid[op_mask, 1, x_vals[op_mask], y_vals[op_mask]] = 1.0
                grid[my_mask, 2, x_vals[my_mask], y_vals[my_mask]] = e_vals[my_mask] / 400.0

        # Now we create per-agent observation (B, 16, 16, 24, 24)
        agent_obs = np.zeros((b_size, self.max_units, 16, self.map_width, self.map_height), dtype=np.float32)
        
        steps_val = np.asarray(self._get_v(jax_obs["player_0"], "steps"))
        p0_pts = self._get_v(jax_obs["player_0"], "team_points")
        
        # Inject Memory Decay Tracking (Update last seen times for all tiles seen in current step)
        if hasattr(self, "last_seen_map"):
            step_t = steps_val[:, None, None] # Broadcast B -> (B, 1, 1)
            self.last_seen_map[:, 0] = np.where(sm0, step_t.astype(np.float32), self.last_seen_map[:, 0])
            self.last_seen_map[:, 1] = np.where(sm1, step_t.astype(np.float32), self.last_seen_map[:, 1])
            
        if hasattr(self, "known_relic_decay_map"):
            step_t = steps_val[:, None, None]
            for b in range(b_size):
                p0_valid = rmask0[b]
                for p in rn0[b][p0_valid]:
                    self.known_relic_decay_map[b, 0, p[0], p[1]] = step_t[b, 0, 0]
                p1_valid = rmask1[b]
                for p in rn1[b][p1_valid]:
                    self.known_relic_decay_map[b, 1, p[0], p[1]] = step_t[b, 0, 0]
        
        for u in range(self.max_units):
            agent_obs[:, u, :8, :, :] = grid
            
            for b in range(b_size):
                t = team_ids[b]
                # Ch 10: Timeline (Urgency)
                curr_step = steps_val[b]
                agent_obs[b, u, 10, :, :] = curr_step / float(self.max_steps * self.match_count)
                # Ch 11: Score Differential
                m_pts = float(np.asarray(p0_pts)[b, t] if p0_pts is not None else 0.0)
                e_pts = float(np.asarray(p0_pts)[b, 1-t] if p0_pts is not None else 0.0)
                agent_obs[b, u, 11, :, :] = np.clip((m_pts - e_pts) / 50.0, -1.0, 1.0)
                # Ch 12: Memory Decay Channel (Stigmergic Exploration)
                if hasattr(self, "last_seen_map"):
                    agent_obs[b, u, 12, :, :] = self.last_seen_map[b, t] / max(1.0, float(curr_step))
                # Ch 14: Relic Memory Decay Channel
                if hasattr(self, "known_relic_decay_map"):
                    agent_obs[b, u, 14, :, :] = self.known_relic_decay_map[b, t] / max(1.0, float(curr_step))
                # Ch 15: Team Point Deltas (Uniform Map Scalar Projection)
                if hasattr(self, "last_point_delta"):
                    agent_obs[b, u, 15, :, :] = np.clip(self.last_point_delta[b] / 5.0, 0.0, 1.0)
                
            for t in [0, 1]:
                 b_mask = (team_ids == t) & m[:, t, u]
                 
                 # Ch 13: Agent-level Trajectory Decay Channel (Agent Stigmergy)
                 if hasattr(self, "agent_trajectory_map"):
                     valid_b = np.where(b_mask)[0]
                     curr_steps_valid = steps_val[valid_b]
                     curr_x = pos[valid_b, t, u, 0]
                     curr_y = pos[valid_b, t, u, 1]
                     # Mark current step in trajectory
                     self.agent_trajectory_map[valid_b, t, u, curr_x, curr_y] = curr_steps_valid.astype(np.float32)
                     
                     # Extract for Channel 14
                     active_b = np.where(team_ids == t)[0]
                     curr_steps_t = steps_val[active_b]
                     div_factor = np.maximum(1.0, curr_steps_t)[:, None, None]
                     agent_obs[active_b, u, 13, :, :] = self.agent_trajectory_map[active_b, t, u, :, :] / div_factor
                     
                 # Ch 8: Self Indicator
                 agent_obs[b_mask, u, 8, pos[b_mask, t, u, 0], pos[b_mask, t, u, 1]] = 1.0
                 # Ch 9: Ghost Coordinate Tracking
                 if hasattr(self, "last_unit_pos"):
                     lx = self.last_unit_pos[b_mask, t, u, 0]
                     ly = self.last_unit_pos[b_mask, t, u, 1]
                     valid_mask = (lx >= 0) & (ly >= 0) & (lx < self.map_width) & (ly < self.map_height)
                     valid_b2 = np.where(b_mask)[0][valid_mask]
                     agent_obs[valid_b2, u, 9, lx[valid_mask], ly[valid_mask]] = 1.0
                 
        return agent_obs

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        if tensordict is not None and "_reset" in tensordict.keys():
            reset_mask = tensordict.get("_reset")
            if reset_mask.ndim > 1:
                reset_mask = reset_mask.squeeze(-1)
        else:
            reset_mask = torch.ones(self.batch_size[0], dtype=torch.bool, device=self.device)

        if not hasattr(self, "env_state"):
            self.rng, _rng = jax.random.split(self.rng)
            rng_batch = jax.random.split(_rng, self.batch_size[0])
            self.jax_obs, self.env_state = self._batch_reset(rng_batch, self.env_params)
            
            # Initialize random team assignment
            self.team_ids = torch.randint(0, 2, (self.batch_size[0],), device=self.device)
            # Mapping state history
            self.prev_points = np.zeros(self.batch_size[0], dtype=np.float32)
            self.prev_energy = np.zeros((self.batch_size[0], self.max_units), dtype=np.float32)
            self.last_seen_map = np.full((self.batch_size[0], 2, self.map_width, self.map_height), -50.0, dtype=np.float32)
            self.prev_visible_count = np.zeros(self.batch_size[0], dtype=np.int32)
            self.known_relic_mask = np.zeros((self.batch_size[0], self.max_units * 3), dtype=bool) 
            self.known_relic_pos = np.zeros((self.batch_size[0], self.max_units * 3, 2), dtype=np.int32)
            self.spawn_pos = np.zeros((self.batch_size[0], 2), dtype=np.int32)
            self.last_unit_pos = np.full((self.batch_size[0], 2, self.max_units, 2), -1, dtype=np.int32)
            self.agent_trajectory_map = np.full((self.batch_size[0], 2, self.max_units, self.map_width, self.map_height), -50.0, dtype=np.float32)
            self.known_relic_decay_map = np.zeros((self.batch_size[0], 2, self.map_width, self.map_height), dtype=np.float32)
            self.last_point_delta = np.zeros(self.batch_size[0], dtype=np.float32)

        if getattr(self, "reward_version", "v1") == "v2":
            if not hasattr(self, "opp_agents"):
                self.opp_agents = [None] * self.batch_size[0]
                self.rulebased_agent_class = None
                
                import sys
                import os
                moth_dir = "/home/carlos/Documents/github/msc_ai_thesis_marl_lux"
                if moth_dir not in sys.path:
                    sys.path.append(os.path.abspath(moth_dir))
                try:
                    from agent import Agent
                    self.rulebased_agent_class = Agent
                except ImportError:
                    pass



        b_size = self.batch_size[0]
        reset_indices = reset_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        
        if len(reset_indices) > 0:
            self.rng, reset_key = jax.random.split(self.rng)
            new_reset_obs, new_reset_state = self._batch_reset(jax.random.split(reset_key, len(reset_indices)), self.env_params)
            
            self.env_state = jax.tree_util.tree_map(
                lambda x, y: x.at[reset_indices].set(y),
                self.env_state, new_reset_state
            )
            # Re-scramble team IDs for resetting environments!
            self.team_ids[reset_indices] = torch.randint(0, 2, (len(reset_indices),), device=self.device)

            if getattr(self, "rulebased_agent_class", None) is not None:
                env_cfg_pseudo = {"max_units": self.max_units, "map_width": self.map_width, "map_height": self.map_height}
                rm_np = reset_mask.cpu().numpy()
                team_ids_np = self.team_ids.cpu().numpy()
                for b in range(self.batch_size[0]):
                    if rm_np[b] or self.opp_agents[b] is None:
                        opp_player_str = "player_1" if team_ids_np[b] == 0 else "player_0"
                        self.opp_agents[b] = self.rulebased_agent_class(opp_player_str, env_cfg_pseudo)
            # Update jax_obs manually 
            # jax.tree_util.tree_map is a bit annoying with dicts representing obs, so we just do step(actions) or rebuild it.
            # For simplicity, we just use the new_reset_obs and replace slices.
            
            # Since the JAX env obs is nested dict of arrays, we can do:
            def _replace_slice(dst, src, ids):
                return jax.tree_util.tree_map(lambda x, y: x.at[ids].set(y), dst, src)
            self.jax_obs = _replace_slice(self.jax_obs, new_reset_obs, reset_indices)

            # Reset state accumulators
            t_ids = self.team_ids.cpu().numpy()
            for idx, b_idx in enumerate(reset_indices):
                t = t_ids[b_idx]
                p0_dict = self._get_v(new_reset_obs["player_0"], "team_points")
                self.prev_points[b_idx] = float(np.asarray(p0_dict)[idx, t] if p0_dict is not None else 0.0)
                
                en = self._get_v(self._get_v(new_reset_obs["player_0"], "units"), "energy")
                if en is not None:
                    en_np = np.asarray(en)
                    self.prev_energy[b_idx] = en_np[idx, t, :] if en_np.ndim == 3 else en_np[idx, t, :, 0]
                else:
                    self.prev_energy[b_idx] = 0.0
                
                tile_dict = self._get_v(self._get_v(new_reset_obs["player_0"], "map_features"), "tile_type")
                if tile_dict is not None:
                    tile_np = np.asarray(tile_dict)
                    self.prev_visible_count[b_idx] = int(np.sum(tile_np[idx] != -1))
                else:
                    self.prev_visible_count[b_idx] = 0
                
                self.known_relic_mask[b_idx] = False
                self.known_relic_pos[b_idx] = 0
                if hasattr(self, "last_seen_map"):
                    self.last_seen_map[b_idx] = -50.0
                self.agent_trajectory_map[b_idx] = -50.0
                self.known_relic_decay_map[b_idx] = 0.0
                self.last_point_delta[b_idx] = 0.0
                
                if hasattr(self, "episode_reward_components"):
                    for k in self.episode_reward_components.keys():
                        self.episode_reward_components[k][b_idx] = 0.0
                
                pos_o = self._get_v(self._get_v(new_reset_obs["player_0"], "units"), "position")
                if pos_o is not None:
                    pos_np = np.asarray(pos_o)
                    self.spawn_pos[b_idx] = pos_np[b_idx, t, 0]
                else:
                    self.spawn_pos[b_idx] = 0
                self.last_unit_pos[b_idx] = -1

        # Build output TensorDict
        # team_ids: (B,)
        t_ids_np = self.team_ids.cpu().numpy()
        
        m = np.asarray(self._get_v(self.jax_obs["player_0"], "units_mask"), bool) # (B, 2, 16)
        
        # action_mask (B, 16, 5). 1 for valid options
        active_units = np.zeros((b_size, self.max_units), dtype=bool)
        for b in range(b_size):
            active_units[b, :] = m[b, t_ids_np[b], :]
            
        action_mask = np.zeros((b_size, self.max_units, 5), dtype=np.bool_)
        action_mask[active_units, :] = True
        action_mask[~active_units, 0] = True
        
        pos = np.asarray(self._get_v(self._get_v(self.jax_obs["player_0"], "units"), "position"), np.int32)
        
        obs_array = self._build_spatial_observation(self.jax_obs, t_ids_np, pos)
        
        agents_td = TensorDict({
            "observation": torch.tensor(obs_array, device=self.device),
            "action_mask": torch.tensor(action_mask, device=self.device)
        }, batch_size=torch.Size([b_size, self.max_units]))
        
        info_dict_reset = {
            "agent_points": torch.zeros((b_size, 1), dtype=torch.float32, device=self.device),
            "opponent_points": torch.zeros((b_size, 1), dtype=torch.float32, device=self.device)
        }
        if hasattr(self, "rc_keys"):
            for k in self.rc_keys:
                info_dict_reset[f"rc_{k}"] = torch.zeros((b_size, 1), dtype=torch.float32, device=self.device)
                
        td = TensorDict({
            "agents": agents_td,
            "done": torch.zeros((b_size, 1), dtype=torch.bool, device=self.device),
            "terminated": torch.zeros((b_size, 1), dtype=torch.bool, device=self.device),
            "truncated": torch.zeros((b_size, 1), dtype=torch.bool, device=self.device),
            "info": TensorDict(info_dict_reset, batch_size=torch.Size([b_size])),
        }, batch_size=torch.Size([b_size]))
        
        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        actions = tensordict.get(("agents", "action"))
        actions = actions.cpu().numpy()
        if actions.ndim == 3:
            actions = actions.squeeze(-1)
        
        t_ids_np = self.team_ids.cpu().numpy()
        b_size = self.batch_size[0]
        
        # Build action array (B, 16, 3), containing (action_id, 0, 0)
        zeros_sap = np.zeros((b_size, self.max_units, 2), dtype=np.int32)
        action_3d = np.concatenate([actions[..., None], zeros_sap], axis=-1)
        
        
        jax_actions_0 = np.zeros_like(action_3d)
        jax_actions_1 = np.zeros_like(action_3d)
        
        if getattr(self, "reward_version", "v1") == "v2" and getattr(self, "rulebased_agent_class", None) is not None:
            #print("[FATAL WARNING] THE NATIVE RULE-BASED AGENT IS RUNNING!")
            self.opp_actions = np.zeros((b_size, self.max_units), dtype=np.int32)
            steps_val = np.asarray(self._get_v(self.jax_obs["player_0"], "steps"))
            for b in range(b_size):
                opp_player_id = 1 if t_ids_np[b] == 0 else 0
                pseudo_obs = self._build_pseudo_obs(self.jax_obs, opp_player_id, b)
                act_col = self.opp_agents[b].act(int(steps_val[b]), pseudo_obs)
                self.opp_actions[b] = act_col[:, 0]
        
        for b in range(b_size):
            if t_ids_np[b] == 0:
                jax_actions_0[b] = action_3d[b]
            else:
                jax_actions_1[b] = action_3d[b]
                
            if hasattr(self, "opp_actions") and self.opp_actions is not None:
                # self.opp_actions is expected to be shape (B, 16)
                opp_3d = np.concatenate([self.opp_actions[b, ..., None], zeros_sap[b]], axis=-1)
                if t_ids_np[b] == 0:
                    jax_actions_1[b] = opp_3d
                else:
                    jax_actions_0[b] = opp_3d
                
        wrapped_actions = {
            "player_0": jnp.array(jax_actions_0, dtype=jnp.int32),
            "player_1": jnp.array(jax_actions_1, dtype=jnp.int32)
        }
        self.rng, _rng = jax.random.split(self.rng)
        rng_batch = jax.random.split(_rng, b_size)
        
        self.jax_obs, self.env_state, reward_dict, terminated, truncated, _ = self._batch_step(
             rng_batch, self.env_state, wrapped_actions, self.env_params
        )
        
        self.last_raw_reward = jax.tree_util.tree_map(lambda x: np.asarray(x), reward_dict)
        self.last_raw_terminated = jax.tree_util.tree_map(lambda x: np.asarray(x), terminated)
        self.last_raw_truncated = jax.tree_util.tree_map(lambda x: np.asarray(x), truncated)
        
        self.abosolute_raw_terminated_obj = terminated
        self.abosolute_raw_truncated_obj = truncated
        
        # Dones
        term_np = np.zeros(b_size, dtype=bool)
        trunc_np = np.zeros(b_size, dtype=bool)
        r_np = np.zeros((b_size, self.max_units), dtype=np.float32)
        
        t0 = np.asarray(terminated["player_0"])
        t1 = np.asarray(terminated["player_1"])
        tr0 = np.asarray(truncated["player_0"])
        tr1 = np.asarray(truncated["player_1"])
        r0 = np.asarray(reward_dict["player_0"])
        r1 = np.asarray(reward_dict["player_1"])
        
        steps = np.asarray(self._get_v(self.jax_obs["player_0"], "steps"))
        # Force TorchRL buffer truncation cleanly on logical episode boundary (150) instead of native horizon (450)
        step_limit_reached = (steps >= self.max_steps * self.match_count)
        
        # Prepare Delta Buffers for Shaped Rewards
        delta_pts = np.zeros(b_size, dtype=np.float32)
        delta_en = np.zeros((b_size, self.max_units), dtype=np.float32)
        delta_vis = np.zeros(b_size, dtype=np.float32)
        
        # Build current boolean masks and positions for computation
        m0 = np.asarray(self._get_v(self.jax_obs["player_0"], "units_mask"), bool)
        m1 = np.asarray(self._get_v(self.jax_obs["player_1"], "units_mask"), bool)
        m = np.where((t_ids_np == 0)[:, None, None], m0, m1)
        
        upos0 = np.asarray(self._get_v(self._get_v(self.jax_obs["player_0"], "units"), "position"), np.int32)
        upos1 = np.asarray(self._get_v(self._get_v(self.jax_obs["player_1"], "units"), "position"), np.int32)
        u_pos_np = np.where((t_ids_np == 0)[:, None, None, None], upos0, upos1)

        current_team_mask = np.zeros((b_size, self.max_units), dtype=bool)
        current_team_pos = np.zeros((b_size, self.max_units, 2), dtype=np.int32)

        for b in range(b_size):
             t = t_ids_np[b]
             term_np[b] = t0[b] if t == 0 else t1[b]
             trunc_np[b] = (tr0[b] if t == 0 else tr1[b]) | step_limit_reached[b]
             
             p_obs = self.jax_obs[f"player_{t}"]
             
             # Extract current mapped state natively for my agent!
             p0_dict = self._get_v(p_obs, "team_points")
             curr_pts = float(np.asarray(p0_dict)[b, t] if p0_dict is not None else 0.0)
             
             en = self._get_v(self._get_v(p_obs, "units"), "energy")
             if en is not None:
                en_np = np.asarray(en)
                curr_en = en_np[b, t, :] if en_np.ndim == 3 else en_np[b, t, :, 0]
             else:
                curr_en = np.zeros(self.max_units, dtype=np.float32)
                
             tile_dict = self._get_v(self._get_v(p_obs, "map_features"), "tile_type")
             if tile_dict is not None:
                tile_np = np.asarray(tile_dict)
                curr_vis = int(np.sum(tile_np[b] != -1))
             else:
                curr_vis = 0
                
             # calculate deltas
             delta_pts[b] = max(0.0, curr_pts - self.prev_points[b])
             delta_en[b, :] = np.maximum(0.0, curr_en - self.prev_energy[b])
             delta_vis[b] = max(0, curr_vis - self.prev_visible_count[b])
             
             # known relics tracking
             rmask_o = self._get_v(p_obs, "relic_nodes_mask")
             if rmask_o is not None:
                 rm_np = np.asarray(rmask_o)
                 newly_seen = rm_np[b] & ~self.known_relic_mask[b, :len(rm_np[b])]
                 self.known_relic_mask[b, :len(rm_np[b])] |= rm_np[b]
                 rpos_o = self._get_v(p_obs, "relic_nodes")
                 if rpos_o is not None:
                     rp_np = np.asarray(rpos_o)
                     self.known_relic_pos[b, :len(rp_np[b])][newly_seen] = rp_np[b][newly_seen]

             self.prev_points[b] = curr_pts
             self.prev_energy[b] = curr_en
             self.prev_visible_count[b] = curr_vis
             
             # Absolute indexing: MY units are at index `t` globally!
             current_team_mask[b] = m[b, t]
             if u_pos_np is not None:
                 current_team_pos[b] = u_pos_np[b, t]
             
        # Execute Shaping 
        if getattr(self, "reward_version", "v1") == "v2":
            from benchmarl.environments.lux.reward_exploration import compute_shaped_rewards_v2
            
            team_footprint = np.full((b_size, self.map_width, self.map_height), -50.0, dtype=np.float32)
            if hasattr(self, "agent_trajectory_map"):
                for b in range(b_size):
                    team_footprint[b] = np.max(self.agent_trajectory_map[b, t_ids_np[b]], axis=0)
            
            shaped_global, shaped_local, reward_components = compute_shaped_rewards_v2(
                current_team_mask=current_team_mask,
                current_team_pos=current_team_pos,
                actions=actions,
                delta_points=delta_pts,
                delta_visible=delta_vis,
                delta_energy=delta_en,
                spawn_pos=self.spawn_pos,
                known_relic_mask=self.known_relic_mask,
                known_relic_pos=self.known_relic_pos,
                step_count=steps,
                footprint_map=team_footprint
            )
        else:
            shaped_reward, reward_components = compute_shaped_rewards(
                current_team_mask=current_team_mask,
                current_team_pos=current_team_pos,
                actions=actions,
                delta_points=delta_pts,
                delta_visible=delta_vis,
                delta_energy=delta_en,
                spawn_pos=self.spawn_pos,
                known_relic_mask=self.known_relic_mask,
                known_relic_pos=self.known_relic_pos,
                step_count=steps
            )
        for b in range(b_size):
             if getattr(self, "reward_version", "v1") == "v2":
                 # Distribute calculated team shaped-reward globally array + the local agent specific scores!
                 r_np[b, :] = (shaped_global[b] + shaped_local[b, :]) * self.reward_scaling
             else:
                 r_np[b, :] = shaped_reward[b] * self.reward_scaling
                 
        reward_components["total_reward"] = r_np.copy()
        self.last_reward_components = reward_components
        
        if not hasattr(self, "episode_reward_components"):
            self.episode_reward_components = {k: np.zeros(b_size, dtype=np.float32) for k in reward_components.keys()}
        
        for k, v in reward_components.items():
            v_np = np.asarray(v)
            if v_np.ndim == 2:
                self.episode_reward_components[k] += np.sum(v_np, axis=-1)
            else:
                self.episode_reward_components[k] += v_np
             
        done_np = term_np | trunc_np
        
        # Build Next Ops using absolute indices
        active_units = np.zeros((b_size, self.max_units), dtype=bool)
        for b in range(b_size):
            active_units[b, :] = m[b, t_ids_np[b], :]
            
        action_mask = np.zeros((b_size, self.max_units, 5), dtype=np.bool_)
        action_mask[active_units, :] = True
        action_mask[~active_units, 0] = True
        
        self.last_point_delta = delta_pts.copy()
        obs_array = self._build_spatial_observation(self.jax_obs, t_ids_np, None)
        
        # Buffer coordinates for Ghost Trace (Ch 9) next step
        pos_cache = self._get_v(self._get_v(self.jax_obs["player_0"], "units"), "position")
        if pos_cache is not None:
            self.last_unit_pos = np.asarray(pos_cache, dtype=np.int32).copy()
        
        # Create output TensorDict
        device = self.device
        
        agents_td = TensorDict({
            "observation": torch.tensor(obs_array, device=device),
            "action_mask": torch.tensor(action_mask, device=device),
            "reward": torch.tensor(r_np, device=device).unsqueeze(-1),
            "done": torch.tensor(done_np, device=device).unsqueeze(-1).unsqueeze(-1).expand(-1, self.max_units, 1),
            "terminated": torch.tensor(term_np, device=device).unsqueeze(-1).unsqueeze(-1).expand(-1, self.max_units, 1),
            "truncated": torch.tensor(trunc_np, device=device).unsqueeze(-1).unsqueeze(-1).expand(-1, self.max_units, 1)
        }, batch_size=torch.Size([b_size, self.max_units]))
        
        if not hasattr(self, "max_agent_points"):
             self.max_agent_points = np.zeros(b_size, dtype=np.float32)
             self.max_opp_points = np.zeros(b_size, dtype=np.float32)

        opp_pts_np = np.zeros(b_size, dtype=np.float32)
        for b in range(b_size):
            t = t_ids_np[b]
            p0_dict = self._get_v(self.jax_obs["player_0"], "team_points")
            opp_pts_np[b] = float(np.asarray(p0_dict)[b, 1-t] if p0_dict is not None else 0.0)
            
            # Catch the highest score before JAX auto-resets the array to 0
            self.max_agent_points[b] = max(self.max_agent_points[b], self.prev_points[b])
            self.max_opp_points[b] = max(self.max_opp_points[b], opp_pts_np[b])

        info_dict = {
            "agent_points": torch.tensor(self.max_agent_points, dtype=torch.float32, device=device).unsqueeze(-1).clone(),
            "opponent_points": torch.tensor(self.max_opp_points, dtype=torch.float32, device=device).unsqueeze(-1).clone()
        }
        
        if hasattr(self, "episode_reward_components"):
            for k, v in self.episode_reward_components.items():
                info_dict[f"rc_{k}"] = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(-1).clone()

        td = TensorDict({
            "agents": agents_td,
            "done": torch.tensor(done_np, device=device).unsqueeze(-1),
            "terminated": torch.tensor(term_np, device=device).unsqueeze(-1),
            "truncated": torch.tensor(trunc_np, device=device).unsqueeze(-1),
            "info": TensorDict(info_dict, batch_size=torch.Size([b_size])),
        }, batch_size=torch.Size([b_size]))
        
        # Reset trackers for the next episode where done
        for b in range(b_size):
            if done_np[b]:
                self.max_agent_points[b] = 0.0
                self.max_opp_points[b] = 0.0
                
        
        return td

    def render(self, mode="rgb_array"):
        if not hasattr(self.raw_env, 'renderer'):
            return None
        
        # Force a single unbatched state render for environment index 0
        state_unbatched = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), self.env_state)
        
        import pygame
        from pygame import surfarray
        # Initialize bare minimum for headless rendering
        if getattr(self.raw_env.renderer, "screen", None) is None:
            if not pygame.get_init():
                pygame.init()
                pygame.display.init()
                pygame.font.init()
                pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
            self.raw_env.renderer.display_options = {
                "show_grid": True,
                "show_relic_spots": False,
                "show_sensor_mask": True,
                "show_vision_power_map": True,
                "show_energy_field": False,
            }
            # We must trick renderer into setting screen/surface
            TILE_SIZE = 64
            params = self.env_params
            screen_width = params.map_width * TILE_SIZE
            screen_height = params.map_height * TILE_SIZE
            self.raw_env.renderer.screen = pygame.Surface((screen_width, screen_height))
            self.raw_env.renderer.surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
            
        self.raw_env.renderer._update_display(state_unbatched, self.env_params)
        surface_array = surfarray.array3d(self.raw_env.renderer.surface)
        
        # Pygame surfarray returns (X, Y, RGB). We transpose to (RGB, Y, X) for Torchvision compatibility natively
        # But BenchMARL usually takes (H, W, C) so we just return (Y, X, C)
        surface_array = np.transpose(surface_array, (1, 0, 2))
        return torch.tensor(surface_array, dtype=torch.uint8)
