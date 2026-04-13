import os
import sys
import numpy as np
import jax
import torch
from tensordict import TensorDict

# Add rulebased agent paths
moth_dir = "/home/carlos/Documents/github/msc_ai_thesis_marl_lux"
sys.path.append(os.path.abspath(moth_dir))

try:
    from rulebased_agent_main import Agent
except ImportError:
    print(f"Warning: Could not import Agent from {moth_dir}. Make sure the path is correct.")
    Agent = None

from benchmarl.environments.lux.lux_env import LuxTorchRLEnv

def build_pseudo_obs(jax_obs, team_id_num):
    """
    Simulates the standard GameState dictionary for the rule-based Agent 
    by unpacking the symmetric JAX observation nested structs for index 0.
    """
    p0 = jax_obs[f"player_{team_id_num}"]
    def get_v(o, k): return getattr(o, k) if hasattr(o, k) else o.get(k)
    
    return {
        "units_mask": np.asarray(get_v(p0, "units_mask"))[0].tolist(),
        "units": {
            "position": np.asarray(get_v(get_v(p0, "units"), "position"))[0].tolist(),
            "energy": np.asarray(get_v(get_v(p0, "units"), "energy"))[0].tolist()
        },
        "relic_nodes": np.asarray(get_v(p0, "relic_nodes"))[0].tolist(),
        "relic_nodes_mask": np.asarray(get_v(p0, "relic_nodes_mask"))[0].tolist(),
        "team_points": np.asarray(get_v(p0, "team_points"))[0].tolist()
    }

def main():
    max_steps = 50
    match_count = 3
    
    # 1. Initialize Wrapper Environment (single instance for debugging)
    # Using TorchRL wrapper
    env = LuxTorchRLEnv(batch_size=1, max_steps=max_steps, match_count=match_count, seed=1994)
    
    # 2. Reset
    print("\n--- Resetting Environment ---")
    td = env.reset()
    
    # We will simulate controlling the team assigned to agent 0 internally
    # LuxTorchRLEnv scrambles teams internally, but we can just use team_ids[0]
    team_id = int(env.team_ids[0].item())
    
    # 3. Create Rule-based Agent
    env_cfg = {
        "max_units": 16,
        "map_width": 24,
        "map_height": 24
    }
    
    if Agent:
        agent = Agent(f"player_{team_id}", env_cfg)
    else:
        agent = None
    
    step = 0

    while True:
        input(f"\n[Press ENTER to execute step {step}] ")
        
        # ACT: Use rule-based agent or random if agent missing
        if agent:
            pseudo_obs = build_pseudo_obs(env.jax_obs, team_id)
            act_rule = agent.act(step, pseudo_obs)
            actions = act_rule[:, 0] # (16,)
        else:
            actions = np.random.randint(0, 5, size=(16,))
            
        print(f"Actions taken: {actions}")
        
        # STEP
        # Pytorch representation building
        td["agents", "action"] = torch.tensor(actions, dtype=torch.long, device=env.device).unsqueeze(0)
        td["action"] = td["agents", "action"]
        
        td = env.step(td).get("next")
        
        dones = td["done"]
        rewards = td["agents", "reward"]
        
        step += 1
        
        print("\n--- Step Results ---")
        print(f"Step: {step} | Terminal: {dones[0].item()}")
        
        # In TorchRL 'team_points' are kept inside the internal jax_obs directly
        p_pts0 = np.asarray(env._get_v(env.jax_obs["player_0"], "team_points"))[0]
        p_pts1 = np.asarray(env._get_v(env.jax_obs["player_1"], "team_points"))[0]
        # In LuxAI_S3, team_points is NOT symmetrically twisted. Index 0 is ALWAYS P0, Index 1 is ALWAYS P1
        print(f"Team Points: P0: {p_pts0[0]} | P1: {p_pts1[1]}")
        
        n_pts = np.asarray(env.env_state.team_points)[0]
        n_wins = np.asarray(env.env_state.team_wins)[0]
        n_steps = np.asarray(env.env_state.steps)[0]
        n_match_steps = np.asarray(env.env_state.match_steps)[0]
        print(f"Native State | steps: {n_steps} | match_steps: {n_match_steps} | wins: P0={n_wins[0]} P1={n_wins[1]} | points: P0={n_pts[0]} P1={n_pts[1]}")
        
        if hasattr(env, "abosolute_raw_terminated_obj"):
            print(f"RAW NATIVE TERMINATED_DICT: {env.abosolute_raw_terminated_obj}")
            print(f"RAW NATIVE TRUNCATED_DICT: {env.abosolute_raw_truncated_obj}")

        # Print shape integrated TorchRL reward
        print(f"TorchRL Agent Shaped Reward Matrix Sum: {torch.sum(rewards[0]).item():.4f}")
        if hasattr(env, "last_reward_components"):
            for k, v in env.last_reward_components.items():
                if np.sum(np.asarray(v)[0]) != 0.0:
                    print(f"  - {k}: {np.sum(np.asarray(v)[0]):.4f}")
        
        # Unit status dynamically tracking symmetric mapping
        my_obs = env.jax_obs[f"player_{team_id}"]
        my_mask = np.asarray(env._get_v(my_obs, "units_mask"))[0, team_id]
        my_pos = np.asarray(env._get_v(env._get_v(my_obs, "units"), "position"))[0, team_id]
        
        active_idx = np.where(my_mask)[0]
        active_coords = [f"u{u}:({int(my_pos[u,0])}, {int(my_pos[u,1])})" for u in active_idx]
        print(f"Active Units: {' | '.join(active_coords)}")
        
        # RENDER
        unbatched_state = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), env.env_state)
        env.raw_env.render(unbatched_state, env.env_params)
        
        if dones[0].item():
            print(f"\n[TorchRL Flagged done=True at step {step}]")
        if step > 155:
            print("\nForce stopping visualizer to prevent infinite loop.")
            break

    env.close()

if __name__ == "__main__":
    main()
