import jax
import torch
import numpy as np
import pandas as pd
from tensordict import TensorDict
from benchmarl.environments.lux.lux_env import LuxTorchRLEnv

import sys
import os

moth_dir = "/home/carlos/Documents/github/msc_ai_thesis_marl_lux"
sys.path.append(os.path.abspath(moth_dir))

try:
    from rulebased_agent_main import Agent
except ImportError:
    Agent = None

def build_pseudo_obs(jax_obs, team_id):
    p0 = jax_obs[f"player_{team_id}"]
    
    def get_v(d, k):
        if hasattr(d, k): return getattr(d, k)
        elif isinstance(d, dict) and k in d: return d[k]
        return None

    return {
        "units_mask": np.asarray(get_v(p0, "units_mask"))[0].tolist(),
        "sensor_mask": np.asarray(get_v(p0, "sensor_mask"))[0].tolist(),
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
    
    env = LuxTorchRLEnv(batch_size=1, max_steps=max_steps, match_count=match_count, seed=42, reward_version="v2")
    td = env.reset()
    
    team_id = int(env.team_ids[0].item())
    
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
    records = []

    print(f"Running automated collection up to step {max_steps * match_count}...")
    
    while True:
        if agent:
            pseudo_obs = build_pseudo_obs(env.jax_obs, team_id)
            act_rule = agent.act(step, pseudo_obs)
            actions = act_rule[:, 0]
        else:
            actions = np.random.randint(0, 5, size=(16,))
            
        td["agents", "action"] = torch.tensor(actions, dtype=torch.long, device=env.device).unsqueeze(0)
        td["action"] = td["agents", "action"]
        
        td = env.step(td).get("next")
        dones = td["done"]
        rewards = td["agents", "reward"]
        
        step += 1
        
        n_pts = np.asarray(env.env_state.team_points)[0]
        n_wins = np.asarray(env.env_state.team_wins)[0]
        n_match_steps = np.asarray(env.env_state.match_steps)[0]
        
        record = {
            "step": step,
            "reward_sum": torch.sum(rewards[0]).item(),
            "reward_per_agent": torch.sum(rewards[0]).item() / 16.0,
            "match_points_P0": n_pts[0],
            "match_points_P1": n_pts[1],
            "match_step": n_match_steps,
            "wins_P0": n_wins[0],
            "wins_P1": n_wins[1]
        }
        
        if hasattr(env, "last_reward_components"):
            for k, v in env.last_reward_components.items():
                record[k] = float(np.sum(np.asarray(v)[0]))
                
        records.append(record)
        
        if dones[0].item() or step >= max_steps * match_count:
            break

    env.close()
    
    df = pd.DataFrame(records)
    csv_file = "reward_distribution.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved {len(df)} steps to {csv_file}")
    
    print("\n--- Distribution Analysis ---")
    metrics = ["reward_per_agent", "base_points", "collision_penalty", 
               "relic_proximity", "relic_discovery", "energy_gain", "stagnation_penalty", "movement_bonus"]
    
    for metric in metrics:
        if metric in df.columns:
            print(f"{metric}: mean={df[metric].mean():.4f}, min={df[metric].min():.4f}, max={df[metric].max():.4f}")
            
    print(f"\nFinal P0 Points: {df['match_points_P0'].iloc[-1]}")
    print(f"Final P1 Points: {df['match_points_P1'].iloc[-1]}")

if __name__ == "__main__":
    main()
