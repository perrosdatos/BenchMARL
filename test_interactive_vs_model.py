import os
import sys
import numpy as np
import jax
import torch
import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra

# Explicitly disable WandB mapping globally
os.environ["WANDB_MODE"] = "disabled"

# Add rulebased agent paths for the opponent
moth_dir = "/home/carlos/Documents/github/msc_ai_thesis_marl_lux"
sys.path.append(os.path.abspath(moth_dir))

try:
    from rulebased_agent_main import Agent
except ImportError:
    print(f"Warning: Could not import Agent from {moth_dir}. Make sure the path is correct.")
    Agent = None

# Import BenchMARL loader
from benchmarl.hydra_config import load_experiment_from_hydra

def build_pseudo_obs(base_lux, jax_obs, player_id=1):
    """
    Simulates the standard GameState dictionary for the rule-based Agent.
    """
    p_key = f"player_{player_id}"
    p_obs = jax_obs[p_key]
    
    _v = base_lux._get_v
    return {
        "units_mask": _v(p_obs, "units_mask")[0].tolist(),
        "units": {
            "position": _v(_v(p_obs, "units"), "position")[0].tolist(),
            "energy": _v(_v(p_obs, "units"), "energy")[0].tolist()
        },
        "relic_nodes": _v(p_obs, "relic_nodes")[0].tolist(),
        "relic_nodes_mask": _v(p_obs, "relic_nodes_mask")[0].tolist(),
        "team_points": _v(p_obs, "team_points")[0].tolist()
    }

def main():
    # 1. Target the specific checkpoint provided
    checkpoint_path = "/home/carlos/Documents/github/msc_ai_thesis_experiments/BenchMARL/outputs/2026-04-19/03-46-34/mappo_match_v2_cnn__bcb1bfdc_26_04_19-03_46_34/checkpoints/checkpoint_1500000.pt"
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # The config path is internal to benchmarl/conf
    benchmarl_conf_path = "benchmarl/conf"
    
    with hydra.initialize(version_base=None, config_path=benchmarl_conf_path):
        # Configure identical to the run sweep/task, using match_v2
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "algorithm=mappo",
                "task=lux/match_v2",
                "model=layers/cnn",
                "model@critic_model=layers/cnn",
                "experiment.sampling_device=cpu",
                "experiment.train_device=cpu",
                "experiment.buffer_device=cpu",
                "experiment.checkpoint_interval=120000",
                "experiment.loggers=[]"
            ],
        )

        print("\nLoading mapped experiment architecture...")
        experiment = load_experiment_from_hydra(cfg, task_name="lux/match_v2")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint state from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            experiment.load_state_dict(state_dict)
        else:
            print(f"Error: Checkpoint {checkpoint_path} not found.")
            return
            
        policy = experiment.algorithm.get_policy_for_collection()
        env = experiment.test_env

        # Extract unbatched lux wrapper for stats and native interaction
        env_wrapper = env
        while hasattr(env_wrapper, "env"):
             if hasattr(env_wrapper, "opp_actions"):
                 break
             env_wrapper = env_wrapper.env
             
        base_lux = env_wrapper if hasattr(env_wrapper, "opp_actions") else env.base_env
        
        env_cfg = {
            "max_units": 16,
            "map_width": 24,
            "map_height": 24
        }
        
        if Agent:
            opp_agent = Agent("player_1", env_cfg)
        else:
            print("Error: Rule-based Agent not found. Cannot run VS match.")
            return
            
        print("\n--- Starting MAPPO vs Rule-based Iterative Match ---")
        
        td = env.reset()
        step = 0
        
        while True:
            input(f"\n[Press ENTER to execute step {step}] ")
            
            # MAPPO Action (Deterministic sampling)
            with torch.no_grad():
                td = policy(td)
            
            model_actions = td.get(("agents", "action")).cpu().numpy()
            
            # Extract player 1 observation natively for Rulebased agent
            pseudo_obs_p1 = build_pseudo_obs(base_lux, base_lux.jax_obs, player_id=1)
            
            opp_action_full = opp_agent.act(step, pseudo_obs_p1)
            opp_actions = opp_action_full[:, 0] # Extract action type (16,)
            
            # Debug actions
            print(f"Model actions P0 (shape {model_actions.shape}):\n{model_actions[0]}")
            print(f"Rulebased actions P1 (shape {opp_actions.shape}):\n{opp_actions}")
            
            # Set opponent actions for the environment (replicating to all n_envs if needed)
            base_lux.opp_actions = opp_actions[None, :].repeat(base_lux.batch_size[0], axis=0) 
            
            # STEP
            td = env.step(td)
            
            # Extract stepped variables
            dones = td.get(("next", "done"))
            rewards = td.get(("next", "agents", "reward"))
            
            step += 1
            
            print("\n--- Step Results ---")
            print(f"Step: {step} | Terminal: {dones[0].any().item()}")
            
            # Team points
            p_pts0 = np.asarray(base_lux._get_v(base_lux.jax_obs["player_0"], "team_points"))[0]
            p_pts1 = np.asarray(base_lux._get_v(base_lux.jax_obs["player_1"], "team_points"))[0]
            print(f"Team Points: P0: {p_pts0[0]} | P1: {p_pts1[1]}")
            
            n_pts = np.asarray(base_lux.env_state.team_points)[0]
            n_wins = np.asarray(base_lux.env_state.team_wins)[0]
            n_steps = np.asarray(base_lux.env_state.steps)[0]
            n_match_steps = np.asarray(base_lux.env_state.match_steps)[0]
            print(f"Native State | steps: {n_steps} | match_steps: {n_match_steps} | wins: P0={n_wins[0]} P1={n_wins[1]} | points: P0={n_pts[0]} P1={n_pts[1]}")
            
            print(f"TorchRL Agent Shaped Reward Matrix Sum: {torch.sum(rewards[0]).item():.4f}")
            if hasattr(base_lux, "last_reward_components"):
                for k, v in base_lux.last_reward_components.items():
                    if np.sum(np.asarray(v)[0]) != 0.0:
                        print(f"  - {k}: {np.sum(np.asarray(v)[0]):.4f}")
            
            # RENDER
            unbatched_state = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), base_lux.env_state)
            base_lux.raw_env.render(unbatched_state, base_lux.env_params)
            
            td = td.get("next")
                
            if dones[0].any().item():
                print(f"\n[Match Finished / TorchRL Flagged done=True at step {step}]")
            if step > 155:
                print("\nForce stopping visualizer to prevent infinite loop.")
                break

if __name__ == "__main__":
    main()
