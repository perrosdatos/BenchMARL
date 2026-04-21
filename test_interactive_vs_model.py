import os
import sys
import numpy as np
import jax
import torch
import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra

import argparse

# Explicitly disable WandB mapping globally
os.environ["WANDB_MODE"] = "disabled"

# Import BenchMARL loader
from benchmarl.hydra_config import load_experiment_from_hydra

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Absolute path to the PyTorch checkpoint .pt file")
    parser.add_argument("--algo", type=str, required=True, help="Algorithm used (e.g. 'mappo' or 'qmix')")
    args, _ = parser.parse_known_args()
    
    checkpoint_path = args.checkpoint    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # The config path is internal to benchmarl/conf
    benchmarl_conf_path = "benchmarl/conf"
    
    with hydra.initialize(version_base=None, config_path=benchmarl_conf_path):
        # Configure identical to the run sweep/task, using match_v2
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"algorithm={args.algo}",
                "task=lux/match_v2",
                "model=layers/cnn",
                "model@critic_model=layers/cnn",
                "experiment.sampling_device=cpu",
                "experiment.train_device=cpu",
                "experiment.buffer_device=cpu",
                "experiment.checkpoint_interval=120000",
                "experiment.loggers=[]",
                "seed=324"
            ],
        )

        print(f"\nLoading mapped {args.algo.upper()} experiment architecture...")
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
        
        print(f"\n--- Starting {args.algo.upper()} vs Rule-based Iterative Match ---")
        
        td = env.reset()
        step = 0
        
        while True:
            input(f"\n[Press ENTER to execute step {step}] ")
            
            # MAPPO Action (Deterministic sampling)
            with torch.no_grad():
                td = policy(td)
            
            # Extract Model Actions
            model_actions = td.get(("agents", "action")).cpu().numpy()
            
            # STEP
            td = env.step(td)
            
            # Opponent actions are computed internally during step and saved in base_lux
            opp_actions = base_lux.opp_actions[0] if hasattr(base_lux, "opp_actions") else "Unknown"
            
            # Debug actions
            print(f"Model actions P0 (shape {model_actions.shape}):\n{model_actions[0]}")
            print(f"Rulebased actions P1:\n{opp_actions}")
            
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
