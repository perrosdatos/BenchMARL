import os
import sys
import torch
import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
from tensordict import TensorDict

# Add rulebased agent paths for the opponent
moth_dir = "/home/carlos/Documents/github/msc_ai_thesis_marl_lux"
sys.path.append(os.path.abspath(moth_dir))

try:
    from rulebased_agent_main import Agent
except ImportError:
    print(f"Warning: Could not import Agent from {moth_dir}.")
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

def play_match(checkpoint_path=None, seed=42):
    # Initialize Hydra configuring the experiment identically to the run command
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # The config path is internal to benchmarl/conf
    benchmarl_conf_path = "benchmarl/conf"
    
    with hydra.initialize(version_base=None, config_path=benchmarl_conf_path):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "algorithm=mappo",
                "task=lux/match",
                "model=layers/cnn",
                "model@critic_model=layers/cnn",
                "experiment.sampling_device=cpu",
                "experiment.train_device=cpu",
                "experiment.buffer_device=cpu"
            ],
        )

        print("\nLoading mapped experiment architecture...")
        experiment = load_experiment_from_hydra(cfg, task_name="lux/match")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint state from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            experiment.load_state_dict(state_dict)
        else:
            print("No valid checkpoint provided. Evaluating randomly initialized MAPPO network.")
            
        policy = experiment.algorithm.get_policy_for_collection()
        env = experiment.test_env

        # Since it's batched initially for MAPPO training (e.g. 10 envs), we just evaluate index 0.
        # But wait, experiment.rollout_env handles `experiment.config.on_policy_n_envs_per_worker`.
        # To make it isolated, we configure evaluate on 1 match:
        
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
            
        print(f"\n--- Starting MAPPO vs Rule-based Match (Seed: {seed}) ---")
        
        # Reset TorchRL map
        td = env.reset()
        step = 0
        
        env_wrapper = env
        while hasattr(env_wrapper, "env"):
             if hasattr(env_wrapper, "opp_actions"):
                 break
             env_wrapper = env_wrapper.env
             
        # env_wrapper contains the opp_actions. But wait! The actual LuxTorchRLEnv is the base env.
        # We can extract the unbatched lux wrapper:
        base_lux = env.base_env
             

        while True:
            # MAPPO Action (Deterministic sampling)
            with torch.no_grad():
                td = policy(td)
            
            # The action is in td.get(("agents", "action"))
            # Shape is (n_envs, 16)
            
            # Extract player 1 observation natively
            pseudo_obs_p1 = build_pseudo_obs(base_lux, base_lux.jax_obs, player_id=1)
            
            opp_action_full = opp_agent.act(step, pseudo_obs_p1)
            opp_action = opp_action_full[:, 0] # Extract action type (16,)
            
            # Set opponent actions for the environment (replicating to all n_envs if needed)
            # base_lux.batch_size is (n_envs,)
            base_lux.opp_actions = opp_action[None, :].repeat(base_lux.batch_size[0], axis=0) 
            
            td = env.step(td)
            
            base_lux.render()
            
            step += 1
            
            # Check dones natively via TorchRL td
            dones = td.get(("next", "done")) # shape (N, 1) or (N, 16, 1)
            done = dones[0].any().item()
            
            if step % 10 == 0:
                print(f"Step {step:3} completed...")
                
            td = td.get("next")
                
            if done:
                print("\n--- Match Finished ---")
                break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    play_match(args.checkpoint, seed=args.seed)
