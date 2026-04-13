from benchmarl.environments import LuxTask
from benchmarl.experiment import Experiment, ExperimentConfig
from torchrl.envs.utils import check_env_specs

import torch
import matplotlib.pyplot as plt

def main():
    print("Testing LuxTask definition and shape retrieval...")
    task = LuxTask.MATCH.get_from_yaml()
    print("Task Name:", task.name)
    print("Config Object:", task.config)
    
    # 2. Test Get Env Fun
    print("\nInitializing TorchRL Environment...")
    env_fun = task.get_env_fun(
        num_envs=2, 
        continuous_actions=False, 
        seed=1994, 
        device="cpu"
    )
    env = env_fun()
    print("Environment Created:", env)
    
    print("\nSpecs:")
    print("Action spec:", env.action_spec)
    print("Observation spec:", env.observation_spec)
    print("Reward spec:", env.reward_spec)
    print("Done spec:", env.done_spec)
    
    # 3. Test Reset
    print("\nTesting Reset...")
    td = env.reset()
    print("Reset TensorDict:")
    print(td)
    print("Observation shape:", td.get(("agents", "observation")).shape)
    print("Action mask shape:", td.get(("agents", "action_mask")).shape)
    
    # 4. Test Step
    print("\nTesting Step...")
    # Sample random valid actions using action mask
    action_mask = td.get(("agents", "action_mask"))
    b, u, c = action_mask.shape
    logits = torch.rand(b, u, c)
    logits[~action_mask] = -float('inf')
    actions = logits.argmax(dim=-1, keepdim=True)
    td.set(("agents", "action"), actions)
    
    result_td = env.step(td)
    print("Step TensorDict:")
    print(result_td)
    print("Reward sum:", result_td.get(("next", "agents", "reward")).sum().item())
    print("Done?", result_td.get(("next", "done")).any().item())
    
    # 5. Test Render
    print("\nTesting Render...")
    try:
        render_output = task.render_callback(None, env, result_td)
        print("Render Output shape:", render_output.shape)
        plt.imsave("test_render_lux.png", render_output.numpy())
        print("Saved render to test_render_lux.png")
    except Exception as e:
        print("Render failed:", e)

if __name__ == "__main__":
    main()
