import torch
import numpy as np
import sys
import os

from tensordict import TensorDict
from benchmarl.environments.lux.lux_with_sap import LuxSapTorchRLEnv

def render_ascii_map(env: LuxSapTorchRLEnv, b_idx=0):
    """
    Reads the global state to print an ASCII representation, bypassing Fog of War.
    """
    state = env.env_state
    map_w = env.map_width
    map_h = env.map_height
    
    # 1. Base map features
    tile_type = np.asarray(env._get_v(state.map_features, "tile_type"))[b_idx]
    
    # 2. Relics
    relic_nodes = np.asarray(env._get_v(state, "relic_nodes"))[b_idx]
    relic_mask = np.asarray(env._get_v(state, "relic_nodes_mask"))[b_idx]
    
    # 3. Units
    u_obj = env._get_v(state, "units")
    upos = np.asarray(env._get_v(u_obj, "position"))[b_idx]
    uenergy = np.asarray(env._get_v(u_obj, "energy"))[b_idx]
    umask = np.asarray(env._get_v(state, "units_mask"))[b_idx]
    
    upos_0 = upos[0]; uenergy_0 = uenergy[0]; umask_0 = umask[0]
    upos_1 = upos[1]; uenergy_1 = uenergy[1]; umask_1 = umask[1]
    
    grid = [["." for _ in range(map_h)] for _ in range(map_w)]
    
    # Fill
    for x in range(map_w):
        for y in range(map_h):
            t = tile_type[x, y]
            if t == 2: grid[x][y] = "#" # Asteroid
            elif t == 1: grid[x][y] = "~" # Nebula
            
    for r_idx in range(len(relic_mask)):
        if relic_mask[r_idx]:
            rx, ry = relic_nodes[r_idx]
            grid[rx][ry] = "@"
            
    # Track which unit is where for detailed printing later
    p0_units = []
    p1_units = []
    
    for u in range(env.max_units):
        if umask_1[u]:
            x, y = upos_1[u]
            if x >= 0 and y >= 0:
                grid[x][y] = "O"
                p1_units.append(f"O{u}(E:{uenergy_1[u][0] if isinstance(uenergy_1[u], np.ndarray) else uenergy_1[u]:.0f})")
                
        if umask_0[u]:
            x, y = upos_0[u]
            if x >= 0 and y >= 0:
                grid[x][y] = "X"
                p0_units.append(f"X{u}(E:{uenergy_0[u][0] if isinstance(uenergy_0[u], np.ndarray) else uenergy_0[u]:.0f})")
                
    # Print transposed for standard terminal viewing
    print("\n" + "="*30)
    for y in range(map_h):
        row = ""
        for x in range(map_w):
             row += grid[x][y] + " "
        print(row)
    print("="*30)
    print(f"Player 0 (X) Units: {', '.join(p0_units)}")
    print(f"Player 1 (O) Units: {', '.join(p1_units)}")
    print("="*30)

def main():
    print("Initializing environment LuxSapTorchRLEnv...")
    env = LuxSapTorchRLEnv(batch_size=1, reward_version="v2")
    
    # Set rulebased agent for player 1
    sys.path.append(os.path.abspath("/home/carlos/Documents/github/msc_ai_thesis_marl_lux"))
    from agent import Agent
    rulebased_bot = Agent("player_1", {"max_units": 16, "map_width": 24, "map_height": 24})
    
    td = env.reset()
    
    step = 0
    while True:
        # P0 uses T_ID to figure out if we are acting for 0 or 1.
        # But wait! LuxSapTorchRLEnv assigns random team_ids internally.
        # Let's peek at team_id for batch 0 to know which "player_n" from JAX we represent natively!
        my_team = int(env.team_ids[0].cpu().numpy())
        opp_team = 1 if my_team == 0 else 0
        
        print(f"\n--- STEP {step} ---")
        print(f"You are Team {my_team} (Mapping to X in ASCII)")
        render_ascii_map(env, 0)
        
        # Human Input for My Team Action
        # Create blank actions
        actions = np.zeros((1, 16), dtype=np.int64)
        
        print("Sap Mechanics: 0=Idle, 1=Up, 2=Right, 3=Down, 4=Left, 5=SAP (Shoots Up)")
        print("Format: <unit_id>,<action_id> (e.g. '0,5' to sap with unit 0). Press Enter to skip.")
        user_in = input("Command: ").strip()
        if user_in:
            try:
                pts = user_in.split(',')
                u_id = int(pts[0].strip())
                a_id = int(pts[1].strip())
                actions[0, u_id] = a_id
                print(f"Agent {u_id} queued action {a_id}")
            except Exception as e:
                print(f"Invalid input ({e}), skipping turn.")
                
        # Get Rulebased actions for Opponent if step < 10
        if step < 10:
            print("Rulebased Opponent is ACTIVE.")
            pseudo_obs = env._build_pseudo_obs(env.jax_obs, opp_team, 0)
            opp_act = rulebased_bot.act(step, pseudo_obs)[:, 0]
            # Since env._step expects to overwrite opp_actions inside for standard runs,
            # we manually hijack env.opp_actions here
            env.opp_actions = np.expand_dims(opp_act, axis=0)
        else:
            print("Rulebased Opponent is IDLE (Stop).")
            env.opp_actions = np.zeros((1, 16), dtype=np.int32)
            
        td["agents", "action"] = torch.tensor(actions, device=env.device)
        
        # Step the environment
        td = env.step(td)
        
        # Print rewards 
        rc = env.last_reward_components
        print(f"Reward this step: Total = {rc['total_reward'][0].sum():.4f}")
        print(f"  Kills  : {rc.get('combat_kill', np.zeros(1))[0].sum():.4f}")
        print(f"  Deaths : {rc.get('combat_death', np.zeros(1))[0].sum():.4f}")
        
        if td["done"][0].item():
            print("\nMatch is finished!")
            break
            
        step += 1

if __name__ == "__main__":
    main()
