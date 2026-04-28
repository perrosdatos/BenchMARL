import numpy as np
from typing import Optional

def compute_shaped_rewards(
    current_team_mask,
    current_team_pos,
    actions,
    delta_points,
    delta_visible,
    delta_energy,
    spawn_pos,
    known_relic_mask,
    known_relic_pos,
    step_count
):
    """
    Computes shaped rewards for the agents.
    """
    batch_size = delta_points.shape[0]
    shaped = np.zeros(batch_size, dtype=np.float32)
    max_units = actions.shape[-1]
    
    components = {
        "base_points": delta_points.astype(np.float32),
        "collision_penalty": np.zeros(batch_size, dtype=np.float32),
        "diagonal_bonus": np.zeros(batch_size, dtype=np.float32),
        "movement_bonus": np.zeros(batch_size, dtype=np.float32),
        "relic_proximity": np.zeros(batch_size, dtype=np.float32),
        "relic_discovery": np.zeros(batch_size, dtype=np.float32),
        "relic_farming": np.zeros(batch_size, dtype=np.float32),
        "fog_discovery": np.zeros(batch_size, dtype=np.float32),
        "energy_gain": np.zeros(batch_size, dtype=np.float32),
        "stagnation_penalty": np.zeros(batch_size, dtype=np.float32),
    }

    # Parameters are directly passed in

    for b in range(batch_size):
        valid_indices = np.where(current_team_mask[b])[0]
        if len(valid_indices) == 0:
            continue

        unit_positions = current_team_pos[b, valid_indices]
        valid_pos_list = [tuple(p) for p in unit_positions if p[0] >= 0]

        # 1. Collision Penalty
        if len(valid_pos_list) > 1:
            unique_pos = set(valid_pos_list)
            collision_count = len(valid_pos_list) - len(unique_pos)
            if collision_count > 0:
                components["collision_penalty"][b] = -0.1 * float(collision_count)

        # 2. Relic Rewards
        seen_relic_indices = np.where(known_relic_mask[b])[0]
        if len(seen_relic_indices) > 0:
            relic_coords = known_relic_pos[b, seen_relic_indices]
            
            total_prox = 0.0
            farming_bonus = 0.0
            
            for u_idx in valid_indices:
                ux, uy = current_team_pos[b, u_idx]
                if ux < 0: continue
                
                # Manhattan distance to nearest relic
                dists = np.abs(relic_coords[:, 0] - ux) + np.abs(relic_coords[:, 1] - uy)
                min_d = np.min(dists)
                
                # Proximity Bonus
                total_prox += 0.005 * (24.0 - min_d)
                
                # Farming Bonus: Near relic (dist < 3) and gaining energy
                if min_d < 3.0 and delta_energy[b, u_idx] > 0:
                    farming_bonus += 0.01

            components["relic_proximity"][b] = float(total_prox)
            components["relic_discovery"][b] = float(len(seen_relic_indices)) * 0.1
            components["relic_farming"][b] = farming_bonus

        # 3. Diagonal Exploration Bonus
        sx, sy = spawn_pos[b]
        diag_sum = 0.0
        active_count = 0
        for ux, uy in unit_positions:
            if ux >= 0:
                dist_away = max(abs(ux - sx), abs(uy - sy))
                diag_sum += float(dist_away)
                active_count += 1
        
        if active_count > 0:
            avg_dist = diag_sum / active_count
            pressure = 0.2 if len(seen_relic_indices) > 0 else 0.8
            components["diagonal_bonus"][b] = (avg_dist / 24.0) * pressure

        # 4. Fog Discovery (with decay)
        if len(seen_relic_indices) == 0:
            step = float(step_count[b]) if step_count is not None else 0.0
            decay = max(0.2, 1.0 - (step / 500.0)) # Decay over 500 steps
            components["fog_discovery"][b] = float(delta_visible[b]) * 0.05 * decay

        # 5. Energy Bonus
        energy_sum = np.sum(delta_energy[b, valid_indices])
        components["energy_gain"][b] = energy_sum * 0.002

        # 6. Action-based bonuses/penalties
        for idx in valid_indices:
            act = int(actions[b, idx])
            if act == 0: # Idle
                if delta_energy[b, idx] <= 0:
                    # Increased penalty for doing nothing and not farming
                    components["stagnation_penalty"][b] -= 0.05 
            else: # Moving
                components["movement_bonus"][b] += 0.002

        # Final reward summation
        shaped[b] = (
            components["base_points"][b] * 10.0 +
            components["collision_penalty"][b] +
            components["diagonal_bonus"][b] +
            components["movement_bonus"][b] +
            components["relic_proximity"][b] +
            components["relic_discovery"][b] +
            components["relic_farming"][b] +
            components["fog_discovery"][b] +
            components["energy_gain"][b] +
            components["stagnation_penalty"][b]
        )

    return shaped, components

def compute_combat_rewards(
    current_team_mask,
    current_team_pos,
    actions,
    delta_points,
    delta_visible,
    delta_energy,
    spawn_pos,
    known_relic_mask,
    known_relic_pos: np.ndarray,
    step_count: np.ndarray,
    footprint_map: Optional[np.ndarray],
    delta_masks: np.ndarray,
    all_positions: np.ndarray
):
    """
    Computes shaped rewards for the agents (V3 WITH SAP COMBAT).
    """
    batch_size = delta_points.shape[0]
    max_units = actions.shape[-1]
    
    shaped_global = np.zeros(batch_size, dtype=np.float32)
    shaped_local = np.zeros((batch_size, max_units), dtype=np.float32)
    
    components = {
        "local_point_generation": np.zeros((batch_size, max_units), dtype=np.float32),
        "relic_discovery": np.zeros(batch_size, dtype=np.float32),
        "fog_discovery": np.zeros(batch_size, dtype=np.float32),
        
        "collision_penalty": np.zeros((batch_size, max_units), dtype=np.float32),
        "dispersion_bonus": np.zeros((batch_size, max_units), dtype=np.float32),
        "novelty_bonus": np.zeros((batch_size, max_units), dtype=np.float32),
        "relic_proximity": np.zeros((batch_size, max_units), dtype=np.float32),
        "relic_farming": np.zeros((batch_size, max_units), dtype=np.float32),
        "energy_gain": np.zeros((batch_size, max_units), dtype=np.float32),
        "stagnation_penalty": np.zeros((batch_size, max_units), dtype=np.float32),
        "overcrowding_penalty": np.zeros((batch_size, max_units), dtype=np.float32),
        "combat_kill": np.zeros((batch_size, max_units), dtype=np.float32),
        "combat_death": np.zeros((batch_size, max_units), dtype=np.float32),
    }

    # Parameters are directly passed in

    for b in range(batch_size):
        valid_indices = np.where(current_team_mask[b])[0]

        unit_positions = current_team_pos[b] # All 16 positions
        valid_pos_list = [tuple(unit_positions[i]) for i in valid_indices]
        
        # 0. Combat Tracking (Kills and Deaths)
        # Death: My unit's mask went from True (1) to False (0) -> delta_masks == -1
        dead_indices = np.where(delta_masks[b, 0, :] == -1)[0]
        for d_idx in dead_indices:
            components["combat_death"][b, d_idx] = 1.0 # Will be negative penalized in summation
            
        # Kill: Opponent unit's mask went from True (1) to False (0) -> delta_masks == -1
        dead_opp_indices = np.where(delta_masks[b, 1, :] == -1)[0]
        if len(dead_opp_indices) > 0:
            for do_idx in dead_opp_indices:
                ox, oy = all_positions[b, 1, do_idx]
                if ox < 0: continue
                
                # Find all valid agents that were close enough to participate in the kill
                # Chebyshev distance <= 2 perfectly covers melee (0), energy void (1), and the hardcoded SAP at (0, -1) + splash
                close_agents = []
                for u_idx in valid_indices:
                    ux, uy = unit_positions[u_idx]
                    if max(abs(ux - ox), abs(uy - oy)) <= 2:
                        close_agents.append(u_idx)
                        
                # Divide the reward equally among all participating agents
                if len(close_agents) > 0:
                    reward_share = 1.0 / len(close_agents)
                    for u_idx in close_agents:
                        components["combat_kill"][b, u_idx] += reward_share

        # ... (rest of the V2 exploration shaping logic) ...
        # 1. Collision Penalty (LOCAL)
        if len(valid_pos_list) > 1:
            pos_counts = {p: valid_pos_list.count(p) for p in set(valid_pos_list)}
            for u_idx in valid_indices:
                cnt = pos_counts[tuple(unit_positions[u_idx])]
                if cnt > 1:
                    # Normalized to max 1.0 across the combined team sum, negative because it's a penalty
                    components["collision_penalty"][b, u_idx] = - (float(cnt) / 16.0) / 16.0

        # 2. Relic Rewards (LOCAL and GLOBAL)
        seen_relic_indices = np.where(known_relic_mask[b])[0]
        if len(seen_relic_indices) > 0:
            relic_coords = known_relic_pos[b, seen_relic_indices]
            # Step A: Compute all (agent, relic) distance pairs
            all_pairs = []
            for u_idx in valid_indices:
                ux, uy = current_team_pos[b, u_idx]
                if ux < 0: continue
                for r_idx, (rx, ry) in enumerate(relic_coords):
                    d = float(abs(rx - ux) + abs(ry - uy))
                    all_pairs.append((d, u_idx, r_idx))
                    
            # Sort by distance (closest pairs first)
            all_pairs.sort(key=lambda x: x[0])
            
            # Step B: Greedy Assignment (Max 4 agents per relic)
            max_agents_per_relic = 4
            relic_counts = {r: 0 for r in range(len(seen_relic_indices))}
            assigned_agents = {} # maps u_idx -> (assigned_r_idx, distance)
            
            for d, u_idx, r_idx in all_pairs:
                if u_idx not in assigned_agents and relic_counts[r_idx] < max_agents_per_relic:
                    assigned_agents[u_idx] = (r_idx, d)
                    relic_counts[r_idx] += 1
                    
            # Step C: Reward Allocation
            eligible_farmers = []
            for u_idx in valid_indices:
                ux, uy = current_team_pos[b, u_idx]
                if ux < 0: continue
                
                if u_idx in assigned_agents:
                    # VIP Logic: They successfully claimed a quota for a relic
                    r_idx, min_d = assigned_agents[u_idx]
                    components["relic_proximity"][b, u_idx] = ((24.0 - min_d) / 24.0) / 16.0
                    if min_d < 3.0:
                        eligible_farmers.append(u_idx)
                else:
                    # Crowd Logic: They failed to claim a quota at ANY relic.
                    # Check how close they are to the nearest relic to apply overcrowding penalty.
                    dists = np.abs(relic_coords[:, 0] - ux) + np.abs(relic_coords[:, 1] - uy)
                    min_d = float(np.min(dists))
                    if min_d < 8.0:
                        components["overcrowding_penalty"][b, u_idx] = - ((8.0 - min_d) / 8.0) / 16.0
                        
            # Farming Bonus: Distributed equally among ONLY the eligible VIP miners
            if len(eligible_farmers) > 0 and delta_points[b] > 0:
                dist_pts = float(delta_points[b]) / float(len(eligible_farmers))
                for u_idx in eligible_farmers:
                    components["relic_farming"][b, u_idx] = dist_pts / 16.0

            # 5x5 Area of Influence for points distribution
            if delta_points[b] > 0:
                agents_in_5x5 = []
                for u_idx in valid_indices:
                    ux, uy = current_team_pos[b, u_idx]
                    if ux < 0: continue
                    # Check if within Chebyshev distance 2 of ANY seen relic (5x5 grid)
                    in_range = False
                    for r_idx in range(len(relic_coords)):
                        rx, ry = relic_coords[r_idx]
                        if max(abs(rx - ux), abs(ry - uy)) <= 2:
                            in_range = True
                            break
                    if in_range:
                        agents_in_5x5.append(u_idx)
                        
                if len(agents_in_5x5) > 0:
                    dist_pts = float(delta_points[b]) / float(len(agents_in_5x5))
                    for u_idx in agents_in_5x5:
                        components["local_point_generation"][b, u_idx] = dist_pts

            # Note: Global discovery (Normalized assuming ~6 max relics, / 6.0)
            components["relic_discovery"][b] = float(len(seen_relic_indices)) / 6.0

        # 3. Dispersion Bonus (LOCAL) replacing Diagonal Bonus
        for u_idx in valid_indices:
            ux, uy = current_team_pos[b, u_idx]
            dist_to_others = 0.0
            for o_idx in valid_indices:
                if u_idx != o_idx:
                    ox, oy = current_team_pos[b, o_idx]
                    dist_to_others += abs(ux - ox) + abs(uy - oy)
            # Normalize by rough max possible distance on 24x24 (approx 360 for 15 agents)
            components["dispersion_bonus"][b, u_idx] = (dist_to_others / 360.0) / 16.0

        # 4. Fog Discovery (with decay) (GLOBAL)
        if len(seen_relic_indices) == 0:
            step = float(step_count[b]) if step_count is not None else 0.0
            decay = max(0.2, 1.0 - (step / 500.0)) # Decay over 500 steps
            components["fog_discovery"][b] = (float(delta_visible[b]) * decay) / 50.0

        # 5. Energy Bonus (LOCAL) (Normalized max capacity = 400)
        for u_idx in valid_indices:
             components["energy_gain"][b, u_idx] = (float(delta_energy[b, u_idx]) / 400.0) / 16.0

        # 6. Action-based bonuses/penalties (LOCAL)
        for idx in valid_indices:
            act = int(actions[b, idx])
            if act == 0: # Idle
                if delta_energy[b, idx] <= 0:
                    # Penalty for doing nothing and not farming
                    components["stagnation_penalty"][b, idx] = - 1.0 / 16.0
            elif act < 5: # Moving
                if footprint_map is not None:
                    ux, uy = current_team_pos[b, idx]
                    last_stepped_step = footprint_map[b, ux, uy]
                    steps_since_stepped = float(step_count[b]) - last_stepped_step
                    # The older the tile, the higher the bonus (cap at 50 turns). 
                    # Use max(0.0, ...) to prevent negative bounds when 'step' resets in Match 2!
                    novelty_raw = max(0.0, min(50.0, steps_since_stepped)) / 50.0
                    components["novelty_bonus"][b, idx] = novelty_raw / 16.0

        # Final reward summation (Multipliers act as tunable Hyper-Parameters here)
        shaped_global[b] = (
            components["relic_discovery"][b] * 1.5 +
            components["fog_discovery"][b] * 1.0
        )
        
        for u_idx in range(max_units):
            shaped_local[b, u_idx] = (
                components["local_point_generation"][b, u_idx] * 32.0 + 
                components["collision_penalty"][b, u_idx] * 8.0 +  
                components["dispersion_bonus"][b, u_idx] * (3.0 if len(seen_relic_indices) == 0 else 1.0) +
                components["novelty_bonus"][b, u_idx] * 2.0 +       
                components["relic_proximity"][b, u_idx] * 2.0 +
                components["relic_farming"][b, u_idx] * 3.0 +
                components["energy_gain"][b, u_idx] * 0.8 +
                components["stagnation_penalty"][b, u_idx] * 1.5 +
                components["overcrowding_penalty"][b, u_idx] * 2.0 +
                components["combat_kill"][b, u_idx] * 10.0 - # Big bonus for kills!
                components["combat_death"][b, u_idx] * 10.0  # Big penalty for dying!
            )

    return shaped_global, shaped_local, components
