import numpy as np

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
