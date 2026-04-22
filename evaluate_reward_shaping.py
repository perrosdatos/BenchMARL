import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
from tensordict import TensorDict

# Import Agent
moth_dir = "/home/carlos/Documents/github/msc_ai_thesis_marl_lux"
if moth_dir not in sys.path:
    sys.path.append(os.path.abspath(moth_dir))

try:
    from rulebased_agent_main import Agent
except ImportError:
    print(f"FATAL ERROR: Could not import Agent from {moth_dir}")
    sys.exit(1)

# Set up BenchMARL environment
sys.path.insert(0, os.path.abspath("."))
from benchmarl.environments.lux.lux_env import LuxTorchRLEnv

def build_pseudo_obs(jax_obs, team_id_num, b):
    """
    Simulates the standard GameState dictionary for rule-based Agent 
    by unpacking the symmetric JAX observation nested structs for ONE specific environment `b`.
    """
    p0 = jax_obs[f"player_{team_id_num}"]
    def get_v(o, k): return getattr(o, k) if hasattr(o, k) else o.get(k)
    
    u_mask = np.asarray(get_v(p0, "units_mask"))[b]
    pos = np.asarray(get_v(get_v(p0, "units"), "position"))[b]
    en = np.asarray(get_v(get_v(p0, "units"), "energy"))[b]
    r_nodes = np.asarray(get_v(p0, "relic_nodes"))[b]
    r_mask = np.asarray(get_v(p0, "relic_nodes_mask"))[b]
    t_pts = np.asarray(get_v(p0, "team_points"))[b]
    
    return {
        "units_mask": u_mask.tolist(),
        "units": {
            "position": pos.tolist(),
            "energy": en.tolist()
        },
        "relic_nodes": r_nodes.tolist(),
        "relic_nodes_mask": r_mask.tolist(),
        "team_points": t_pts.tolist()
    }

def run_evaluation(version):
    BATCH_SIZE = 400
    MAX_STEPS = 150
    MATCH_COUNT = 3
    
    print(f"Initializing {BATCH_SIZE} environments concurrently for reward version '{version}'...")
    env = LuxTorchRLEnv(
        batch_size=BATCH_SIZE, 
        max_steps=MAX_STEPS, 
        match_count=MATCH_COUNT, 
        seed=int(time.time()), 
        reward_version=version
    )
    
    td = env.reset()
    
    env_cfg = {
        "max_units": 16,
        "map_width": 24,
        "map_height": 24
    }
    
    agents_p0 = [Agent(f"player_0", env_cfg) for _ in range(BATCH_SIZE)]
    agents_p1 = [Agent(f"player_1", env_cfg) for _ in range(BATCH_SIZE)]
    
    all_data = []
    
    for step in range(MAX_STEPS):
        sys.stdout.write(f"\rExecuting Step {step+1}/{MAX_STEPS}...")
        sys.stdout.flush()
        
        actions_controlled = np.zeros((BATCH_SIZE, 16), dtype=np.int32)
        actions_opponent = np.zeros((BATCH_SIZE, 16), dtype=np.int32)
        
        t_ids = env.team_ids.numpy() 
        
        for b in range(BATCH_SIZE):
            # Act for P0
            obs_0 = build_pseudo_obs(env.jax_obs, 0, b)
            act_0 = agents_p0[b].act(step, obs_0)[:, 0] # (16,)
            
            # Act for P1
            obs_1 = build_pseudo_obs(env.jax_obs, 1, b)
            act_1 = agents_p1[b].act(step, obs_1)[:, 0] # (16,)
            
            # Route
            if t_ids[b] == 0:
                actions_controlled[b] = act_0
                actions_opponent[b] = act_1
            else:
                actions_controlled[b] = act_1
                actions_opponent[b] = act_0
                
        # Step Environment
        td["agents", "action"] = torch.tensor(actions_controlled, dtype=torch.long, device=env.device).unsqueeze(-1)
        td["action"] = td["agents", "action"]
        env.opp_actions = actions_opponent
        
        td = env.step(td).get("next")
        
        # Extract Output
        comp = env.last_reward_components
        t_reward = td["agents", "reward"].squeeze(-1).sum(-1).numpy()
        
        def safe_get_attr(obj, key):
            return getattr(obj, key) if hasattr(obj, key) else obj.get(key)
            
        p0_mask = np.asarray(safe_get_attr(env.jax_obs["player_0"], "units_mask"))[:, 0, :]
        p1_mask = np.asarray(safe_get_attr(env.jax_obs["player_1"], "units_mask"))[:, 0, :]
        active_counts = np.where(t_ids == 0, p0_mask.sum(-1), p1_mask.sum(-1))
        
        # Use our native Environment hooks to bypass observation wiping!
        controlled_pts = np.asarray(env.max_agent_points)
        opp_pts = np.asarray(env.max_opp_points)
        
        # Determine the total points of the team assigned to this row
        # In our logging structure, we log the Controlled team's points correctly
        team_pts = controlled_pts
        
        for b in range(BATCH_SIZE):
            row = {
                "step": step,
                "env_id": b,
                "team_id": "Team 0" if t_ids[b] == 0 else "Team 1",
                "total_shaped_reward": float(t_reward[b]),
                "active_units_count": int(active_counts[b]),
                "current_team_points": float(team_pts[b]),
                "base_points": float(np.sum(comp["base_points"][b])) if "base_points" in comp else 0.0,
                "local_point_generation": float(np.sum(comp["local_point_generation"][b])) if "local_point_generation" in comp else 0.0,
                "collision_penalty": float(np.sum(comp["collision_penalty"][b])),
                "diagonal_bonus": float(np.sum(comp["diagonal_bonus"][b])) if "diagonal_bonus" in comp else 0.0,
                "dispersion_bonus": float(np.sum(comp["dispersion_bonus"][b])) if "dispersion_bonus" in comp else 0.0,
                "overcrowding_penalty": float(np.sum(comp["overcrowding_penalty"][b])) if "overcrowding_penalty" in comp else 0.0,
                "movement_bonus": float(np.sum(comp["movement_bonus"][b])) if "movement_bonus" in comp else 0.0,
                "novelty_bonus": float(np.sum(comp["novelty_bonus"][b])) if "novelty_bonus" in comp else 0.0,
                "relic_proximity": float(np.sum(comp["relic_proximity"][b])),
                "relic_farming": float(np.sum(comp["relic_farming"][b])),
                "relic_discovery": float(np.sum(comp["relic_discovery"][b])) if "relic_discovery" in comp else 0.0,
                "fog_discovery": float(np.sum(comp["fog_discovery"][b])) if "fog_discovery" in comp else 0.0,
                "energy_gain": float(np.sum(comp["energy_gain"][b])),
                "stagnation_penalty": float(np.sum(comp["stagnation_penalty"][b])),
                "is_terminal": bool(td["done"][b].item())
            }
            if row["is_terminal"]:
                agent_pts = td["info", "agent_points"][b].item()
                opp_pts = td["info", "opponent_points"][b].item()
                row["final_agent_points"] = agent_pts
                row["final_opponent_points"] = opp_pts
                
                if agent_pts > opp_pts:
                    row["winner"] = "Agent"
                elif opp_pts > agent_pts:
                    row["winner"] = "Opponent"
                else:
                    row["winner"] = "Tie"
            else:
                row["final_agent_points"] = np.nan
                row["final_opponent_points"] = np.nan
                row["winner"] = "None"
                
            all_data.append(row)
            
    print("\nSimulation complete. Generating reports...")
    return pd.DataFrame(all_data)

def generate_report(df, version):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"{version}_{timestamp}"
    
    local_data_dir = os.path.abspath(f"../local-data/{folder_name}")
    reports_dir = os.path.abspath(f"../reports/{folder_name}")
    
    os.makedirs(local_data_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    csv_path = os.path.join(local_data_dir, "simulation_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved raw data to {csv_path}")
    
    html_template = """
    <html>
    <head>
        <title>Reward Shape [{version}] Analysis</title>
        <style>
            body {{ font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif; background-color: #0E1117; color: #FAFAFA; margin: 0; padding: 30px; }}
            .tab {{ display: flex; gap: 10px; margin-bottom: 20px; }}
            .tab button {{ flex: 1; background-color: #262730; border: none; border-radius: 8px; cursor: pointer; padding: 15px; font-weight: 600; color: white; transition: 0.2s; }}
            .tab button:hover {{ background-color: #31333F; transform: translateY(-2px); }}
            .tab button.active {{ background-color: #FF4B4B; box-shadow: 0 4px 12px rgba(255,75,75,0.3); }}
            .tabcontent {{ display: none; animation: fadeEffect 0.5s; }}
            @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            h1 {{ text-align: center; color: #FF4B4B; font-size: 2.5rem; font-weight: 800; margin-bottom: 50px; }}
            .plot-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 20px; }}
            .plot-box {{ background: #131722; border-radius: 12px; padding: 15px; width: 48%; border: 1px solid #2A2D35; box-shadow: 0 8px 24px rgba(0,0,0,0.5); box-sizing: border-box; }}
            .plot-box.full {{ width: 100%; }}
            @media(max-width: 1200px) {{ .plot-box {{ width: 100%; }} }}
        </style>
        <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
    </head>
    <body>
        <h1>A/B Reward Shaping Validation ({version_upper})</h1>
        
        <div class="tab">
          <button class="tablinks active" onclick="openTab(event, 'Global')">🌍 Global Distribution</button>
          <button class="tablinks" onclick="openTab(event, 'Team')">⚔️ Team Splits</button>
          <button class="tablinks" onclick="openTab(event, 'WinLoss')">🏆 Win/Loss Correlation</button>
          <button class="tablinks" onclick="openTab(event, 'Agents')">🤖 By Active Agents</button>
          <button class="tablinks" onclick="openTab(event, 'Points')">📈 Team Points</button>
        </div>

        <div id="Global" class="tabcontent" style="display:block;">
            <div class="plot-container">
                {global_plots}
            </div>
        </div>

        <div id="Team" class="tabcontent">
            <div class="plot-container">
                {team_plots}
            </div>
        </div>
        
        <div id="WinLoss" class="tabcontent">
            <div class="plot-container">
                {winloss_plots}
            </div>
        </div>
        
        <div id="Agents" class="tabcontent">
            <div class="plot-container">
                {agents_plots}
            </div>
        </div>
        
        <div id="Points" class="tabcontent">
            <div class="plot-container">
                {points_plots}
            </div>
        </div>

        <script>
        function openTab(evt, tabName) {{
          var i, tabcontent, tablinks;
          tabcontent = document.getElementsByClassName("tabcontent");
          for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].style.display = "none"; }}
          tablinks = document.getElementsByClassName("tablinks");
          for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
          document.getElementById(tabName).style.display = "block";
          evt.currentTarget.className += " active";
        }}
        </script>
    </body>
    </html>
    """
    
    # 1. Global Distribution
    f_global = px.histogram(df, x="total_shaped_reward", nbins=60, 
                            title="Global Total Shaped Reward Distribution (100 Envs)",
                            color_discrete_sequence=["#FF4B4B"], template="plotly_dark")
    
    comp_cols = ["base_points", "local_point_generation", "relic_discovery", "relic_farming", "fog_discovery", "energy_gain", "collision_penalty", 
                 "stagnation_penalty", "diagonal_bonus", "dispersion_bonus", "overcrowding_penalty", "movement_bonus", "novelty_bonus", "relic_proximity"]
    df_melt = df.melt(id_vars=["step", "env_id"], value_vars=comp_cols, var_name="Component", value_name="Value")
    f_comp = px.box(df_melt, x="Component", y="Value", color="Component", 
                    title="Shaping Factors Density", template="plotly_dark")
    
    gl_html = f"<div class='plot-box'>{f_global.to_html(full_html=False, include_plotlyjs=False)}</div>"
    gl_html += f"<div class='plot-box'>{f_comp.to_html(full_html=False, include_plotlyjs=False)}</div>"
    
    # 2. Team Distributions
    f_team = px.histogram(df, x="total_shaped_reward", color="team_id", barmode="overlay", nbins=60,
                          title="Frame-by-Frame Reward per Team", template="plotly_dark",
                          color_discrete_map={"Team 0": "#00CC96", "Team 1": "#AB63FA"})
    
    df_sums = df.groupby(["env_id", "team_id"]).sum(numeric_only=True).reset_index()
    f_team_sum = px.box(df_sums, x="team_id", y="total_shaped_reward", color="team_id",
                         title="Total Episode Sum of Rewards (per Team)", template="plotly_dark",
                         color_discrete_map={"Team 0": "#00CC96", "Team 1": "#AB63FA"})
                         
    tm_html = f"<div class='plot-box'>{f_team.to_html(full_html=False, include_plotlyjs=False)}</div>"
    tm_html += f"<div class='plot-box'>{f_team_sum.to_html(full_html=False, include_plotlyjs=False)}</div>"
    
    # 3. Win Loss
    win_map = df[df["is_terminal"] == True][["env_id", "winner"]]
    df_wl = df.drop(columns=["winner"]).merge(win_map, on="env_id", how="left")
    
    df_decided = df_wl[df_wl["winner"].isin(["Agent", "Opponent"])].copy()
    
    def calc_did_win(row):
        is_p0 = "0" in row["team_id"]
        # In our mapping, 'Agent' means the environment Controlled team won.
        # But wait, env.team_ids mapped randomly!
        # If winner == 'Agent', it means `t_ids[b]` team won.
        # So "Team 0" won if t_ids[b] == 0 AND winner == Agent.
        # Let's simplify. `total_shaped_reward` is logged for the Controlled Agent ONLY in our script.
        # The script logs row from the perspective of the MARL agent's seat.
        if row["winner"] == "Agent": return "Won Episode"
        if row["winner"] == "Opponent": return "Lost Episode"
        return "Unknown"
        
    if not df_decided.empty:
        df_decided["Match_Outcome"] = df_decided.apply(calc_did_win, axis=1)
        f_wl = px.histogram(df_decided, x="total_shaped_reward", color="Match_Outcome", barmode="group",
                            histnorm="probability density", nbins=50,
                            title="Is the Shaped Reward actually rewarding winning plays?", template="plotly_dark",
                            color_discrete_map={"Won Episode": "#00CC96", "Lost Episode": "#EF553B"})
                            
        wl_html = f"<div class='plot-box full'>{f_wl.to_html(full_html=False, include_plotlyjs=False)}</div>"
    else:
        wl_html = "<div class='plot-box full'><h3 style='text-align:center;'>Not enough variance to compute Correlation (All matches tied/zero points!)</h3></div>"
    
    # 4. By Active Agents
    df_agents = df.copy()
    df_agents["active_units_count"] = df_agents["active_units_count"].astype(str)
    
    f_agents = px.box(df_agents, x="active_units_count", y="total_shaped_reward", color="active_units_count",
                      title="Total Shaped Reward Behavior vs Num of Active Agents", template="plotly_dark",
                      color_discrete_sequence=px.colors.sequential.Plasma,
                      category_orders={"active_units_count": [str(i) for i in range(17)]})
    
    ag_html = f"<div class='plot-box full'>{f_agents.to_html(full_html=False, include_plotlyjs=False)}</div>"
    
    comp_cols = ["base_points", "local_point_generation", "relic_farming", "energy_gain", "collision_penalty", 
                 "stagnation_penalty", "diagonal_bonus", "dispersion_bonus", "overcrowding_penalty", "movement_bonus", "novelty_bonus", "relic_proximity"]

    for comp in comp_cols:
        f_comp_ag = px.box(df_agents, x="active_units_count", y=comp, color="active_units_count",
                          title=f"Component: {comp.replace('_', ' ').title()}", template="plotly_dark",
                          color_discrete_sequence=px.colors.sequential.Plotly3,
                          category_orders={"active_units_count": [str(i) for i in range(17)]})
        f_comp_ag.update_layout(showlegend=False)
        ag_html += f"<div class='plot-box'>{f_comp_ag.to_html(full_html=False, include_plotlyjs=False)}</div>"
        
    # Multiline Time Series Plots (Step vs Average Reward by Active Units)
    df_agents_time = df.groupby(["step", "active_units_count"])[["total_shaped_reward"] + comp_cols].mean().reset_index()
    df_agents_time["active_units_count"] = df_agents_time["active_units_count"].astype(str)

    f_ag_time = px.scatter(df_agents_time, x="step", y="total_shaped_reward", color="active_units_count",
                        title="Avg Total Reward vs Step (Grouped by Active Agents)", template="plotly_dark",
                        category_orders={"active_units_count": [str(i) for i in range(17)]})
    ag_html += f"<div class='plot-box full'>{f_ag_time.to_html(full_html=False, include_plotlyjs=False)}</div>"

    for comp in comp_cols:
        f_c_time = px.scatter(df_agents_time, x="step", y=comp, color="active_units_count",
                           title=f"Avg {comp.replace('_', ' ').title()} vs Step", template="plotly_dark",
                           category_orders={"active_units_count": [str(i) for i in range(17)]})
        ag_html += f"<div class='plot-box full'>{f_c_time.to_html(full_html=False, include_plotlyjs=False)}</div>"
    
    # 5. Team Points Progression
    # We aggregate average points at each step across the combinations:
    pts_agg = df.groupby(["step", "team_id"])["current_team_points"].mean().reset_index()
    f_pts = px.line(pts_agg, x="step", y="current_team_points", color="team_id",
                    title="Average Team Points Progression over Episode", template="plotly_dark",
                    color_discrete_map={"Team 0": "#00CC96", "Team 1": "#AB63FA"})
    
    pt_html = f"<div class='plot-box full'>{f_pts.to_html(full_html=False, include_plotlyjs=False)}</div>"
    
    
    final_html = html_template.format(
        version=version,
        version_upper=version.upper(),
        global_plots=gl_html,
        team_plots=tm_html,
        winloss_plots=wl_html,
        agents_plots=ag_html,
        points_plots=pt_html
    )
    
    html_path = os.path.join(reports_dir, f"reward_shaping_report_{version}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)
        
    print(f"Analysis Report successfully generated at:\n => {html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1")
    args = parser.parse_args()
    
    import warnings
    warnings.filterwarnings("ignore")
    
    dfo = run_evaluation(args.version)
    generate_report(dfo, args.version)
