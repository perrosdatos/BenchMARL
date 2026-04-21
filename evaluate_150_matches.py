import os
import sys
import time
import numpy as np
import jax
import torch
import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
import pandas as pd
import plotly.express as px
import random
import argparse

# Explicitly disable WandB mapping globally
os.environ["WANDB_MODE"] = "disabled"

# Set global deterministic flags
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# the rulebased agent is completely integrated inside BenchMARL lux_env in v2!

# Import BenchMARL loader
from benchmarl.hydra_config import load_experiment_from_hydra

def generate_html_report(df, csv_path, step_points_history, algo_name):
    print(f"\nGenerating interactive HTML report based strictly on the DataFrame...")
    
    html_template = """
    <html>
    <head>
        <title>{algo_name} Model vs Rulebased Evaluation</title>
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
        <h1>{algo_name} Model vs Rulebased Performance Report</h1>
        
        <div class="tab">
          <button class="tablinks active" onclick="openTab(event, 'Outcomes')">🏆 Match Outcomes</button>
          <button class="tablinks" onclick="openTab(event, 'Points')">🎯 Points Distribution</button>
          <button class="tablinks" onclick="openTab(event, 'Rewards')">💎 Shaped Rewards</button>
          <button class="tablinks" onclick="openTab(event, 'Bias')">⚖️ Team Allocation Bias</button>
        </div>

        <div id="Outcomes" class="tabcontent" style="display:block;">
            <div class="plot-container">
                {outcomes_plots}
            </div>
        </div>

        <div id="Points" class="tabcontent">
            <div class="plot-container">
                {points_plots}
            </div>
        </div>
        
        <div id="Rewards" class="tabcontent">
            <div class="plot-container">
                {rewards_plots}
            </div>
        </div>
        
        <div id="Bias" class="tabcontent">
            <div class="plot-container">
                {bias_plots}
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
    
    # 1. Outcomes
    df["Result"] = df["model_won"].apply(lambda x: "Model Won" if x == 1 else "Opponent Won or Tie")
    f_win = px.pie(df, names="Result", title="Global Match Outcomes Distribution", color="Result",
                   color_discrete_map={"Model Won": "#00CC96", "Opponent Won or Tie": "#EF553B"},
                   template="plotly_dark")
    
    f_steps = px.histogram(df, x="steps", nbins=40, title="Match Length (Steps) Distribution", 
                           template="plotly_dark", color_discrete_sequence=["#FF4B4B"])
    
    outcomes_html = f"<div class='plot-box'>{f_win.to_html(full_html=False, include_plotlyjs=False)}</div>"
    outcomes_html += f"<div class='plot-box'>{f_steps.to_html(full_html=False, include_plotlyjs=False)}</div>"
    
    # 2. Points (with Step Evolution Plot!)
    df_pts_model = df[["match_id", "model_points"]].rename(columns={"model_points": "Points"})
    df_pts_model["Entity"] = "Model"
    df_pts_opp = df[["match_id", "opp_points"]].rename(columns={"opp_points": "Points"})
    df_pts_opp["Entity"] = "Rulebased Opponent"
    df_pts = pd.concat([df_pts_model, df_pts_opp])
    
    f_pts_hist = px.histogram(df_pts, x="Points", color="Entity", barmode="overlay", nbins=40,
                              title="End-of-Match Team Points Overlay", template="plotly_dark",
                              color_discrete_map={"Model": "#00CC96", "Rulebased Opponent": "#AB63FA"})
                              
    f_pts_box = px.box(df_pts, x="Entity", y="Points", color="Entity",
                       title="Points Variance", template="plotly_dark",
                       color_discrete_map={"Model": "#00CC96", "Rulebased Opponent": "#AB63FA"})
                       
    # Prepare Line Chart for Averages
    max_steps_found = max(len(m[0]) for m in step_points_history) if step_points_history else 0
    avg_m_pts_step = np.zeros(max_steps_found)
    avg_o_pts_step = np.zeros(max_steps_found)
    counts = np.zeros(max_steps_found)
    
    for m_hist, o_hist in step_points_history:
        for i in range(len(m_hist)):
            avg_m_pts_step[i] += m_hist[i]
            avg_o_pts_step[i] += o_hist[i]
            counts[i] += 1
            
    counts[counts == 0] = 1
    avg_m_pts_step /= counts
    avg_o_pts_step /= counts
    
    df_pts_time = pd.DataFrame({
        "step": range(max_steps_found),
        "Model Avg Points": avg_m_pts_step,
        "Rulebased Avg Points": avg_o_pts_step
    })
    
    f_pts_line = px.line(df_pts_time, x="step", y=["Model Avg Points", "Rulebased Avg Points"],
                         title="Average Team Points Progression over Match Steps", template="plotly_dark",
                         color_discrete_map={"Model Avg Points": "#00CC96", "Rulebased Avg Points": "#AB63FA"})
    
    points_html = f"<div class='plot-box full'>{f_pts_line.to_html(full_html=False, include_plotlyjs=False)}</div>"
    points_html += f"<div class='plot-box'>{f_pts_hist.to_html(full_html=False, include_plotlyjs=False)}</div>"
    points_html += f"<div class='plot-box'>{f_pts_box.to_html(full_html=False, include_plotlyjs=False)}</div>"
    
    # 3. Rewards
    f_rew = px.histogram(df, x="model_reward", nbins=50, title="Model Accumulated Match Reward", 
                         template="plotly_dark", color_discrete_sequence=["#636EFA"])
                         
    f_rew_pts = px.scatter(df, x="model_points", y="model_reward", color="Result",
                           title="Shaped Reward vs Earned Points Correlation", template="plotly_dark",
                           color_discrete_map={"Model Won": "#00CC96", "Opponent Won or Tie": "#EF553B"})
                           
    rewards_html = f"<div class='plot-box'>{f_rew.to_html(full_html=False, include_plotlyjs=False)}</div>"
    rewards_html += f"<div class='plot-box'>{f_rew_pts.to_html(full_html=False, include_plotlyjs=False)}</div>"
    
    # 4. Bias (Team Allocations)
    df["Team Assignment"] = df["model_team_id"].apply(lambda x: f"Team {x}")
    df_bias_win = df.groupby("Team Assignment")["model_won"].mean().reset_index()
    df_bias_win["Win Rate %"] = df_bias_win["model_won"] * 100
    
    f_bias = px.bar(df_bias_win, x="Team Assignment", y="Win Rate %", color="Team Assignment",
                    title="Win Rate Dependent on Assigned Starter Position", template="plotly_dark",
                    color_discrete_map={"Team 0": "#FFA15A", "Team 1": "#19D3F3"})
                    
    df_pts_bias = df.groupby("Team Assignment")[["model_points", "model_reward"]].mean().reset_index()
    f_bias_pts = px.bar(df_pts_bias, x="Team Assignment", y="model_points", color="Team Assignment",
                        title="Average Points Earned by Starter Position", template="plotly_dark",
                        color_discrete_map={"Team 0": "#FFA15A", "Team 1": "#19D3F3"})
                        
    bias_html = f"<div class='plot-box'>{f_bias.to_html(full_html=False, include_plotlyjs=False)}</div>"
    bias_html += f"<div class='plot-box'>{f_bias_pts.to_html(full_html=False, include_plotlyjs=False)}</div>"

    final_html = html_template.format(
        algo_name=algo_name.upper(),
        outcomes_plots=outcomes_html,
        points_plots=points_html,
        rewards_plots=rewards_html,
        bias_plots=bias_html
    )
    
    html_path = csv_path.replace("_data.csv", "_html_visuals.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)
        
    print(f"Rich HTML Validation Report successfully generated at:\n => {html_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Absolute path to the PyTorch checkpoint .pt file")
    parser.add_argument("--algo", type=str, required=True,
                        help="Algorithm used (e.g. 'mappo' or 'qmix')")
    # Using parse_known_args in case test runners pass other flags
    args, _ = parser.parse_known_args()
    
    checkpoint_path = args.checkpoint
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    benchmarl_conf_path = "benchmarl/conf"
    
    with hydra.initialize(version_base=None, config_path=benchmarl_conf_path):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"algorithm={args.algo}",
                "task=lux/match_v2",
                "model=layers/cnn_lux_16ch",
                "model@critic_model=layers/cnn_lux_16ch",
                "experiment.sampling_device=cpu",
                "experiment.train_device=cpu",
                "experiment.buffer_device=cpu",
                "experiment.checkpoint_interval=120000",
                "experiment.loggers=[]", # Completely disables wandb initialization in hydra dict config
                "seed=42"  # Stricts benchmarl and torchrl instantiation to exact same environments every run
            ],
        )

        print(f"\nLoading initialized {args.algo.upper()} network architecture...")
        experiment = load_experiment_from_hydra(cfg, task_name="lux/match_v2")
        
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            experiment.load_state_dict(state_dict)
        else:
            print(f"Error: Checkpoint {checkpoint_path} not found.")
            return
            
        policy = experiment.algorithm.get_policy_for_collection()
        env = experiment.test_env

        env_wrapper = env
        while hasattr(env_wrapper, "env"):
             if hasattr(env_wrapper, "opp_actions"):
                 break
             env_wrapper = env_wrapper.env
             
        base_lux = env_wrapper if hasattr(env_wrapper, "opp_actions") else env.base_env
        batch_size = base_lux.batch_size[0] if base_lux.batch_size else 1


        total_matches = 150
        matches_played = 0
        match_results = []
        step_points_history = []
        
        print(f"\nStarting {total_matches} evaluation matches. Batch Size Evaluated Sequentially: {batch_size}")
        
        while matches_played < total_matches:
            sys.stdout.write(f"\rMatches Completed: {matches_played}/{total_matches} | Calculating step actions...")
            sys.stdout.flush()
            
            td = env.reset()
            step_obj = 0
            
            model_teams = np.asarray(base_lux.team_ids)
            ep_rewards = [0.0 for _ in range(batch_size)]
            
            # Step Arrays
            current_ep_points_model = [[] for _ in range(batch_size)]
            current_ep_points_opponent = [[] for _ in range(batch_size)]

            while True:
                # MAPPO Action 
                with torch.no_grad():
                    from torchrl.envs.utils import set_exploration_type
                    from torchrl.envs.utils import ExplorationType
                    with set_exploration_type(ExplorationType.DETERMINISTIC): # Greedy decision making
                        td = policy(td)
                
                # Capture real team points from state right before any potential wiping occurs in env.step
                current_state_pts = np.asarray(base_lux.env_state.team_points)
                
                # We do NOT assign base_lux.opp_actions manually!
                # lux_env.py explicitly calculates opponent actions natively under the hood during env.step()!
                td = env.step(td)
                
                dones = td.get(("next", "done")).squeeze(-1)
                rewards = td.get(("next", "agents", "reward")).squeeze(-1) 
                
                for b in range(batch_size):
                    ep_rewards[b] += float(torch.sum(rewards[b]).item())
                    
                    m_team = model_teams[b]
                    current_ep_points_model[b].append(current_state_pts[b, m_team].item())
                    current_ep_points_opponent[b].append(current_state_pts[b, 1 - m_team].item())
                
                step_obj += 1
                td = td.get("next")
                
                if dones.any(): 
                    break
            
            sys.stdout.write(f"\rMatches Completed: {matches_played}/{total_matches} | Finalizing Episode Batch...   ")
            sys.stdout.flush()

            for b in range(batch_size):
                if matches_played >= total_matches:
                    break
                    
                m_team = model_teams[b]
                o_team = 1 - m_team
                
                # Because TorchRL AutoResetEnv wipes the final env state upon 'done',
                # we grab the points dynamically from the final stored historical tick just before reset
                m_pts = current_ep_points_model[b][-1] if current_ep_points_model[b] else 0
                o_pts = current_ep_points_opponent[b][-1] if current_ep_points_opponent[b] else 0
                
                is_win = 1 if m_pts > o_pts else 0 
                
                match_results.append({
                    "match_id": matches_played,
                    "model_team_id": m_team,
                    "model_won": is_win,
                    "model_points": m_pts,
                    "opp_points": o_pts,
                    "model_reward": ep_rewards[b],
                    "steps": step_obj
                })
                
                step_points_history.append((current_ep_points_model[b], current_ep_points_opponent[b]))
                
                matches_played += 1
                
            sys.stdout.write(f"\rMatches Completed: {matches_played}/{total_matches} | Ready!                        \n")
            sys.stdout.flush()

        print("\nAll matches completed.")
        
        reports_dir = "/home/carlos/Documents/github/msc_ai_thesis_experiments/reports"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_folder = os.path.join(reports_dir, f"validation_v2_report_{args.algo.lower()}_{timestamp}")
        os.makedirs(report_folder, exist_ok=True)
        report_prefix = os.path.join(report_folder, "report")
        
        df = pd.DataFrame(match_results)
        csv_path = f"{report_prefix}_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved exact metric data to CSV at: {csv_path}")
        
        # Trigger dynamic rich report directly
        generate_html_report(df, csv_path, step_points_history, args.algo)

if __name__ == "__main__":
    main()
