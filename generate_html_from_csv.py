import os
import sys
import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def generate_html_report(csv_path):
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check if necessary columns exist
    expected_cols = ["match_id", "model_team_id", "model_won", "model_points", "opp_points", "model_reward", "steps"]
    for c in expected_cols:
        if c not in df.columns:
            print(f"Error: expected column {c} not found in the CSV.")
            return

    # Base Template
    html_template = """
    <html>
    <head>
        <title>MAPPO Model vs Rulebased Evaluation</title>
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
        <h1>Model vs Rulebased Performance Report</h1>
        
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
    
    # 2. Points
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
    
    points_html = f"<div class='plot-box'>{f_pts_hist.to_html(full_html=False, include_plotlyjs=False)}</div>"
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
        outcomes_plots=outcomes_html,
        points_plots=points_html,
        rewards_plots=rewards_html,
        bias_plots=bias_html
    )
    
    html_path = csv_path.replace("_data.csv", "_html_visuals.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)
        
    print(f"\nHTML Validation Report successfully generated at:\n => {html_path}")

if __name__ == "__main__":
    reports_dir = "/home/carlos/Documents/github/msc_ai_thesis_experiments/reports/"
    csv_files = glob.glob(os.path.join(reports_dir, "validation_v2_report_*_data.csv"))
    
    if not csv_files:
        print("Waiting for CSV to be produced by evaluate_150_matches.log... No CSVs found yet.")
    else:
        latest_csv = max(csv_files, key=os.path.getmtime)
        generate_html_report(latest_csv)
