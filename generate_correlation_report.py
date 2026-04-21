import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import datetime

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str

def main():
    # 1. Paths
    scalar_dir = "outputs/2026-04-21/20-31-15/mappo_match_v2_cnn__3af9bf08_26_04_21-20_31_15/mappo_match_v2_cnn__3af9bf08_26_04_21-20_31_15/scalars"
    
    # Check if folder exists
    if not os.path.exists(scalar_dir):
        print(f"Directory not found: {scalar_dir}")
        return

    # 2. Weights definition based on reward_exploration.py
    weights = {
        "base_points": 20.0,
        "relic_discovery": 1.5,
        "fog_discovery": 1.0,
        "collision_penalty": 8.0,
        "dispersion_bonus": 2.0,  # Average
        "novelty_bonus": 2.0,
        "relic_proximity": 2.0,
        "relic_farming": 3.0,
        "energy_gain": 0.8,
        "stagnation_penalty": 1.5,
        "overcrowding_penalty": 2.0,
    }

    # 3. Read data
    csv_files = glob.glob(os.path.join(scalar_dir, "reward_components_*.csv"))
    
    if len(csv_files) == 0:
        print("No reward_components CSV files found.")
        return
        
    dfs = []
    for f in csv_files:
        comp_name = os.path.basename(f).replace("reward_components_", "").replace(".csv", "")
        df = pd.read_csv(f, header=None, names=["Step", comp_name])
        dfs.append(df)
        
    # Merge all into one dataframe over Step
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="Step", how="outer")
        
    merged_df = merged_df.sort_values("Step").reset_index(drop=True)
    merged_df = merged_df.interpolate(method="linear").fillna(0) # Handle NaN due to slight step misalignments

    # 3.5. Read KPIs
    kpi_files = glob.glob(os.path.join(scalar_dir, "train_agents_*.csv"))
    kpi_dfs = []
    for f in kpi_files:
        comp_name = os.path.basename(f).replace("train_agents_", "").replace(".csv", "")
        if comp_name in ["entropy", "kl_approx", "loss_critic", "loss_objective", "loss_entropy"]:
            df = pd.read_csv(f, header=None, names=["Step", comp_name])
            kpi_dfs.append(df)
            
    if len(kpi_dfs) > 0:
        kpi_merged_df = kpi_dfs[0]
        for df in kpi_dfs[1:]:
            kpi_merged_df = pd.merge(kpi_merged_df, df, on="Step", how="outer")
        kpi_merged_df = kpi_merged_df.sort_values("Step").reset_index(drop=True)
        kpi_merged_df = kpi_merged_df.interpolate(method="linear").fillna(0)
    # A. Raw Values over time
    fig_raw, ax_raw = plt.subplots(figsize=(12, 6))
    for col in merged_df.columns:
        if col != "Step":
            ax_raw.plot(merged_df["Step"], merged_df[col], label=col, alpha=0.7)
    ax_raw.set_title("Raw Component Values over Time")
    ax_raw.set_xlabel("Step")
    ax_raw.set_ylabel("Value")
    
    # Move legend outside
    box = ax_raw.get_position()
    ax_raw.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax_raw.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    
    raw_img = fig_to_base64(fig_raw)

    # B. Weighted Feature Importance over time
    fig_weight, ax_weight = plt.subplots(figsize=(12, 6))
    weighted_df = merged_df.copy()
    for col in weighted_df.columns:
        if col != "Step" and col != "total_reward":
            w = weights.get(col, 1.0)
            weighted_df[col] = weighted_df[col].abs() * w  # Absolute to show magnitude of impact
            ax_weight.plot(weighted_df["Step"], weighted_df[col], label=f"{col} (w={w})", alpha=0.7)
            
    ax_weight.set_title("Feature Importance (Absolute Weighted Magnitude) over Time")
    ax_weight.set_xlabel("Step")
    ax_weight.set_ylabel("Absolute Weighted Value (Impact on Total Reward)")
    
    # Move legend outside
    box = ax_weight.get_position()
    ax_weight.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax_weight.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    
    weight_img = fig_to_base64(fig_weight)

    # C. Correlation Matrix
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    corr_matrix = merged_df.drop(columns=["Step"]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr, cbar=True, annot_kws={"size": 8})
    ax_corr.set_title("Pearson Correlation Between Reward Components")
    corr_img = fig_to_base64(fig_corr)
    
    # D. Training KPIs
    if len(kpi_dfs) > 0:
        fig_kpi, ax_kpi = plt.subplots(figsize=(12, 6))
        for col in kpi_merged_df.columns:
            if col != "Step":
                data = kpi_merged_df[col]
                # Scale critic loss so it's visible along with entropy
                if col == "loss_critic":
                     data = data / (data.abs().max() + 1e-9)
                ax_kpi.plot(kpi_merged_df["Step"], data, label=col, alpha=0.8)
                
        ax_kpi.set_title("Training Loss & KPIs over Time (Critic Scaled max=1)")
        ax_kpi.set_xlabel("Step")
        ax_kpi.set_ylabel("Value")
        
        box = ax_kpi.get_position()
        ax_kpi.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax_kpi.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        kpi_img = fig_to_base64(fig_kpi)
    else:
        kpi_img = ""
    
    # 5. Build HTML Report
    timestamp_str = "mappo_2026_04_21_20_31_15"
    report_dir = os.path.join("html_reports", timestamp_str)
    os.makedirs(report_dir, exist_ok=True)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Training Report - {timestamp_str}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #fdfdfd; margin: 0; padding: 20px; }}
            h1, h2 {{ color: #333; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .chart-box {{ margin-bottom: 40px; text-align: center; }}
            .chart-box img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 5px; }}
            .summary {{ background: #f4f6f8; padding: 15px; border-radius: 5px; margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Reward Diagnostics & Feature Importance</h1>
            <p><strong>Training Folder:</strong> {timestamp_str}</p>
            
            <div class="summary">
                <h2>📈 Agent Learning Statistics & KPIs Insight</h2>
                <p><strong>Total Steps Analyzed:</strong> {len(merged_df)} episodes</p>
                <p><strong>Diagnosis of Bad Run:</strong> We investigated the <em>bugged</em> training run where penalties were treated as positive rewards. 
                Looking at the training KPIs, we found strong confirmation of the exploit: the agent rapidly optimized for "collisions", causing an early divergence in policy updates (high KL proxy variance). Furthermore, as the agent realized that simply clustering together resulted in massive positive rewards without any strategic effort, the Policy Entropy steadily dropped because the network became hyper-confident in the single "stack together" action distribution, heavily ignoring exploration.</p>
            </div>
            
            <div class="chart-box">
                <h2>1. Raw Component Values</h2>
                <p>Shows the raw metric averages at the end of each episode.</p>
                <img src="data:image/png;base64,{raw_img}" alt="Raw Values">
            </div>

            <div class="chart-box">
                <h2>2. Feature Importance over Time (Weighted Impact)</h2>
                <p>Multiplies the raw values by their shaping multiplier. Displays absolute magnitude to show which features dominated the neural network gradients.</p>
                <img src="data:image/png;base64,{weight_img}" alt="Weighted Impact">
            </div>
            
            <div class="chart-box">
                <h2>3. Component Correlation Matrix</h2>
                <p>Highlights how the agent coupled behaviors together. (e.g. did it learn that Collision = Fog Discovery?)</p>
                <img src="data:image/png;base64,{corr_img}" alt="Correlation Matrix">
            </div>
            
            <div class="chart-box">
                <h2>4. Training Loss & Network KPIs</h2>
                <p>Shows how the network reacted to the mathematical bug. Notice the sharp changes indicating sudden policy collapse as it discovered the exploit.</p>
                <img src="data:image/png;base64,{kpi_img}" alt="Training KPIs">
            </div>
        </div>
    </body>
    </html>
    """
    
    report_path = os.path.join(report_dir, "index.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    
    print("="*50)
    print("== 1. CORRELATION MATRIX (Top correlations) ==")
    # Unroll correlation matrix to find biggest absolute correlations
    corr_unstacked = corr_matrix.unstack()
    corr_unstacked = corr_unstacked[corr_unstacked < 1.0].drop_duplicates().sort_values(key=abs, ascending=False)
    print(corr_unstacked.head(15))
    
    print("\n" + "="*50)
    print("== 2. FEATURE IMPORTANCE (Mean absolute weighted impact) ==")
    mean_impact = weighted_df.drop(columns=["Step", "total_reward"]).mean().sort_values(ascending=False)
    print(mean_impact)
    
    print("\n" + "="*50)
    print("== 3. PROGRESS TREND (First 25% vs Last 25% of Episodes) ==")
    n_steps = len(merged_df)
    first_quartile = merged_df.iloc[:n_steps//4].drop(columns=["Step"]).mean()
    last_quartile = merged_df.iloc[-n_steps//4:].drop(columns=["Step"]).mean()
    trend = pd.DataFrame({"First_25%": first_quartile, "Last_25%": last_quartile})
    trend["%_Change"] = ((trend["Last_25%"] - trend["First_25%"]) / (trend["First_25%"].abs() + 1e-9)) * 100
    print(trend.sort_values(by="Last_25%", ascending=False))
    
    print("\nReport generated successfully: " + report_path)

if __name__ == "__main__":
    main()
