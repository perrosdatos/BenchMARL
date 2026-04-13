import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics():
    # Load the generated CSV
    csv_path = "reward_distribution.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Create a figure with 2 subplots sharing the X-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- Top Plot: Rewards per agent ---
    ax1.plot(df['step'], df['reward_per_agent'], color='purple', linewidth=2, label='Reward per Agent')
    ax1.set_title("TorchRL / MAPPO Scaled Reward per Agent", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Reward Value", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    
    # --- Bottom Plot: Team Points & Win Resets ---
    # We will plot both points and explicitly show the match boundaries
    match_boundaries = []
    prev = -1
    for idx, row in df.iterrows():
        if row['match_step'] < prev:
            match_boundaries.append(row['step'])
        prev = row['match_step']
        
    ax2.plot(df['step'], df['match_points_P0'], color='blue', linewidth=2.5, label='P0 Team Points')
    ax2.plot(df['step'], df['match_points_P1'], color='red', linewidth=1.5, alpha=0.8, linestyle='--', label='P1 Team Points')
    
    # Mark Match Resets
    for idx, bound in enumerate(match_boundaries):
        if bound > 0:
            ax2.axvline(x=bound, color='gray', linestyle=':', alpha=0.8, label='Match Reset' if idx == 0 else "")
            
    ax2.set_title("LuxAI_S3 Native Team Points Accumulation (Rule-Based P0 vs Random P1)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Environment Logical Steps (TorchRL)", fontsize=12)
    ax2.set_ylabel("Team Points", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save directly to the artifacts directory
    output_path = "/home/carlos/.gemini/antigravity/brain/6fb15728-691e-4485-9561-83c6d0e7acc8/reward_plot.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot successfully saved to {output_path}")

if __name__ == "__main__":
    plot_metrics()
