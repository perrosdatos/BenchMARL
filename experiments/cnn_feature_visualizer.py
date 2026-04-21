import os
import sys
import glob
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pathlib import Path

# Important: Add benchmarl to path if running sideways
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarl.experiment import Experiment

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str

def find_latest_checkpoint(base_dir):
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoints", "checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {base_dir}")
    # Sort by the number in the filename
    checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("checkpoint_", "").replace(".pt", "")))
    return checkpoints[-1]

def main():
    print("="*50)
    print("CNN Feature Visualizer")
    print("="*50)
    
    # 1. Force CPU and Disable WandB
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["WANDB_MODE"] = "disabled" # Disable weights and biases tracking for evaluation scripts
    device = torch.device("cpu")
    
    # Monkey patch torch.load to always map to CPU
    _original_load = torch.load
    def _cpu_load(*args, **kwargs):
        kwargs['map_location'] = 'cpu'
        return _original_load(*args, **kwargs)
    torch.load = _cpu_load
    
    print("Forcing computation on CPU and patching torch.load...")

    # 2. Paths
    run_dir = "outputs/2026-04-21/20-31-15/mappo_match_v2_cnn__3af9bf08_26_04_21-20_31_15"
    if not os.path.exists(run_dir):
        # Fallback to the known absolute path structure based on previous interactions if possible
        run_dir = "BenchMARL/" + run_dir
        if not os.path.exists(run_dir):
             run_dir = "../" + run_dir.replace("BenchMARL/", "")
             if not os.path.exists(run_dir):
                 print(f"Error: Could not find run directory. Running from {os.getcwd()}")
                 return
    
    config_path = os.path.join(run_dir, "config.pkl")
    latest_ckpt = find_latest_checkpoint(run_dir)
    print(f"Loading experiment config from: {config_path}")
    print(f"Using checkpoint: {latest_ckpt}")

    # 3. Load Experiment
    with open(config_path, "rb") as f:
        task = pickle.load(f)
        task_config = pickle.load(f)
        alg_config = pickle.load(f)
        model_config = pickle.load(f)
        seed = pickle.load(f)
        exp_config = pickle.load(f)
        critic_config = pickle.load(f)
        callbacks = pickle.load(f)

    # Modify exp_config so it doesn't collect grad or run forever
    exp_config.collect_with_grad = False
    exp_config.restore_file = latest_ckpt
    exp_config.train_device = "cpu"
    exp_config.sampling_device = "cpu"
    
    print("Reconstructing Experiment and Loading Weights...")
    experiment = Experiment(
        task=task,
        algorithm_config=alg_config,
        model_config=model_config,
        seed=seed,
        config=exp_config,
        critic_model_config=critic_config
    )
    
    # 4. Rollout for 100 steps
    print("Rolling out environment for 100 steps to fill the map (Using Random Policy to avoid Stagnation)...")
    # Rollout manually using policy=None (Random Actions) to force environmental diversity
    try:
        td = experiment.test_env.rollout(max_steps=100, policy=None, break_when_any_done=True)
        # Select the last step observations
        last_step_td = td[..., -1]
    except Exception as e:
        print(f"Error in rollout: {e}")
        return

    # Render Environment if possible
    env_img_b64 = None
    try:
        print("Rendering environment image at T=100...")
        # Lux env uses _render() returning RGB array or env.render()
        if hasattr(experiment.test_env, "render"):
            img = experiment.test_env.render()
            if img is not None:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title("Environment T=100")
                env_img_b64 = fig_to_base64(fig)
        else:
             print("Env does not have render method natively exposed here.")
    except Exception as e:
        print(f"Failed to render env: {e}")

    # 5. Extract input observation channels
    raw_obs = last_step_td.get(("agents", "observation"))
    if raw_obs is None:
        raw_obs = last_step_td.get("observation")
    
    if raw_obs is not None:
        # Squeeze batch if present
        while len(raw_obs.shape) > 3:
            raw_obs = raw_obs[0]
        raw_obs = raw_obs.cpu().numpy() # [Channels, W, H]
    else:
        print("Warning: Could not find raw observations in tensordict keys:", last_step_td.keys())
        raw_obs = np.zeros((16, 10, 10))

    num_input_channels = raw_obs.shape[0]
    print(f"Detected {num_input_channels} input observation channels.")

    # 6. Extract features with PyTorch hooks
    print("Registering hooks and running forward pass...")
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook

    hook_handles = []
    for name, module in experiment.policy.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            h = module.register_forward_hook(get_activation(name))
            hook_handles.append(h)

    with torch.no_grad():
        _ = experiment.policy(last_step_td)

    for h in hook_handles:
        h.remove()
        
    print(f"Captured {len(activations)} convolutional layers.")

    # 7. Generate Convolution Means (Once)
    print("Generating Convolution Spatial Means...")
    conv_htmls = ""
    for layer_name, act in activations.items():
        while len(act.shape) > 3:
            act = act[0] 
        if len(act.shape) < 3: continue
        
        # Calculate Mean Activation across all filters in the layer and transpose
        mean_act = np.mean(act, axis=0).T 
        
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(mean_act, cmap='magma', aspect='equal')
        ax.axis('off')
        ax.set_title(f"Mean Spatial Attention", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        layer_mean_b64 = fig_to_base64(fig)
        conv_htmls += f'''
        <div style="display:inline-block; margin:10px; border:1px solid #444; border-radius:8px; padding:10px; background:#2a2a2a; width: 45%;">
            <h4 style="margin-bottom:5px; color:#fdfdfd;">Layer: {layer_name}</h4>
            <p style="font-size:12px; margin-top:0; color:#aaa;">Shape: {mean_act.shape} ({act.shape[0]} Filters)</p>
            <img src="data:image/png;base64,{layer_mean_b64}" alt="{layer_name}" style="width: 100%; border-radius: 5px;"/>
        </div>
        '''

    # 8. Generate HTML Tabs per Input Channel
    print("Generating HTML maps (Tabbed UI by Input Channel)...")
    
    CHANNEL_NAMES = [
        "My Units Map", "Enemy Units", "My Energy", "Map Energy", 
        "Nebula Tiles", "Asteroids", "Sensor Fog of War", "Relic Nodes",
        "Self Indicator", "Ghost Coordinates (Old Enemies)", "Match Timeline", "Score Differential",
        "Memory Stigmergy (Pheromones)", "Agent Trajectory", "Relic Memory", "Points Delta"
    ]

    tab_buttons_html = ""
    tab_content_html = ""
    
    for ch in range(num_input_channels):
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"Unknown"
        
        # Generate Input Channel Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        # Transpose (.T) to align the observation matrix visually with Pygame Y-down rendering
        im = ax.imshow(raw_obs[ch].T, cmap='cividis')
        ax.set_title(f"{ch_name}\n(Raw Input Channel {ch} - Shape: {raw_obs[ch].T.shape})")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        input_ch_b64 = fig_to_base64(fig)
        
        tab_id = f"input_ch_{ch}"
        active_class = " active" if ch == 0 else ""
        display_style = "block" if ch == 0 else "none"
        tab_buttons_html += f'<button class="tablinks{active_class}" onclick="openLayer(event, \'{tab_id}\')" title="Channel {ch}">Ch {ch}: {ch_name}</button>\n'
        
        tab_content_html += f'''
        <div id="{tab_id}" class="tabcontent" style="display:{display_style};">
            <div class="grid">
                <div class="sidebar">
                    <h2>Frozen Environment Snapshot</h2>
                    { f'<img src="data:image/png;base64,{env_img_b64}" alt="Env"/>' if env_img_b64 else '<p><i>Render missing</i></p>' }
                    <hr>
                    <h2>Network Input Insight</h2>
                    <p>Visualizing what information is fed into <strong>Channel {ch} ({ch_name})</strong> before reaching the neural network. Properly aligned to map rotation.</p>
                    <img src="data:image/png;base64,{input_ch_b64}" alt="Input Channel {ch}"/>
                </div>
                <div class="main-content">
                    <h2>Aggregated Convolutional Attention</h2>
                    <p>Filters across each convolution layer have been averaged to highlight the core spatial areas that dominate the network's processing.</p>
                    {conv_htmls}
                </div>
            </div>
        </div>
        '''

    reports_dir = os.path.join("html_reports", "cnn_visuals")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "index.html")
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CNN Feature Maps Interpretability - Team 0 (Input Channels)</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a1a; color: #fdfdfd; margin: 0; padding: 20px; }}
            h1, h2, h3, p {{ color: #4DA8DA; font-weight: 300; }}
            p {{ color: #ccc; }}
            .container {{ max-width: 1500px; margin: auto; background: #262626; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }}
            
            /* Tabs */
            .tab {{ overflow: hidden; background-color: #333; border-radius: 8px 8px 0 0; display: flex; flex-wrap: wrap; }}
            .tab button {{ background-color: inherit; border: none; outline: none; cursor: pointer; padding: 12px 18px; transition: 0.3s; color: white; font-size: 15px; border-right: 1px solid #444; border-bottom: 1px solid #444; }}
            .tab button:hover {{ background-color: #444; }}
            .tab button.active {{ background-color: #4DA8DA; color: #1a1a1a; font-weight: bold; }}
            .tabcontent {{ padding: 25px; background-color: #222; border-radius: 0 0 8px 8px; }}
            
            .grid {{ display: flex; gap: 30px; }}
            .sidebar {{ flex: 1.2; text-align: center; background: #333; padding: 20px; border-radius: 8px; align-self: flex-start; position: sticky; top: 20px; }}
            .sidebar img {{ max-width: 100%; border-radius: 8px; margin-bottom: 15px; border: 1px solid #555; }}
            .main-content {{ flex: 2.8; text-align: center; background: #333; padding: 20px; border-radius: 8px; }}
            hr {{ border-color: #555; margin: 20px 0; }}
        </style>
        <script>
        function openLayer(evt, layerName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].style.display = "none"; }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
            document.getElementById(layerName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Lux AI Spatial Interpretability - Team 0 (Agent 0)</h1>
            <p><strong>Source Checkpoint:</strong> {latest_ckpt}</p>
            <p>Select an <strong>Input Channel</strong> to visualize the raw tensor data entering the network. The side-by-side view compares the explicit channel matrix with the agent's internal convolutional attention filters.</p>
            
            <div class="tab">
                {tab_buttons_html}
            </div>
            {tab_content_html}
        </div>
    </body>
    </html>
    """
    
    with open(report_path, "w") as f:
        f.write(html_template)
        
    print(f"Success! Visualizer HTML generated at: {report_path}")

if __name__ == "__main__":
    main()
