"""
Model Inspector - Diagnose model input requirements
"""

import torch
import sys
sys.path.append("/Users/shahi/Developer/Project-highD/src")
from models import ImprovedTrajectoryTransformer, SimpleSLSTM

MODEL_PATH = "/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt"
MODEL_TYPE = "slstm"

device = torch.device("cuda" if torch.cuda.is_available() 
                     else "mps" if torch.backends.mps.is_available() 
                     else "cpu")

print("="*60)
print("MODEL INSPECTOR")
print("="*60)

# Load model
if MODEL_TYPE == "slstm":
    model = SimpleSLSTM(pred_len=25)
else:
    model = ImprovedTrajectoryTransformer(pred_len=25)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print(f"\n✓ Loaded {MODEL_TYPE.upper()} model\n")

# Inspect model structure
print("Model Architecture:")
print("-" * 60)
for name, module in model.named_children():
    print(f"{name}: {module.__class__.__name__}")
    if hasattr(module, 'input_size'):
        print(f"  → input_size: {module.input_size}")
    if hasattr(module, 'hidden_size'):
        print(f"  → hidden_size: {module.hidden_size}")

print("\n" + "=" * 60)
print("Testing different input configurations...")
print("=" * 60)

obs_len = 20
pred_len = 25

# Test configurations
test_configs = [
    {"nd_shape": (1, 20, 5, 7), "ns_shape": (1, 5, 2), "desc": "Original (5 neighbors)"},
    {"nd_shape": (1, 20, 8, 7), "ns_shape": (1, 8, 2), "desc": "8 neighbors"},
    {"nd_shape": (1, 20, 4, 7), "ns_shape": (1, 4, 2), "desc": "4 neighbors"},
    {"nd_shape": (1, 20, 10, 7), "ns_shape": (1, 10, 2), "desc": "10 neighbors"},
]

# Create test observation
obs = torch.randn(1, obs_len, 7).to(device)
lane = torch.zeros(1, 3).to(device)

print(f"\nObservation shape: {obs.shape}")
print(f"Lane shape: {lane.shape}\n")

for config in test_configs:
    nd = torch.zeros(config["nd_shape"]).to(device)
    ns = torch.zeros(config["ns_shape"]).to(device)
    
    print(f"\nTesting: {config['desc']}")
    print(f"  nd shape: {nd.shape}")
    print(f"  ns shape: {ns.shape}")
    
    try:
        with torch.no_grad():
            if hasattr(model, "multi_att"):  # Transformer
                last_obs_pos = obs[:, -1, :2]
                pred = model(obs, nd, ns, lane, last_obs_pos=last_obs_pos)
            else:  # SLSTM
                pred = model(obs, nd, ns, lane)
        
        print(f"  ✓ SUCCESS! Output shape: {pred.shape}")
        print(f"  → USE THIS CONFIGURATION!")
        break
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:100]}")
        continue

print("\n" + "=" * 60)

# Also check what's in the model's forward method
print("\nModel forward method signature:")
print("-" * 60)
import inspect
sig = inspect.signature(model.forward)
print(f"def forward{sig}")

print("\n" + "=" * 60)
print("Inspection complete!")
print("=" * 60)