import json
import torch
import numpy as np
import submission

replay_path = r'C:\Users\Manit\Downloads\93966516.json'
with open(replay_path, 'r') as f:
    replay = json.load(f)

steps = replay['steps']
bot = submission.FrontierGrandmasterUnifiedBot()
model = bot.model
model.eval()

print("=" * 80)
print("NEURAL NETWORK RAW PREDICTIONS ACROSS EARLY & MID GAME")
print("=" * 80)

macro_names = submission.MACRO_ACTIONS

for test_step in [0, 24, 72, 120, 240, 360, 480, 600, 700]:
    obs = steps[test_step][0].get('observation', {})
    day = obs.get('day', 0)
    hour = obs.get('hour', 0)
    
    grid_np, scalars_np = submission.encode_observation(obs)
    grid_t = torch.tensor(grid_np, dtype=torch.float32).unsqueeze(0)
    scalars_t = torch.tensor(scalars_np, dtype=torch.float32).unsqueeze(0)
    
    latent, x_spatial, _ = model.extract_features(grid_t, scalars_t)
    logits_t = model.policy_head(latent)
    val_logits_t = model.value_head(latent)
    
    logits = logits_t.squeeze(0).detach().cpu().numpy()
    val = submission.categorical_to_scalar(val_logits_t, is_logits=True).item()
    mask = submission.compute_action_mask(obs)
    
    masked_logits = np.where(mask, logits, -1e9)
    probs = torch.softmax(torch.tensor(masked_logits), dim=-1).numpy()
    
    print(f"\nStep {test_step:>3} (Day {day:>2}, Hr {hour:>2}) | Root Value Est: {val:.2f}:")
    for m_idx in range(len(macro_names)):
        print(f"  [{m_idx}] {macro_names[m_idx]:<30}: raw_logit={logits[m_idx]:>6.2f}, mask={str(mask[m_idx]):<5}, prob={probs[m_idx]*100:>5.1f}%")
