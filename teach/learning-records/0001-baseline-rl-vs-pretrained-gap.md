# Pretrained Supervised Checkpoint Outperforming Online RL Due to Reward Misalignment and Policy Drift

The pretrained AWIL checkpoint achieves $30k–$40k+ terminal cash by imitating grandmaster replays, whereas online RL training degrades performance because Program 2 optimizes relative opponent margin (settling for small wins against weak opponents) and unanchored policy gradients drift away from expert macro strategies. Fixing this requires switching to an absolute cash / net-worth potential in `pbrs_gae.py` and adding an AWAC/KL policy regularization anchor during RL fine-tuning.
