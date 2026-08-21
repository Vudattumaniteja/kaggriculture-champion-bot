# Mathematical PBRS Formulation, 80/20 Hybrid Potential, and AWAC KL Anchoring

Reward shaping preserves policy invariance strictly when non-cash asset values decay to zero at turn 720, guaranteeing that the shaped return telescopes to terminal cash. Switching from pure relative margin to an 80/20 hybrid potential (80% absolute farm net worth + 20% relative lead) prevents low-cash satisficing, while adding an AWAC advantage-weighted policy loss with a KL anchor to the pretrained checkpoint prevents policy collapse during RL fine-tuning.
