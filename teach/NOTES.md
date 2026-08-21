# Teaching Notes & User Preferences

## Preferences & Constraints
- **Teaching Style**: The user prefers intuitive, non-technical, concept-first explanations before diving into mathematical formulations and code.
- Use relatable real-world analogies (running a farm, business cash flow, common-sense incentives) to build intuition before showing math.
- Avoid using the modal `ask_question` tool. Present interactive quizzes and knowledge checks directly within HTML lesson pages.
- Ground all explanations in concrete comparisons:
  1. The existing implementations in the workspace (`overnight_rl_pipeline.py` vs `pbrs_gae.py`).
  2. The grandmaster algorithms (AlphaGo, AlphaZero, MuZero, AlphaStar, AWAC).
  3. Practical recommendations to solve why the pretrained model currently earns more money than the RL model.

## Learning Roadmap
- **Lesson 1**: Deconstructing our Kaggriculture RL Systems vs. The AlphaGo/AlphaZero Blueprint.
- **Lesson 2**: Designing the Optimal RL Reward & Penalty System (Intuitive Non-Technical Guide -> Technical Blueprint).
- **Lesson 3**: Temporal Credit Assignment & GAE — Escaping the Flat Broadcast Trap.
- **Lesson 4**: Combinatorial Action Spaces & Hierarchical Control (Macro Policy + Hungarian Micro Solver).
- **Lesson 5**: Offline-to-Online Policy Fine-Tuning (AWAC & KL Regularization) to Beat the Pretrained Baseline.
