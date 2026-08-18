# Issue 01: RL Rewards, Potential-Based Shaping (PBRS), GAE Credit Assignment & League Ecosystem

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/3  
**Spec Reference**: [.scratch/kaggriculture-rl-rewards-league/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-rl-rewards-league/spec.md)  

**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  
**Master Architectural Spec**: [docs/issue_05_rl_rewards_credit_and_league_spec.md](file:///C:/Users/Manit/Desktop/docs/issue_05_rl_rewards_credit_and_league_spec.md)  

---

## Summary
Implement the complete Reinforcement Learning (RL) training pipeline for the Kaggriculture championship agent:
1. **Purge Reward Pathologies**: Remove the piecewise $3k bank kink cliff (`growth_bonus` / `cap_loss_penalty`) and flat Monte Carlo trajectory broadcast from `overnight_rl_pipeline.py`.
2. **Maturity-Aware PBRS & GAE**: Integrate `MaturityAwarePBRS` state potential with biological asset discounting and `TrajectoryGAEProcessor` ($\gamma=0.995, \lambda=0.95$) for step-wise credit assignment and peak market arbitrage timing.
3. **1001-Bin Symlog Two-Hot Value Head**: Train categorical cross-entropy value loss over $[-15.0, +15.0]$ symlog space.
4. **500k PER Buffer with Causal Verification**: Log verified Hungarian macro transitions $(s_t, a_{\text{eff}}, r'_t, s_{t+1})$ with binary SumTree proportional sampling ($\alpha=0.6, \beta: 0.4 \to 1.0$).
5. **Prioritized Fictitious Self-Play (PFSP) Single-Champion League**: Dedicate 100% of gradient updates to the Champion bot, matchmaking across 40% checkpoints ($P(i) \propto \max(0.05, (1 - \text{WinRate}_i)^{1.5})$), 40% mirror self-play, and 20% heuristic specialists in `mega_league.py`.
6. **80/20 Fast-Forward Curriculum Ingestion**: Fast-forward 20% of matches in $<150\text{ms}$ using heuristic pairings into Midgame ($t \in [288, 432]$) and Endgame ($t \in [528, 648]$) states for dense liquidation training.
7. **Mask-Safe Exploration & Cosine Entropy Decay**: Embed Dirichlet root noise ($\alpha=0.3, \epsilon=0.25$) over legal action masks and decay policy entropy regularization from $0.05 \to 0.002$.

## Testing Seams
- **Agent Seam**: `agent(obs, config=None)` evaluated on `kaggle_environments.make("kaggriculture")` against standard `"starter"` baseline.
- **GAE & PBRS Seam**: `TrajectoryGAEProcessor.process_match(observations, value_estimates)` verifying GAE advantage calculation, shaped reward telescoping, and value target consistency.
- **Fast-Forward Warmup Seam**: Sub-trajectory simulation starting at $t_{\text{start}} > 0$ validating execution latency ($<200\text{ms}$) and clean GAE advantage recursion.
