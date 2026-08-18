# Issue 01: Hierarchical ML/SSL/RL Bot Architecture & I/O Pipeline

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/1  
**Spec Reference**: [.scratch/kaggriculture-architecture/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-architecture/spec.md)  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)

---

## Summary
Implement the locked **Hierarchical 2-Tier Architecture**:
1. **Observation Encoder**: $(24, 10, 10)$ spatial grid + $72$-dim scalar economic & market state vector.
2. **Backbone Trunk**: FiLM-modulated ResNet conditioning visual perception on market elasticity, cash flow, and town shop demand.
3. **Decoupled Multi-Head Policy**: Factorized heads for crop heatmaps ($5 \times 10 \times 10$), livestock herd targets (3 dims), workforce recruitment ($0..12$), land unlocks ($[0, 1]$), seed orders (5 dims), and continuous market liquidation ($9 \times [0, 1]$).
4. **Critic Value Heads**: Distributional Two-Hot (64 log-bins) + Auxiliary Win-Probability.
5. **Two-Stage Guardrails & Hungarian Micro Solver**: Cow feed wheat reservation, midnight shed overflow auto-liquidation, and Linear Sum Assignment worker routing.

## Testing Seams
- **Agent Seam**: `agent(obs, config=None)` evaluated on `kaggle_environments.make("kaggriculture")` against standard baseline opponents.
- **Encoder Seam**: `encode_observation(obs) -> (spatial, scalar)` shape & boundary validation.
- **Model Forward Seam**: `network(spatial, scalar, masks)` output tensor dimensional checks.
- **Assignment Seam**: `solve_turn_actions(...)` collision-free dispatch and 1-to-1 task uniqueness.
