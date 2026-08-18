Type: research
Status: resolved

## Question

What are the exact simulation specifications, observation schemas, action commands, market price elasticity equations, crop growth cycles, livestock yields, and Kaggle runtime constraints for Kaggriculture?

## Answer

Comprehensive investigation completed against the primary `kaggle-environments` engine and competition specifications:

1. **Simulation Structure**: 30 in-game days $\times$ 24 turns = 720 discrete turns. Match concludes at turn 719.
2. **Economy & Capital**: Starts with $3,000 cash. Winner is decided purely by highest bank balance at step 719.
3. **Geography & Quadrants**: 10×10 grid divided into four 5×5 quadrants (NW starts unlocked; NE costs $1,000; SW costs $2,000; SE costs $4,000). Central shed at `(4,4)`, `(5,4)`, `(4,5)`, `(5,5)` stores up to 100 items. Excess end-of-day shed inventory is permanently discarded.
4. **Crops**:
   - Wheat (2-day growth, $10 seed, 6 yield, feeds livestock).
   - Carrot (2-day growth, $20 seed, 4 yield, high early liquidity).
   - Tomato (8-day initial, regrows, $50 seed).
   - Strawberry (10-day initial, regrows, $100 seed).
   - Melon (10-day growth, $80 seed, $250 payout, highly volatile market).
   - Unwatered crops turn to weeds at midnight. Fertilizer doubles yield bonuses for 3 days.
5. **Livestock**:
   - Goose (Coop, $300, 1 Wheat/day, produces Eggs with logarithmic glut-proof price curve).
   - Cow (Pasture, $400, 1 Wheat/day, produces Milk with linear price curve, floors at ~76 sales).
   - Sheep (Pasture, $500, 1 Wheat/day, produces Wool with quadratic price curve).
   - Daily `CARE` multiplies yield by $(1 + \text{care\_counter})$. All living animals passively generate 1 Fertilizer daily. Animals escape if unfed for 2 consecutive days.
6. **Labor**:
   - 1 main Farmer + daily hired Farmhands. Daily hiring cost follows Fibonacci scaling: $\text{cost} = \text{fib}(n)$.
7. **Market & Town Shops**:
   - Dynamic inventory pricing pool ($I_0 = 10,000$). Prices decay when flooded.
   - Town shops unlock dynamically and consume market supply every 4 turns, creating periodic price spikes.
8. **Kaggle Evaluation Constraints**:
   - `actTimeout`: 1.0 second per turn, 60.0s banked overage time.
   - Single standalone Python submission file `submission.py` exposing `def agent(obs, config):`.
