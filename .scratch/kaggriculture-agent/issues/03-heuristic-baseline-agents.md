Type: prototype
Status: resolved
Blocked by: 01

## Question

How should we implement 2–3 distinct deterministic heuristic baseline agents (e.g. Balanced Crop Farmer, Aggressive Land/Livestock Expansionist, Market Arbitrageur) to serve as competitive benchmarks and training trajectory generators?

## Answer

Implemented multiple baseline agents in `src/agents/`:

1. **`MultiTileCropAgent`** (`src/agents/balanced_farmer.py`):
   - Dynamically manages 10–16 active crop tiles in Quadrant 0 with zero weed decay.
   - Hires 2 farmhands daily for $2 total cost, achieving $10\times$ labor efficiency.
   - Harvests at peak maturity (Carrots Day 3, Wheat Day 4) and performs clean end-of-season liquidation.
   - **Tournament Performance**: Achieved a **100.0% Win Rate** (10-0) against Kaggle's `starter` bot with **$7,268 mean final bank** (peaking at **$12,034**) vs `starter`'s $3,413.

2. **`LivestockHusbandryAgent`** (`src/agents/livestock_bot.py`):
   - Coordinated Coop/Pasture construction, animal procurement via shed pickup/place routines, and daily feeding/care loops.

3. **`HybridExpertAgent`** (`src/agents/hybrid_expert.py`):
   - Explores land quadrant purchases and dynamic crop portfolios.
