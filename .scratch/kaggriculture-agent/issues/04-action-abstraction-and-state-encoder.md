Type: grilling
Status: resolved
Blocked by: 01

## Question

How should the observation state be encoded into neural feature tensors (grid spatial maps, temporal scalars, market pricing vectors) and how should the macro-action space be structured to bridge the neural policy network with the low-level chore engine?

## Answer

Implemented in `src/models/encoder.py`:

1. **State Encoder (`encode_observation`)**:
   - **Spatial Grid Tensor $(11, 10, 10)$**: Channels for tile unlocked status, plant presence, crop type, normalized growth age, watering status, harvest yield readiness, structure types (Coop/Pasture), animal types, weed presence, farmer position, and farmhand counts.
   - **Global Economic Scalar Vector $(32,)$**: Normalized bank balance, opponent balance, net worth differential, step/day/hour progress, daily hires, shed inventory counts, seed stocks, and dynamic commodity market prices.

2. **Hierarchical Macro-Action Vocabulary (9 Actions)**:
   - `0: FARM_CARROTS_INTENSIVE`
   - `1: FARM_WHEAT_EXPANSION`
   - `2: FARM_DIVERSIFIED`
   - `3: BUY_LAND_EXPANSION`
   - `4: HIRE_EXTRA_LABOR`
   - `5: BUILD_GOOSE_COOP`
   - `6: BUILD_SHEEP_PASTURE`
   - `7: HARVEST_AND_LIQUIDATE_ALL`
   - `8: HOLD_AND_PASS`
   - Macro-actions are received by the deterministic low-level worker assignment engine, which handles unit routing, daily watering, weed clearing, and market trade placement with zero illegal move risk.
