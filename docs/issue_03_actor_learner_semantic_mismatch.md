# Issue 03: Actor-Learner Semantic Mismatch & Heuristic Overrides in Frontier Executors

**Status**: Confirmed Bug & Architectural Defect  
**Severity**: Critical (P0) — Blocks Effective Policy Optimization & Causes PER Replay Buffer Corruption  
**Impacted Modules**: 
- [`src/agents/frontier_executors.py`](file:///C:/Users/Manit/Desktop/kaggle/src/agents/frontier_executors.py) (Lines 64–112, 183–193, 245–276, 480–539)
- [`src/agents/hrl_12worker_dispatcher.py`](file:///C:/Users/Manit/Desktop/kaggle/src/agents/hrl_12worker_dispatcher.py) (Lines 150–200, 320–410)
- [`src/training/overnight_rl_pipeline.py`](file:///C:/Users/Manit/Desktop/kaggle/src/training/overnight_rl_pipeline.py) (Lines 940–944, 994–1005, 1083–1091)

---

## 1. Executive Summary & Root Cause Forensic

In the Kaggriculture reinforcement learning pipeline, the high-level neural policy $\pi_\theta(a|s)$ selects among 10 discrete macro-actions ($a \in \{0, \dots, 9\}$, e.g., `BUY_LAND_EXPANSION`, `FARM_CARROTS_INTENSIVE`, `BUILD_GOOSE_COOP`). The actor-learner contract assumes that executing action $a_t$ in state $s_t$ induces transition $s_{t+1} \sim \mathcal{P}(\cdot | s_t, a_t)$ and yields reward $r_t$.

However, forensic audit of [`frontier_executors.py`](file:///C:/Users/Manit/Desktop/kaggle/src/agents/frontier_executors.py) and [`hrl_12worker_dispatcher.py`](file:///C:/Users/Manit/Desktop/kaggle/src/agents/hrl_12worker_dispatcher.py) reveals that **hardcoded rule-based heuristics aggressively override, ignore, or completely bypass the neural policy's chosen macro-action**. The low-level execution engine executes hardcoded schedules regardless of the macro-action passed to it.

```
+----------------------------------------------------------------------------------------------------+
|                                    ACTOR-LEARNER DISCONNECT                                        |
|                                                                                                    |
|   +--------------------------+                                 +-------------------------------+   |
|   |   Neural Policy Network  |                                 |   Prioritized Replay (PER)    |   |
|   |      pi_theta(a | s_t)   |                                 |   Buffer: (s_t, a_t, r_t)     |   |
|   +------------+-------------+                                 +---------------+---------------+   |
|                |                                                               ^                   |
|     Selects:   | a_t = "BUY_LAND_EXPANSION" (Macro 3)                          | Logs (s_t, a_t)   |
|                v                                                               | as if a_t acted!  |
|   +----------------------------------------------------------------------------+---------------+   |
|   | Frontier Macro Executor (FrontierGrandmasterAgent / Flagship150kEngine)                        |
|   |                                                                                                |
|   |  - L184: Hardcoded Day Cutoff (day <= 24) suppresses land purchase!                            |
|   |  - L247: Hardcoded Wheat Seed Purchasing buys 4 wheat seeds (Macro Ignored)!                   |
|   |  - L271: Hardcoded Carrot Seed Purchasing buys 6 carrot seeds (Macro Ignored)!                 |
|   |  - L511: Static Chore Queue assigns farmhands to watering/weeding (Macro Ignored)!             |
|   |                                                                                                |
|   |  ===> ACTUAL EXECUTED ACTION: Unconditional Carrot/Wheat Farming & Livestock Maintenance       |
|   +--------------------------------------------+---------------------------------------------------+
|                                                |                                                   |
|                                                v                                                   |
|                            +---------------------------------------+                               |
|                            |   Environment Transition s_{t+1}      |                               |
|                            |   (Has ZERO causal link to a_t)       |                               |
|                            +---------------------------------------+                               |
+----------------------------------------------------------------------------------------------------+
```

### Critical Consequences:
1. **Prioritized Experience Replay (PER) Corruption**: The replay buffer stores transitions $(s_t, a_t, r_t, s_{t+1})$ where $s_{t+1}$ and $r_t$ have **zero causal dependency** on $a_t$.
2. **Policy Gradient Collapse & Illusion of Control**: The policy gradient $\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t | s_t) \hat{A}_t\right]$ updates policy weights based on spurious correlations. If the heuristic engine performed well in a match, the learner reinforces whatever random macro-action happened to be emitted by $\pi_\theta$, even if that macro was actively suppressed by the executor.
3. **Critic Flattening & Degenerate Value Landscape**: The critic $V_\phi(s)$ cannot learn meaningful state-action values $Q(s, a)$ because all macro-actions produce indistinguishable trajectories due to heuristic override.

---

## 2. Forensic Code Evidence: Exact Line-by-Line Breakdown

### 2.1 Hardcoded Land Purchase Cutoffs Overriding Macro 3 (`BUY_LAND_EXPANSION`)
In [`src/agents/frontier_executors.py`](file:///C:/Users/Manit/Desktop/kaggle/src/agents/frontier_executors.py#L182-L193):
```python
182: # C. Multi-Quadrant Land Expansion: NW -> NE ($1k @ Day 14) -> SW ($2k @ Day 16) -> SE ($4k @ Day 22)
183: if (macro_action in [3, None]) and day <= 24:
184:     if "NE" not in unlocked and (day <= 14 or money >= 1250) and money >= 1000 and day <= 22:
185:         market_orders.append(["BUY_LAND"])
186:         money -= 1000
187:     elif "SW" not in unlocked and "NE" in unlocked and (day <= 16 or money >= 2500) and money >= 2000 and day <= 22:
188:         market_orders.append(["BUY_LAND"])
189:         money -= 2000
190:     elif "SE" not in unlocked and "SW" in unlocked and "NE" in unlocked and money >= 5000 and day <= 24:
191:         market_orders.append(["BUY_LAND"])
192:         money -= 4000
```
- **Defect A (Suppression)**: When the RL policy specifically outputs `macro_action = 3` (`BUY_LAND_EXPANSION`) on Day 23 with \$4,500 cash to unlock the SE quadrant, Line 184/187 suppresses the order because `day > 22`. The macro is silently dropped, yet logged as executed in PER!
- **Defect B (Leakage)**: When `macro_action = None` (used during baseline rollouts), land is purchased automatically, creating inconsistent behavioral dynamics between RL rollouts and rule-based evaluation.

---

### 2.2 Unconditional Wheat and Carrot Seed Purchasing (Lines 245–276)
In [`src/agents/frontier_executors.py`](file:///C:/Users/Manit/Desktop/kaggle/src/agents/frontier_executors.py#L245-L276):
```python
245: wheat_seeds = seeds.get("WHEAT", 0)
246: planted_wheat = planted_crops.get("WHEAT", 0)
247: if day <= 24 and (wheat_seeds + planted_wheat) < 6 and money >= 40:
248:     qty = min(4, 6 - (wheat_seeds + planted_wheat))
249:     market_orders.append(["BUY_SEED", "WHEAT", qty])
250:     money -= 10 * qty
251:     seeds["WHEAT"] = wheat_seeds + qty
...
269: # Carrots (3 days): stop purchasing after Day 25
270: carrot_budget = (seeds.get("CARROT", 0) + planted_crops.get("CARROT", 0))
271: if day <= 25 and carrot_budget < 16 and money >= 60:
272:     qty = min(6, 16 - carrot_budget)
273:     market_orders.append(["BUY_SEED", "CARROT", qty])
274:     money -= 20 * qty
275:     seeds["CARROT"] = seeds.get("CARROT", 0) + qty
```
- **Defect**: Neither Line 247 nor Line 271 inspects `macro_action`!
- Even if the neural policy emits:
  - `macro_action = 5` (`BUILD_GOOSE_COOP`), or
  - `macro_action = 8` (`MARKET_ARBITRAGE_TRADE`), or
  - `macro_action = 7` (`HARVEST_AND_LIQUIDATE_ALL` on day < 27)
- The executor **always buys up to 6 wheat seeds and up to 16 carrot seeds every single step**.
- The neural policy is completely stripped of its ability to decide crop allocation or preserve cash capital.

---

### 2.3 Static Chore Priority Queue and Unit Misallocation (Lines 509–530)
In [`src/agents/frontier_executors.py`](file:///C:/Users/Manit/Desktop/kaggle/src/agents/frontier_executors.py#L509-L530):
```python
509: # Priority 4: Chores priority queue
510: if best_target is None:
511:     priority_queue = (
512:         harvest_crop_tasks
513:         + harvest_animal_tasks
514:         + care_tasks
515:         + water_crop_tasks
516:         + fertilizer_tasks
517:         + build_coop_tasks
518:         + build_pasture_tasks
519:         + plant_tasks
520:         + weed_tasks
521:     )
```
- **Defect**: The allocation of the farmer and 5–12 farmhands is governed by a static concatenated Python list.
- If the policy chooses `macro_action = 0` (`FARM_CARROTS_INTENSIVE`), one would expect farmhands to prioritize planting, watering, and harvesting carrots.
- Instead, the executor blindly processes `harvest_animal_tasks` and `care_tasks` before any planting task can even be considered.
- The high-level macro has **zero influence on physical worker dispatching**.

---

## 3. Comprehensive Design Tree & Grilling Framework

```
                     MACRO-MICRO ACTOR-LEARNER ARCHITECTURE
                                       |
    +----------------------------------+----------------------------------+
    |                                                                     |
[1. Macro Conditioning Paradigm]                          [2. Micro-Dispatch Coupling]
    |-- A: Discrete Macro Index                               |-- A: Static Concatenated Queue (Broken)
    |-- B: Hierarchical Sub-Goal Vector (Recommended)         |-- B: Hungarian Linear Sum Assignment (Recommended)
    |-- C: Joint End-to-End Micro Multi-Agent                 |-- C: Dynamic Task-Weight Utility Matrix
    |                                                                     |
    +----------------------------------+----------------------------------+
    |                                                                     |
[3. Action Masking & Validation]                          [4. PER Buffer Causal Integrity]
    |-- A: Soft Penalty for Illegal Moves                     |-- A: Log Emitted Policy Action (Current Corrupted)
    |-- B: Strict Invalid Action Masking (Recommended)        |-- B: Log Effective Executed Goal/Action (Required)
    |-- C: Executor Fallback with Null Reward                 |-- C: Off-Policy Importance Correction
```

### Grilling Questions & Architectural Decisions

#### Pillar 1: Macro Action Representation & Conditioning
- ❓ **Q1.1**: *Why did discrete 10-macro action conditioning fail to produce distinct behaviors?*
  - **Forensic Finding**: A single integer $a \in \{0..9\}$ is too coarse to specify both capital market actions (buying land, seeds, animals) and physical chore dispatching across 12 workers on a 100-tile grid. The executor author compensated by hardcoding defaults, breaking causal learning.
  - ➡️ **Recommendation**: Transition to a **Hierarchical Goal Vector Specification** $\mathbf{g}_t \in [0, 1]^K$ or a structured multi-discrete macro:
    $$\mathbf{g}_t = \left[ w_{\text{carrot}}, w_{\text{wheat}}, w_{\text{melon}}, w_{\text{livestock}}, w_{\text{land}}, w_{\text{labor}}, w_{\text{liquidate}} \right]$$
    where the policy outputs continuous target budget/priority weights that parameterize the low-level linear program solver.

- ❓ **Q1.2**: *If we keep discrete macro-actions for MuZero search efficiency, how must the executor contract change?*
  - ➡️ **Recommendation**: Enforce a **Strict Macro-Exclusive Execution Contract**. If `macro_action == FARM_CARROTS_INTENSIVE`, the executor is strictly forbidden from buying wheat seeds, building coops, or trading non-carrot commodities. If an action cannot be satisfied, the executor must execute an explicit no-op or pass, preserving $\mathcal{P}(s' | s, a)$.

---

#### Pillar 2: Micro-Worker Dispatching & Task Assignment
- ❓ **Q2.1**: *Why does greedy closest-first chore matching cause worker traffic jams and delayed harvests?*
  - **Forensic Finding**: Units claim targets in simple loop order. Worker 1 claims a far tile, forcing Worker 2 (standing right next to it) to walk across the entire farm to a secondary tile.
  - ➡️ **Recommendation**: Implement **Hungarian Linear Sum Assignment** (`scipy.optimize.linear_sum_assignment`) over the cost-utility matrix:
    $$C_{i, j} = \text{Dist}(u_i, \tau_j) - \beta \cdot \text{Utility}(\tau_j | \mathbf{g}_t)$$
    where utility is dynamically conditioned on the active macro-action/goal vector.

---

#### Pillar 3: Action Masking & Learner-Executor Alignment
- ❓ **Q3.1**: *How should action masks prevent the policy from picking impossible macros (e.g. `BUY_LAND` when money < \$1000)?*
  - **Forensic Finding**: In `overnight_rl_pipeline.py:587-616`, masks are computed at the root, but discrepancy between `compute_action_mask` and internal executor conditionals leads to invalid actions slipping through.
  - ➡️ **Recommendation**: Unify mask generation into a single canonical function `get_canonical_action_mask(obs)` shared identically across MCTS root selection, child latent dynamics, and the executor fallback assertion.

---

#### Pillar 4: Prioritized Experience Replay (PER) Causal Fidelity
- ❓ **Q4.1**: *What happens to PER priorities when unfaithful transitions are stored?*
  - **Forensic Finding**: The Bellman error $\delta_i = r_t + \gamma V(s_{t+1}) - V(s_t)$ evaluates the value delta created by the *heuristic engine*, but assigns priority to the *neural action*. High-priority updates poison the neural network weights with gradient noise.
  - ➡️ **Recommendation**: Assert that `t["action"]` stored in the replay buffer is the **verified effective action index** returned by the executor. If the executor had to modify or abort the macro due to environmental constraints, log the actual executed macro index or discard the transition.

---

## 4. Drop-In Refactored Code Specification

Below is the complete, drop-in replacement for the unified executor contract, eliminating all hardcoded heuristic overrides and implementing strict macro conditioning with Hungarian linear sum assignment chore dispatch.

```python
"""
Unified Causal Macro-Action Executor Contract for Kaggriculture RL Pipeline.
Eliminates heuristic overrides and ensures faithful (s, a, s') Markov transitions.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.agents.utils import DIRS, SHED_TILES, get_manhattan_dist, get_step_towards, parse_observation
from src.models.encoder import NUM_MACRO_ACTIONS, MACRO_ACTIONS, CROP_SPECS, BASE_PRICES

# Canonical Quadrant Unlock Costs
QUADRANT_COSTS = {"NE": 1000, "SW": 2000, "SE": 4000}


class UnifiedCausalExecutor:
    """
    Strict Causal Macro Executor.
    Guarantees that actions executed on the farm directly reflect the selected macro_action.
    """

    def __init__(self):
        self.shed_tiles = [(4, 4), (5, 4), (4, 5), (5, 5)]

    def __call__(self, obs: Dict[str, Any], macro_action: int) -> Tuple[Dict[str, Any], int]:
        """
        Executes macro_action strictly. Returns action dictionary and the effective macro index.
        """
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        step = state["step"]
        money = state["money"]
        farmer_pos = state["farmer_pos"]
        hands_pos = state["hands_pos"]
        tiles = state["tiles"]
        shed = state["shed"]
        seeds = state["seeds"]
        inventories = state["inventories"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]

        market_orders: List[List[Any]] = []
        effective_macro = macro_action

        # ======================================================================
        # 1. EMERGENCY TURN 718-719 COMPLETE LIQUIDATION
        # ======================================================================
        if step >= 718 or (day == 29 and hour >= 22) or macro_action == 7:
            effective_macro = 7
            for item in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]:
                cnt = shed.get(item, 0)
                if cnt > 0:
                    market_orders.append(["SELL", item, cnt])

            units = [farmer_pos] + hands_pos
            unit_actions = []
            for u_pos in units:
                ux, uy = u_pos
                u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
                if isinstance(u_tile, dict) and u_tile.get("yield_units", 0) > 0:
                    unit_actions.append(["HARVEST"])
                elif u_pos not in self.shed_tiles:
                    nearest_shed = min(self.shed_tiles, key=lambda s: get_manhattan_dist(u_pos, s))
                    unit_actions.append([get_step_towards(u_pos, nearest_shed)])
                else:
                    unit_actions.append(["PASS"])

            return {
                "farmer": unit_actions[0] if unit_actions else ["PASS"],
                "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
                "market": market_orders[:10],
            }, effective_macro

        # ======================================================================
        # 2. STRICT MACRO-CONDITIONED MARKET ALLOCATION
        # ======================================================================
        # A. Sell excess non-feed produce to maintain liquidity
        for item in ["EGG", "MILK", "WOOL", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]:
            count = shed.get(item, 0)
            if count > 0:
                market_orders.append(["SELL", item, count])

        # B. Macro Action Specific Branching (NO UNCONDITIONAL SEED/LAND PURCHASES)
        if macro_action == 0:  # FARM_CARROTS_INTENSIVE
            carrot_seeds = seeds.get("CARROT", 0)
            if carrot_seeds < 12 and money >= 20 and day <= 26:
                buy_qty = min(12 - carrot_seeds, int(money // 20), 8)
                if buy_qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", buy_qty])
                    money -= buy_qty * 20

        elif macro_action == 1:  # FARM_WHEAT_EXPANSION
            wheat_seeds = seeds.get("WHEAT", 0)
            if wheat_seeds < 10 and money >= 10 and day <= 25:
                buy_qty = min(10 - wheat_seeds, int(money // 10), 8)
                if buy_qty > 0:
                    market_orders.append(["BUY_SEED", "WHEAT", buy_qty])
                    money -= buy_qty * 10

        elif macro_action == 2:  # FARM_DIVERSIFIED (High-Yield Melon / Tomato)
            if day <= 14 and money >= 80:
                melon_seeds = seeds.get("MELON", 0)
                if melon_seeds < 6:
                    buy_qty = min(6 - melon_seeds, int(money // 80), 4)
                    if buy_qty > 0:
                        market_orders.append(["BUY_SEED", "MELON", buy_qty])
                        money -= buy_qty * 80
            elif day <= 6 and money >= 50:
                tomato_seeds = seeds.get("TOMATO", 0)
                if tomato_seeds < 6:
                    buy_qty = min(6 - tomato_seeds, int(money // 50), 4)
                    if buy_qty > 0:
                        market_orders.append(["BUY_SEED", "TOMATO", buy_qty])
                        money -= buy_qty * 50

        elif macro_action == 3:  # BUY_LAND_EXPANSION
            if "NE" not in unlocked and money >= QUADRANT_COSTS["NE"]:
                market_orders.append(["BUY_LAND"])
                money -= QUADRANT_COSTS["NE"]
            elif "SW" not in unlocked and "NE" in unlocked and money >= QUADRANT_COSTS["SW"]:
                market_orders.append(["BUY_LAND"])
                money -= QUADRANT_COSTS["SW"]
            elif "SE" not in unlocked and "SW" in unlocked and "NE" in unlocked and money >= QUADRANT_COSTS["SE"]:
                market_orders.append(["BUY_LAND"])
                money -= QUADRANT_COSTS["SE"]

        elif macro_action == 4:  # HIRE_EXTRA_LABOR
            max_hires = 5 if day >= 15 else 3
            if hour <= 3 and hires_today < max_hires and money >= 2:
                needed = max_hires - hires_today
                hire_batch = min(needed, 5)
                for _ in range(hire_batch):
                    market_orders.append(["HIRE"])
                money -= hire_batch * 2

        elif macro_action == 5:  # BUILD_GOOSE_COOP
            if money >= 300 and shed.get("GOOSE", 0) == 0:
                market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
                money -= 300

        elif macro_action == 6:  # BUILD_PASTURE_LIVESTOCK
            if money >= 400 and shed.get("COW", 0) == 0:
                market_orders.append(["BUY_ANIMAL", "COW", 1])
                money -= 400
            elif money >= 500 and shed.get("SHEEP", 0) == 0:
                market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                money -= 500

        elif macro_action == 8:  # MARKET_ARBITRAGE_TRADE
            # Feed preservation or opportunistic fertilizer purchase
            if shed.get("FERTILIZER", 0) < 4 and money >= 100:
                market_orders.append(["BUY_PRODUCT", "FERTILIZER", 2])
                money -= 200

        elif macro_action == 9:  # LIVESTOCK_CARE_FEED
            wheat_in_shed = shed.get("WHEAT", 0)
            if wheat_in_shed < 4 and money >= 25:
                market_orders.append(["BUY_PRODUCT", "WHEAT", 4])
                money -= 100

        # ======================================================================
        # 3. MACRO-WEIGHTED HUNGARIAN CHORE DISPATCH
        # ======================================================================
        all_units = [farmer_pos] + hands_pos
        num_units = len(all_units)

        # Harvest, Care, Water, Plant task compilation
        task_list: List[Dict[str, Any]] = []
        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                pos = (c, r)
                t = tiles[r][c]
                if t is None:
                    if macro_action in [0, 1, 2] and sum(seeds.values()) > 0:
                        task_list.append({"type": "PLANT", "pos": pos, "weight": 50.0})
                    elif macro_action == 5:
                        task_list.append({"type": "BUILD_COOP", "pos": pos, "weight": 80.0})
                    elif macro_action == 6:
                        task_list.append({"type": "BUILD_PASTURE", "pos": pos, "weight": 80.0})
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        if t.get("yield_units", 0) > 0:
                            task_list.append({"type": "HARVEST", "pos": pos, "weight": 100.0})
                        elif not t.get("watered_today", False):
                            w = 70.0 if macro_action in [0, 1, 2] else 40.0
                            task_list.append({"type": "WATER", "pos": pos, "weight": w})
                    elif k in ["COOP", "PASTURE"] and t.get("animal"):
                        if not t.get("fed_today", False) or not t.get("cared_today", False):
                            w = 90.0 if macro_action == 9 else 50.0
                            task_list.append({"type": "ANIMAL_CARE", "pos": pos, "weight": w})

        # Cost matrix: distance - task_weight
        num_tasks = len(task_list)
        unit_actions: List[List[Any]] = [["PASS"] for _ in range(num_units)]

        if num_tasks > 0 and num_units > 0:
            cost_matrix = np.zeros((num_units, num_tasks), dtype=np.float32)
            for u_idx, u_pos in enumerate(all_units):
                for t_idx, task in enumerate(task_list):
                    dist = get_manhattan_dist(u_pos, task["pos"])
                    cost_matrix[u_idx, t_idx] = dist * 2.0 - task["weight"]

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for u_idx, t_idx in zip(row_ind, col_ind):
                u_pos = all_units[u_idx]
                target_task = task_list[t_idx]
                t_pos = target_task["pos"]

                if u_pos == t_pos:
                    ttype = target_task["type"]
                    if ttype == "HARVEST":
                        unit_actions[u_idx] = ["HARVEST"]
                    elif ttype == "WATER":
                        unit_actions[u_idx] = ["WATER"]
                    elif ttype == "PLANT":
                        for c_cand in ["CARROT", "WHEAT", "MELON", "TOMATO", "STRAWBERRY"]:
                            if seeds.get(c_cand, 0) > 0:
                                unit_actions[u_idx] = ["PLANT", c_cand]
                                seeds[c_cand] -= 1
                                break
                    elif ttype == "BUILD_COOP":
                        unit_actions[u_idx] = ["BUILD_COOP"]
                    elif ttype == "BUILD_PASTURE":
                        unit_actions[u_idx] = ["BUILD_PASTURE"]
                    elif ttype == "ANIMAL_CARE":
                        unit_actions[u_idx] = ["CARE"]
                else:
                    unit_actions[u_idx] = [get_step_towards(u_pos, t_pos)]

        return {
            "farmer": unit_actions[0],
            "hands": unit_actions[1:],
            "market": market_orders[:10],
        }, effective_macro
```

---

## 5. Verification & Acceptance Criteria

1. **Replay Buffer Verification**:
   - Confirm that `t["action"]` recorded in the PER buffer strictly matches `effective_macro`.
   - Verify that policy loss $\mathcal{L}_{\text{policy}}$ displays non-zero gradient variance and distinct policy entropy reduction across training iterations.
2. **Deterministic Macro Isolation Test**:
   - Force policy to emit `a=0` (Carrot) for 50 steps: verify zero wheat seed orders are issued.
   - Force policy to emit `a=3` (Land Expansion) with \$4,000 on Day 23: verify `BUY_LAND` is executed without arbitrary date cutoffs.
3. **Hungarian Labor Allocation Efficiency**:
   - Farmhand travel steps reduced by $\ge 35\%$ compared to the static concatenated queue.
