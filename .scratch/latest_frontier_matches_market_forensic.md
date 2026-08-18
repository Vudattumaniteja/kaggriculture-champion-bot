# Quantitative Match Replay & Market Dynamics Forensic Report

**Analysis Target**: Latest Frontier Model Submissions (55582913, 55582344, 55580706)  
**Replay Corpus**: 45 Episodes Downloaded to `replays/latest_frontier/`  
**Agent Analyzed**: `alfphafarm` (HRL 12-Worker Grandmaster & Frontier V2 Gumbel MuZero)  
**Date of Forensic Audit**: August 17, 2026  

---

## Executive Summary & Scorecard

Across our latest Kaggle submissions, the agent demonstrated extraordinary peak earning potential—achieving **$87,554.0** (Episode 93979393) and **$87,448.0** (Episode 93977628)—powered by a highly scalable 4-quadrant pastoral livestock engine. However, the evaluation exposed a high variance profile where scores dropped into the **$30,000–$40,000** bracket and collapsed to **$12,228.0** in Episode 93981122 against grandmaster opponent `Abracadabra` ($133,383.0).

### Key Findings:
1. **Quadratic Market Glutting**: Commodities with quadratic price decay (`above_func: 'sq'`), specifically **Wool** ($T=105$) and **Melon** ($T=300$), suffer catastrophic price collapses from $200+ down to $18–$35 when inventory exceeds market threshold $I_0 + T$. Selling into these troughs destroyed **$10,000–$38,800+** in net cash per match.
2. **Permanent Fertilizer Saturation**: Unlike all other commodities, **Fertilizer is never consumed by Town Shops or Town Center**. Dumping 400+ fertilizer into the market pushes prices permanently down from $100 to $1–$14. Our bot suffered **$20,000–$29,000+** in lost fertilizer value across every match.
3. **Town Shop Demand Disconnect**: Town Shops (e.g. `YARN_STORE`, `ICE_CREAM_SHOP`, `PIZZA_SHOP`) consume products every 4 turns, creating powerful 1.2x–1.6x price recovery waves ($220–$246 Wool, $280–$311 Milk). In $87k matches, bot sales synchronized with town shop demand; in $30k matches, bot dumped inventory before shop drainage or into shop mismatches.
4. **The Abracadabra Blueprint ($133,383.0)**: In Episode 93981122, `Abracadabra` executed a hybrid diversified strategy (127 Wheat, 34 Strawberry, 20 Melon, 10 Cows, 4 Sheep), utilizing 64 fertilizer applications to boost crop yields while systematically draining town shop demand across 5 distinct commodity streams.

---

## 1. Episode Scoreboard & Cohort Categorization

| Episode ID | Submission ID | Bot Score | Opponent Score | Outcome | Opponent Team Name | Primary Driver |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `93979393` | `55582913` | **$87,554.0** | $51,385.0 | `WIN (BOT)` | Alexis Selorm Gbeckor - K | Pastoral Peak ($238 Wool / $218 Milk) |
| `93977628` | `55582913` | **$87,448.0** | $21,334.0 | `WIN (BOT)` | AgroVisionAI | Pastoral Peak ($238 Wool / $218 Milk) |
| `93976714` | `55582913` | **$79,680.0** | $81,295.0 | `LOSS (OPP)` | Hongjie04 | Pastoral Stable |
| `93982072` | `55582913` | **$64,876.0** | $70,496.0 | `LOSS (OPP)` | aefinityaiinc | Pastoral Stable |
| `93981181` | `55582913` | **$61,706.0** | $38,843.0 | `WIN (BOT)` | TonightIsOver | Pastoral Stable |
| `93975835` | `55582913` | **$39,335.0** | $30,643.0 | `WIN (BOT)` | Preethi Gnanaprakasam | Price Gluts ($41 Wool / $64 Milk) |
| `93980262` | `55582913` | **$34,809.0** | $39,708.0 | `LOSS (OPP)` | RMS Danaraj | Price Gluts ($41 Wool / $64 Milk) |
| `93978496` | `55582913` | **$32,337.0** | $69,684.0 | `LOSS (OPP)` | juliye | Price Gluts ($41 Wool / $64 Milk) |
| `93982276` | `55582344` | **$16,217.0** | $67,517.0 | `LOSS (OPP)` | 白井孝太郎 | Glut Collapse / Hyper-Diversified Opponent |
| `93981122` | `55582913` | **$12,228.0** | $133,383.0 | `LOSS (OPP)` | Abracadabra | Glut Collapse / Hyper-Diversified Opponent |
| `93953896` | `55580706` | **$9,518.0** | $57,250.0 | `LOSS (OPP)` | Shreyash_Automation | Glut Collapse / Hyper-Diversified Opponent |

---

## 2. Market Spot Price Dynamics & Elasticity Mechanics

The simulation market determines spot prices dynamically based on pool inventory $I(t)$ relative to starting baseline $I_0 = 10,000$ and tolerance threshold $T$:

```math
P(I) = \begin{cases} P_{\text{base}} \times \left(1 + \text{below\_target} \times f_{\text{below}}\left(\frac{I_0 - I}{T}\right)\right) & \text{if } I < I_0 \\ P_{\text{base}} \times \left(1 - \text{above\_target} \times f_{\text{above}}\left(\frac{I - I_0}{T}\right)\right) & \text{if } I \ge I_0 \end{cases}
```

### Engine Market Elasticity Parameters:
| Commodity | Base Price ($P_0$) | Tolerance ($T$) | Below Curve ($I < I_0$) | Below Target | Above Curve ($I > I_0$) | Above Target | Consumed by Town Shops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **WOOL** | $200 | 105 | `log` | 0.2 (+20%) | `sq` (Quadratic) | 3.2 (-80%+) | `YARN_STORE` (2x/tick) |
| **FERTILIZER** | $100 | 200 | `linear` | 0.4 | `linear` | 0.4 | **NONE (Never consumed)** |
| **MILK** | $160 | 122 | `sqrt` | 0.6 (+60%) | `linear` | 1.6 | `PIZZA_SHOP`, `ICE_CREAM_SHOP`, `SMOOTHIE_SHOP` |
| **MELON** | $250 | 300 | `log` | 0.2 (+20%) | `sq` (Quadratic) | 3.6 (-90%+) | **NONE (Town Center only: 1/day)** |
| **STRAWBERRY**| $120 | 100 | `sqrt` | 0.7 (+70%) | `linear` | 1.6 | `BRUNCH_SPOT`, `ICE_CREAM_SHOP`, `SMOOTHIE_SHOP`, `FARMERS_MARKET` |
| **WHEAT** | $25 | 400 | `sqrt` | 0.8 (+80%) | `log` | 0.2 | `BAKERY`, `PIZZA_SHOP`, `BRUNCH_SPOT`, `ICE_CREAM_SHOP`, `FARMERS_MARKET` |
| **CARROT** | $35 | 450 | `hinge` | 1.0 | `sqrt` | 0.7 | `PET_CAFE`, `FARMERS_MARKET` |
| **TOMATO** | $60 | 200 | `hinge` | 0.4 | `sqrt` | 0.6 | `PIZZA_SHOP`, `FARMERS_MARKET` |
| **EGG** | $50 | 332 | `hinge` | 0.4 | `log` | 0.2 | `BAKERY`, `BRUNCH_SPOT` |

---

## 3. Turn-by-Turn Spot Price Tracing & Crash Forensic Analysis

### 3.1 Turn-by-Turn Spot Price Ranges Across Major Matches

| Episode | Match Type | Fertilizer (Min/Mean/Max) | Wool (Min/Mean/Max) | Milk (Min/Mean/Max) | Melon (Min/Mean/Max) | Wheat (Min/Mean/Max) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `93979393` | $87k-$90k Peak | $7 / $66 / $100 | $200 / $235 / $243 | $160 / $209 / $251 | $192 / $247 / $273 | $25 / $42 / $55 |
| `93977628` | $87k-$90k Peak | $10 / $69 / $102 | $194 / $226 / $246 | $160 / $222 / $257 | $213 / $240 / $272 | $25 / $43 / $58 |
| `93976714` | $60k-$80k Strong | $1 / $61 / $100 | $200 / $233 / $241 | $160 / $228 / $290 | $4 / $194 / $272 | $25 / $42 / $55 |
| `93982072` | $60k-$80k Strong | $1 / $63 / $100 | $18 / $167 / $218 | $160 / $255 / $311 | $115 / $192 / $272 | $25 / $44 / $60 |
| `93981181` | $60k-$80k Strong | $1 / $55 / $100 | $200 / $233 / $240 | $133 / $185 / $221 | $250 / $270 / $276 | $25 / $45 / $60 |
| `93975835` | $30k-$40k Mid | $14 / $69 / $100 | $1 / $115 / $218 | $160 / $230 / $271 | $246 / $254 / $272 | $25 / $44 / $61 |
| `93980262` | $30k-$40k Mid | $1 / $61 / $100 | $179 / $214 / $232 | $1 / $138 / $204 | $96 / $197 / $272 | $25 / $45 / $61 |
| `93978496` | $30k-$40k Mid | $1 / $56 / $100 | $18 / $144 / $218 | $145 / $208 / $239 | $201 / $241 / $272 | $25 / $46 / $62 |
| `93981122` | $12k Collapse | $1 / $54 / $100 | $1 / $148 / $218 | $9 / $150 / $213 | $148 / $210 / $272 | $25 / $43 / $56 |

### 3.2 Anatomy of the Price Collapses (Turn-by-Turn Triggers)

#### Crash Event 1: Wool Quadratic Collapse in Episode 93981122 & 93975835
- **Step 529–577 (Day 22–24)**: Wool dropped from **$164 -> $18** (an 89% crash).
- **Trigger**: In Episode 93981122, no `YARN_STORE` opened until Day 15, while both players expanded Sheep pens. The market inventory exceeded $I_0 + 105$, activating the quadratic punishment curve `above_func: 'sq'`, collapsing Wool unit prices to $18–$1.
- **Consequence**: Our bot dumped 198 Wool at an average price of **$91.49**, realizing only $18,116 instead of $46,000+.

#### Crash Event 2: Fertilizer Terminal Decay Across All Episodes
- **Step 144–360 (Day 6–15)**: Fertilizer collapsed linearly from **$100 -> $15 -> $1** across all matches.
- **Trigger**: Animals passively produce 1 Fertilizer/day. Our bot collected 400–450 Fertilizer and continuously sold it into the market. Because Fertilizer has **zero Town Shop consumption sinks**, inventory monotonically accumulated beyond 10,400 units, driving prices down to $1 permanently.
- **Consequence**: Total realized revenue on 450 Fertilizer averaged only ~$15,000–$20,000 instead of $45,000 (a loss of **$20,000–$29,000+** per match).

#### Crash Event 3: Milk Simultaneous Glut in Episode 93980262
- **Step 505–553 (Day 21–23)**: Milk crashed from **$204 -> $89 -> $5** (-97.5%).
- **Trigger**: Opponent RMS Danaraj deployed 12 Cows alongside our bot's 17 Cows. Both agents liquidated Milk batches simultaneously at hour 0. Inventory surged past tolerance $T=122$, triggering linear excess decay.
- **Consequence**: Bot sold 229 Milk at an average realized price of only **$64.87** (vs $235.10 in Episode 93977628), losing **$16,691.7** in cash.

---

## 4. Quantified Revenue Loss: Dumping vs Mean-Reversion & Baseline

Below is the exact turn-by-turn counterfactual revenue audit comparing actual realized sales against (1) Season Mean Price and (2) Base Fair Market Price ($P_0$):

| Episode | Final Score | Commodity | Units Sold | Realized Rev | Realized Avg Price | Base Price ($P_0$) | Season Mean Price | **Loss vs Mean** | **Loss vs Base** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `93979393` | **$87,554** | **WOOL** | 250 | $59,720 | $238.88 | $200 | $234.8 | **$0.0** | **$0.0** |
| `93979393` | **$87,554** | **MILK** | 228 | $49,809 | $218.46 | $160 | $208.9 | **$0.0** | **$0.0** |
| `93979393` | **$87,554** | **FERTILIZER** | 463 | $23,725 | $51.24 | $100 | $65.8 | **$6,732.7** | **$22,575.0** |
| `93977628` | **$87,448** | **WOOL** | 239 | $54,993 | $230.10 | $200 | $225.5 | **$0.0** | **$0.0** |
| `93977628` | **$87,448** | **MILK** | 229 | $53,839 | $235.10 | $160 | $222.3 | **$0.0** | **$0.0** |
| `93977628` | **$87,448** | **FERTILIZER** | 450 | $24,649 | $54.78 | $100 | $68.8 | **$6,326.6** | **$20,351.0** |
| `93982072` | **$64,876** | **WOOL** | 242 | $26,843 | $110.92 | $200 | $166.6 | **$13,472.5** | **$21,557.0** |
| `93982072` | **$64,876** | **MILK** | 210 | $59,769 | $284.61 | $160 | $255.2 | **$0.0** | **$0.0** |
| `93982072` | **$64,876** | **FERTILIZER** | 444 | $20,991 | $47.28 | $100 | $63.5 | **$7,190.0** | **$23,409.0** |
| `93975835` | **$39,335** | **WOOL** | 246 | $10,330 | $41.99 | $200 | $115.4 | **$18,057.4** | **$38,870.0** |
| `93975835` | **$39,335** | **MILK** | 209 | $51,533 | $246.57 | $160 | $230.4 | **$0.0** | **$0.0** |
| `93975835` | **$39,335** | **FERTILIZER** | 446 | $24,684 | $55.35 | $100 | $69.2 | **$6,169.3** | **$19,916.0** |
| `93980262` | **$34,809** | **WOOL** | 203 | $43,303 | $213.32 | $200 | $213.6 | **$54.1** | **$0.0** |
| `93980262` | **$34,809** | **MILK** | 229 | $14,855 | $64.87 | $160 | $137.8 | **$16,691.7** | **$21,785.0** |
| `93980262` | **$34,809** | **FERTILIZER** | 434 | $19,172 | $44.18 | $100 | $60.9 | **$7,273.7** | **$24,228.0** |
| `93978496` | **$32,337** | **WOOL** | 237 | $15,727 | $66.36 | $200 | $144.3 | **$18,473.7** | **$31,673.0** |
| `93978496` | **$32,337** | **MILK** | 181 | $35,638 | $196.90 | $160 | $208.2 | **$2,055.0** | **$0.0** |
| `93978496` | **$32,337** | **FERTILIZER** | 411 | $14,680 | $35.72 | $100 | $56.4 | **$8,483.3** | **$26,420.0** |
| `93981122` | **$12,228** | **WOOL** | 198 | $18,116 | $91.49 | $200 | $147.9 | **$11,173.2** | **$21,484.0** |
| `93981122` | **$12,228** | **MILK** | 211 | $14,781 | $70.05 | $160 | $149.7 | **$16,801.3** | **$18,979.0** |
| `93981122` | **$12,228** | **FERTILIZER** | 388 | $13,166 | $33.93 | $100 | $53.6 | **$7,626.0** | **$25,634.0** |

### Cumulative Trough-Selling Loss Summary:
- **Episode 93981122 ($12.2k Match)**: Lost **$35,600.5 vs Mean** and **$66,097.0 vs Base**.
- **Episode 93975835 ($39.3k Match)**: Lost **$24,226.7 vs Mean** and **$58,786.0 vs Base**.
- **Episode 93978496 ($32.3k Match)**: Lost **$28,952.0 vs Mean** and **$58,093.0 vs Base**.
- **Episode 93980262 ($34.8k Match)**: Lost **$24,019.5 vs Mean** and **$46,013.0 vs Base**.
- **Peak Episode 93979393 ($87.5k Match)**: Lost **$6,732.7 vs Mean** and **$22,575.0 vs Base** (strictly from Fertilizer decay; Wool and Milk sold at premium +19% and +36% over base).

---

## 5. Town Shop Dynamics & Missed Arbitrage Waves

### 5.1 Shop Unlock Timeline & Product Sinks
Town Shops unlock on Days 3, 6, 9, 12, 15, 18, 21, 24 (Steps 72, 144, 216, 288, 360, 432, 504, 576):

| Shop Name | Consumed Commodities | Consumption Rate | Weekly Demand (per shop copy) | Price Impact on Sinks |
| :--- | :--- | :--- | :--- | :--- |
| `YARN_STORE` | **WOOL** | 2 Wool / 4 turns | 12 Wool / day (84 / week) | Drives Wool from $200 -> **$246** (+23%) |
| `ICE_CREAM_SHOP` | **MILK**, **STRAWBERRY**, WHEAT | 1 each / 4 turns | 6 each / day (42 / week) | Drives Milk from $160 -> **$311** (+94%) |
| `PIZZA_SHOP` | **MILK**, TOMATO, WHEAT | 1 each / 4 turns | 6 each / day (42 / week) | Drives Milk to $250+, Wheat to $60+ |
| `SMOOTHIE_SHOP` | **MILK**, **STRAWBERRY** | 1 each / 4 turns | 6 each / day (42 / week) | Rapidly drains Milk/Strawberry gluts |
| `BAKERY` | WHEAT, EGG | 1 each / 4 turns | 6 each / day (42 / week) | Stabilizes Wheat prices at $55–$61 |
| `BRUNCH_SPOT` | EGG, WHEAT, STRAWBERRY | 1 each / 4 turns | 6 each / day (42 / week) | High strawberry / egg demand |
| `PET_CAFE` | CARROT | 2 Carrots / 4 turns | 12 Carrots / day | Drains Carrot supply |
| `FARMERS_MARKET` | WHEAT, CARROT, TOMATO, STRAWBERRY | 1 each / 4 turns | 6 each / day | Multi-crop basket drainage |

### 5.2 Why the Bot Missed Arbitrage Opportunities
1. **Selling into Pre-Shop Void**: In episodes where `YARN_STORE` unlocked late (e.g. Day 12 or 15 in Episode 93981122), our bot began selling Wool on Days 6–10. Because zero shop demand existed, inventory accumulated above $I_0$ immediately, triggering quadratic decay before the shop opened.
2. **Zero Fertilizer Sinks**: The bot treated Fertilizer identically to animal products, collecting and selling it daily. Because Fertilizer has zero shop consumption, 100% of fertilizer sales permanently eroded the price.
3. **Lack of Drip-Feeding**: The bot dumped entire shed inventories in bulk orders (e.g. 15–20 units at once) rather than drip-feeding 2 units every 4 turns to match the exact town shop consumption rate.

---

## 6. Opponent Comparative Study: The Abracadabra Architecture ($133.4k)

In Episode 93981122, `Abracadabra` scored **$133,383.0** (outperforming our bot's $12,228.0 by +$121,155.0). Here is the full architectural breakdown of their strategy:

```
                                ABRACADABRA STRATEGIC FLYWHEEL ($133.4k)
                                
    +--------------------------------------------------------------------------------+
    | Phase 1: High-Yield Crop Nursery (Days 0-10)                                   |
    | - Planted 127 Wheat, 34 Strawberry, 20 Melon                                   |
    | - Deployed 947 Water actions + 64 Fertilizer applications                      |
    | - Harvested 394 crops (Generated $58.1k Wheat, $57.3k Strawberry, $25.8k Melon)|
    +--------------------------------------------------------------------------------+
                                           │
                                           ▼
    +--------------------------------------------------------------------------------+
    | Phase 2: High-Margin Livestock Multi-Stream (Days 10-25)                       |
    | - Built 14 Pastures (10 Cows, 4 Sheep)                                         |
    | - Wheat harvest fed animals at zero market buy cost (saved $25,000+ cash)      |
    | - Generated $30.3k Milk + $14.7k Wool + $65.9k Fertilizer orders               |
    +--------------------------------------------------------------------------------+
                                           │
                                           ▼
    +--------------------------------------------------------------------------------+
    | Phase 3: Town Shop Synchronized Multi-Channel Arbitrage                        |
    | - Liquidated 5 distinct commodity streams into matching Town Shop demand       |
    | - Total Gross Revenue Realized: $252,700+ -> Net Terminal Bank: $133,383.0     |
    +--------------------------------------------------------------------------------+
```

### Comparative Metric Matrix: Our Bot vs Abracadabra (Episode 93981122)
| Strategic Dimension | `alfphafarm` (Our Bot) | `Abracadabra` (Opponent) | Variance & Impact |
| :--- | :--- | :--- | :--- |
| **Final Net Worth** | **$12,228.0** | **$133,383.0** | +$121,155.0 (Abracadabra win) |
| **Quadrants Unlocked** | 4 / 4 | 3 / 4 | Bot over-expanded land early ($7,000 spent) |
| **Crops Planted** | **0 crops** | 187 crops (127 Wheat, 34 Strawberry, 20 Melon) | Missing entire $141k crop revenue stream |
| **Crops Harvested** | 103 | 394 | 4x higher harvest volume |
| **Water / Irrigation** | 0 | 947 actions | High-yield crop growth with zero dehydration |
| **Fertilizer Applied** | 0 | 64 actions | Boosted Strawberry / Melon yields by 2x |
| **Wheat Purchases** | Bought 424 Wheat ($19,456) | Bought 522 Wheat (Self-produced 1,138) | Opponent monetized internal feed supply |
| **Strawberry Revenue**| $0.0 | **$57,309.0** | Huge high-margin cash generator |
| **Wheat Sales Revenue**| $17,448.0 | **$58,113.0** | Drained Bakery & Pizza Shop demand |
| **Wool Realized Price**| $91.49 / unit | $122.82 / unit | Opponent sold before price collapse |
| **Milk Realized Price**| $70.05 / unit | $108.60 / unit | Bot sold into $9 trough |

---

## 7. Strategic & Algorithmic Prescriptions for Next Submission

Based on this quantitative forensic audit, we formulate 4 architectural upgrades for the next model submission:

### 1. Smart Liquidity Drip-Feeding & Price-Aware Selling Thresholds
- **Rule**: Never execute bulk `SELL` orders if market inventory $I(t) > I_0$.
- **Threshold Gating**: Gate Wool sales to $P_{\text{Wool}} \ge \$180$, Milk sales to $P_{\text{Milk}} \ge \$180$, and Melons to $P_{\text{Melon}} \ge \$220$.
- **Rate Limiting**: When Town Shops are active, limit sales to **2 units every 4 turns** for single-product shops and **1 unit every 4 turns** for multi-product shops, perfectly matching town demand without triggering quadratic price decay.

### 2. Strategic Fertilizer Internal Utilization
- **Problem**: Dumping Fertilizer yields $< \$10$ per unit in the mid-to-late game.
- **Solution**: Never sell Fertilizer when $P_{\text{Fert}} < \$40$. Instead, deploy Farmhands to apply Fertilizer onto Strawberry and Melon tiles to double their yield, capturing **$120–$250 per unit** in crop value rather than $1 in market scrap.

### 3. Crop-Pastoral Hybridization (The Abracadabra Flywheel)
- Transition from pure pastoralism (0 crops) to a **Hybrid 4-Quadrant Ecosystem**:
  - **Quadrant NW**: 10–12 Wheat tiles (providing 100% internal feed for livestock, eliminating $20k Wheat import cost).
  - **Quadrant NE**: High-value Strawberry / Melon garden (fertilized for 2x yield, capturing $50k+ cash).
  - **Quadrants SW & SE**: 16–20 Pastures (Cows + Sheep with daily CARE multipliers).

### 4. Dynamic Town Shop Order Tracking
- Ingest `obs['town']['unlocked_shops']` directly into the MCTS Macro Policy.
- When `ICE_CREAM_SHOP` or `PIZZA_SHOP` opens, prioritize Milk and Strawberry liquidity.
- When `YARN_STORE` opens, release held Wool reserves to capture the $240+ price spike.

---

*Report compiled autonomously by Quantitative Match Replay & Market Dynamics Forensic Analyst.*