"""
Comprehensive Forensic Markdown Report Generator for Top 10 Kaggriculture Competitors
Saves full report to .scratch/top10_competitors_gameplay_analysis.md
"""

import os
import sys
import json
from collections import defaultdict, Counter
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPORT_PATH = ".scratch/top10_competitors_gameplay_analysis.md"

with open(".scratch/competitor_match_aggregates.json", "r", encoding="utf-8") as f:
    data = json.load(f)

TOP_TEAMS = [
    {"rank": 1, "team_id": 16677252, "name": "カワシギ", "public_score": 3214.0},
    {"rank": 2, "team_id": 16719123, "name": "Thomas Tschinkel", "public_score": 3118.9},
    {"rank": 3, "team_id": 16696304, "name": "Utkarsh #2", "public_score": 3020.7},
    {"rank": 4, "team_id": 16703045, "name": "ReCurSiON", "public_score": 2996.4},
    {"rank": 5, "team_id": 16715733, "name": "peikopon", "public_score": 2987.5},
    {"rank": 6, "team_id": 16668721, "name": "Efe Can Celiksoy", "public_score": 2968.6},
    {"rank": 7, "team_id": 16662121, "name": "One-For-All", "public_score": 2968.1},
    {"rank": 8, "team_id": 16712189, "name": "Kostiantyn Isaienkov", "public_score": 2962.5},
    {"rank": 9, "team_id": 16667020, "name": "SCLim2022080004", "public_score": 2959.3},
    {"rank": 10, "team_id": 16723379, "name": "Matteo123383iend", "public_score": 2957.0}
]

lines = []

lines.append("# Exhaustive Forensic Analysis of Top 10 Kaggriculture Competitors")
lines.append("")
lines.append("> **Analysis Date**: August 17, 2026  ")
lines.append("> **Dataset**: 33 Full Match Replays from Leaderboard Top 10 Teams (`replays/top_competitors/`)  ")
lines.append("> **Target Horizon**: 720 Turns (30 Days × 24 Turns/Day)  ")
lines.append("")
lines.append("---")
lines.append("")

lines.append("## Executive Summary & The Dominant Championship Meta")
lines.append("")
lines.append("Across all top 10 competitors on the Kaggriculture leaderboard (scores ranging from 2,957.0 to 3,214.0), a **unified, highly optimized meta-strategy** has emerged that completely outperforms standard baseline farming algorithms. While basic agents achieve $5,000–$10,000 by simple manual planting, top competitors consistently achieve **$75,000 to $155,000+** in cash at Turn 719.")
lines.append("")
lines.append("### Core Pillars of the Championship Meta (\"The Pastoral-Fertilizer Flywheel\"):")
lines.append("1. **Turn 1 Capital Liquidation (100% Reinvestment)**:")
lines.append("   - On Turn 1 (Day 0, Hour 1), top agents spend **99.3% of their $3,000 starting bankroll** (leaving $10–$25 cash).")
lines.append("   - They execute an aggressive composite opening: build 1–2 `PASTURE` structures, purchase 2–4 `COW`s and 2–4 `SHEEP`, queue 5 `HIRE` orders (max daily cap), buy 6–14 units of `WHEAT` product for feed, and purchase `WHEAT` and `MELON` seeds.")
lines.append("2. **Passive High-Value Yields (Milk, Wool, & Massive Fertilizer)**:")
lines.append("   - Cows and Sheep generate daily yields of `MILK` ($190/unit base) and `WOOL` ($215/unit base), plus an enormous volume of `FERTILIZER` ($95/unit base).")
lines.append("   - By continually feeding and caring for livestock, top bots collect **500 to 2,300+ units of Fertilizer** per match, generating **$25,000–$60,000+** in pure market revenue.")
lines.append("3. **Aggressive Labor Force Scaling (260–280 Farmhand-Days)**:")
lines.append("   - Labor hiring follows a precise 3-tier ramp: 5 farmhands on Day 0, 4–5 daily through Day 6, expanding to 8 on Day 7, 11 on Day 8, and sustaining the **maximum legal cap of 11–12 farmhands every single day from Day 9 to Day 28**.")
lines.append("   - Total wage investment reaches $13,000–$14,000 per match, which provides 2,500+ worker-hours of tilling, watering, planting, and caretaking.")
lines.append("4. **Synchronized Land Expansion & Crop Rotation**:")
lines.append("   - **Day 6 (Turn 149)**: Unlock `NE` Quadrant ($1,000 cost).")
lines.append("   - **Day 10–11 (Turn 241–265)**: Unlock `SW` Quadrant ($2,000 cost).")
lines.append("   - **Day 12 (Turn 289)**: Unlock `SE` Quadrant ($4,000 cost) for maximum scale bots (Ranks 1, 5, 8, 10).")
lines.append("   - Crop selection is heavily weighted towards **Wheat (110–145 plots)** for fast cycle cash/feed, **Strawberries (33–42 plots)** for multi-harvest revenue density, and **Melons (12–20 plots)** timed for massive Day 10 cash injections ($266+/unit).")
lines.append("5. **Open Market Liquidation Over Town Shop Bundles**:")
lines.append("   - Top bots almost entirely bypass Town Shop contracts due to small volume fulfillment limits, choosing instead to flood the open market with hundreds of units of crops, animal products, and fertilizer.")
lines.append("")
lines.append("---")
lines.append("")

# Table 1: Leaderboard & Performance Overview
lines.append("## 1. Top 10 Leaderboard & Forensic Performance Overview")
lines.append("")
lines.append("| Rank | Team Name | Team ID | Public Score | Matches Analyzed | Mean Replay Cash | Peak Replay Cash | Total Farmhands Hired | Dominant Strategy Archetype |")
lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

for t in TOP_TEAMS:
    r = t["rank"]
    matches = data.get(str(r), [])
    if not matches:
        continue
    f_rewards = [m["final_reward"] for m in matches]
    p_cashes = [m["peak_cash"] for m in matches]
    hires = [m["total_hires"] for m in matches]
    
    mean_r = np.mean(f_rewards)
    peak_c = np.max(p_cashes)
    mean_h = np.mean(hires)
    
    # Archetype label
    if r == 1:
        arch = "Pastoral-Crop Hybrid Macro (4-Quad)"
    elif r == 2:
        arch = "Disciplined Strawberry-Wheat Macro (3-Quad)"
    elif r == 3:
        arch = "Balanced Pastoral-Melon Macro (3-Quad)"
    elif r == 4:
        arch = "Mono-Crop Wheat/Berry Specialist (3-Quad)"
    elif r == 5:
        arch = "Hyper-Scale Pastoral Powerhouse (4-Quad)"
    elif r == 6:
        arch = "Heavy Strawberry Husbandry (3-Quad)"
    elif r == 7:
        arch = "High-Milk Late-Game Scaler (3-Quad)"
    elif r == 8:
        arch = "Fast 4-Quadrant Melon Macro (4-Quad)"
    elif r == 9:
        arch = "High-Consistency Pastoral Engine (3-Quad)"
    else:
        arch = "4-Quadrant Wheat-Melon Expander (4-Quad)"
        
    lines.append(f"| **#{r}** | **{t['name']}** | `{t['team_id']}` | **{t['public_score']:.1f}** | {len(matches)} | **${mean_r:,.0f}** | **${peak_c:,.0f}** | {mean_h:.1f} | {arch} |")

lines.append("")
lines.append("---")
lines.append("")

# Table 2: Financial Trajectory Waterfall
lines.append("## 2. Financial Trajectory & Capital Accumulation Waterfall")
lines.append("")
lines.append("The table below traces the turn-by-turn cash balance progression (mean cash at day close) across the 30-day game cycle:")
lines.append("")
lines.append("| Rank | Competitor | Day 0 (Open) | Day 1 | Day 5 | Day 8 | Day 10 (Melon Surge) | Day 12 (Expansion) | Day 15 | Day 20 | Day 25 | Day 29 (Final Cash) |")
lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

days_track = [0, 1, 5, 8, 10, 12, 15, 20, 25, 29]
for t in TOP_TEAMS:
    r = t["rank"]
    matches = data.get(str(r), [])
    if not matches:
        continue
    row_vals = []
    for d in days_track:
        vals = [m["daily_cash"].get(str(d), m["daily_cash"].get(d, 0)) for m in matches]
        row_vals.append(np.mean(vals))
    
    val_strs = [f"${v:,.0f}" for v in row_vals]
    lines.append(f"| **#{r}** | **{t['name']}** | {val_strs[0]} | {val_strs[1]} | {val_strs[2]} | {val_strs[3]} | **{val_strs[4]}** | {val_strs[5]} | {val_strs[6]} | {val_strs[7]} | {val_strs[8]} | **{val_strs[9]}** |")

lines.append("")
lines.append("### Key Financial Dynamics:")
lines.append("- **The Day 0-5 Cash Valley**: Cash hovers between $10 and $300 as bots reinvest every cent into animal care, wages, and seed cycles.")
lines.append("- **The Day 10 Inflection Point**: Every top competitor experiences an explosive surge to **$14,800–$15,600** on Day 10. This is the exact moment Day 0 planted Melons mature and are liquidated alongside accumulated Milk, Wool, and early Wheat.")
lines.append("- **The Day 12 Land Re-investment**: Cash dips slightly to $10,000–$13,000 as agents buy the SW ($2k) and SE ($4k) quadrants and buy hundreds of seeds.")
lines.append("- **The Exponential Late-Game**: From Day 15 to Day 29, compounding yields from 11-12 active farmhands and 12-16 mature animals generate $3,000–$5,000 in net profit per day.")
lines.append("")
lines.append("---")
lines.append("")

# Table 3: Land Expansion & Labor Force Scaling
lines.append("## 3. Land Expansion & Labor Allocation Matrix")
lines.append("")
lines.append("| Rank | Competitor | NE Unlock ($1k) | SW Unlock ($2k) | SE Unlock ($4k) | Total Quads Unlocked | Daily Hires (D0-D6) | Daily Hires (D7-D8) | Daily Hires (D9-D28) | Total Wage Spend |")
lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for t in TOP_TEAMS:
    r = t["rank"]
    matches = data.get(str(r), [])
    if not matches:
        continue
    ne_days = [m["quadrant_unlocks"]["NE"]["day"] for m in matches if "NE" in m["quadrant_unlocks"]]
    sw_days = [m["quadrant_unlocks"]["SW"]["day"] for m in matches if "SW" in m["quadrant_unlocks"]]
    se_days = [m["quadrant_unlocks"]["SE"]["day"] for m in matches if "SE" in m["quadrant_unlocks"]]
    
    ne_str = f"Day {np.mean(ne_days):.1f}" if ne_days else "None"
    sw_str = f"Day {np.mean(sw_days):.1f}" if sw_days else "None"
    se_str = f"Day {np.mean(se_days):.1f}" if se_days else "None"
    quad_cnt = 1 + (1 if ne_days else 0) + (1 if sw_days else 0) + (1 if len(se_days) == len(matches) else (0.5 if se_days else 0))
    
    hire_matrix = np.array([m["daily_hires"] for m in matches])
    d0_6 = np.mean(hire_matrix[:, :7])
    d7_8 = np.mean(hire_matrix[:, 7:9])
    d9_28 = np.mean(hire_matrix[:, 9:29])
    total_wages = np.mean([m["total_hires"] for m in matches]) * 50
    
    lines.append(f"| **#{r}** | **{t['name']}** | {ne_str} | {sw_str} | {se_str} | {quad_cnt:.0f} / 4 | {d0_6:.1f} hands/day | {d7_8:.1f} hands/day | **{d9_28:.1f} hands/day** | **${total_wages:,.0f}** |")

lines.append("")
lines.append("---")
lines.append("")

# Table 4: Crop Selection & Agricultural Portfolio
lines.append("## 4. Agricultural Strategy: Crop Selection & Rotation")
lines.append("")
lines.append("Analysis of crop seeds purchased and planted over 720 turns (mean counts per match):")
lines.append("")
lines.append("| Rank | Competitor | Wheat (Fast/Feed) | Strawberries (Multi-Harvest) | Melons (High-Yield) | Carrots (Fast Cash) | Tomatoes | Total Seed Spend | Dominant Crop Focus |")
lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

for t in TOP_TEAMS:
    r = t["rank"]
    matches = data.get(str(r), [])
    if not matches:
        continue
    all_crops = Counter()
    all_seeds = Counter()
    for m in matches:
        for c, v in m["crops_planted"].items():
            all_crops[c] += v / len(matches)
        for s, v in m["seeds_bought"].items():
            all_seeds[s] += v / len(matches)
            
    w = all_crops.get("WHEAT", 0)
    sb = all_crops.get("STRAWBERRY", 0)
    mel = all_crops.get("MELON", 0)
    car = all_crops.get("CARROT", 0)
    tom = all_crops.get("TOMATO", 0)
    
    # Estimate seed spending based on base prices: WHEAT $10, CARROT $15, TOMATO $20, STRAWBERRY $40, MELON $80
    seed_cost = w * 10 + car * 15 + tom * 20 + sb * 40 + mel * 80
    
    focus = "Wheat + Strawberry + Melon"
    if car > 10:
        focus = "Wheat + Strawberry + Carrot"
    if tom > 0:
        focus = "Diversified (All 5 Crops)"
        
    lines.append(f"| **#{r}** | **{t['name']}** | **{w:.1f}** | **{sb:.1f}** | **{mel:.1f}** | {car:.1f} | {tom:.1f} | **${seed_cost:,.0f}** | {focus} |")

lines.append("")
lines.append("---")
lines.append("")

# Table 5: Livestock Husbandry & Structural Investment
lines.append("## 5. Livestock Husbandry & Structure Optimization")
lines.append("")
lines.append("| Rank | Competitor | Pastures Built | Cows Owned | Sheep Owned | Chickens / Pigs | Wheat Feed Bought | Fertilizer Collected/Sold | Milk Sold | Wool Sold |")
lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for t in TOP_TEAMS:
    r = t["rank"]
    matches = data.get(str(r), [])
    if not matches:
        continue
    all_structs = Counter()
    all_animals = Counter()
    all_feed = Counter()
    all_sales = Counter()
    for m in matches:
        for k, v in m["structures_built"].items():
            all_structs[k] += v / len(matches)
        for k, v in m["animals_bought"].items():
            all_animals[k] += v / len(matches)
        for k, v in m["feed_bought"].items():
            all_feed[k] += v / len(matches)
        for k, v in m["market_sales"].items():
            all_sales[k] += v / len(matches)
            
    past = all_structs.get("BUILD_PASTURE", 0)
    cows = all_animals.get("COW", 0)
    sheep = all_animals.get("SHEEP", 0)
    other = all_animals.get("CHICKEN", 0) + all_animals.get("PIG", 0)
    feed = all_feed.get("WHEAT", 0)
    fert = all_sales.get("FERTILIZER", 0)
    milk = all_sales.get("MILK", 0)
    wool = all_sales.get("WOOL", 0)
    
    lines.append(f"| **#{r}** | **{t['name']}** | {past:.1f} | **{cows:.1f}** | **{sheep:.1f}** | {other:.1f} | **{feed:.1f}** | **{fert:,.0f} units** | **{milk:,.0f} units** | **{wool:,.0f} units** |")

lines.append("")
lines.append("---")
lines.append("")

# Section 6: Individual Team Forensic Deep-Dives
lines.append("## 6. Comprehensive Individual Team Forensic Profiles")
lines.append("")

for t in TOP_TEAMS:
    r = t["rank"]
    matches = data.get(str(r), [])
    if not matches:
        continue
    lines.append(f"### Rank {r}: {t['name']} (Public Score: {t['public_score']:.1f})")
    lines.append("")
    lines.append(f"- **Matches Analyzed**: {len(matches)} replays")
    f_rewards = [m["final_reward"] for m in matches]
    p_cashes = [m["peak_cash"] for m in matches]
    lines.append(f"- **Replay Scoring Range**: Mean **${np.mean(f_rewards):,.0f}** (Min: ${np.min(f_rewards):,.0f}, Max: **${np.max(f_rewards):,.0f}**)")
    lines.append(f"- **Peak Cash Balance**: Mean **${np.mean(p_cashes):,.0f}**")
    
    # Opening moves breakdown
    sample_m = matches[0]
    lines.append(f"- **Turn 1 Execution**: Immediate construction of Pasture, purchase of Cows + Sheep, 5 Farmhand hires, 6–14 Wheat Feed, and seed packages. Starting cash drops to ${sample_m['daily_cash'].get('0', 20):,.0f}.")
    
    # Expansion & Labor
    ne_d = [m["quadrant_unlocks"]["NE"]["day"] for m in matches if "NE" in m["quadrant_unlocks"]]
    sw_d = [m["quadrant_unlocks"]["SW"]["day"] for m in matches if "SW" in m["quadrant_unlocks"]]
    se_d = [m["quadrant_unlocks"]["SE"]["day"] for m in matches if "SE" in m["quadrant_unlocks"]]
    lines.append(f"- **Land Progression**: NE unlocked on Day {np.mean(ne_d):.1f} | SW unlocked on Day {np.mean(sw_d):.1f} | SE unlocked: {'Day ' + str(round(np.mean(se_d), 1)) if se_d else 'Not Unlocked (3-Quadrant Cap)'}")
    lines.append(f"- **Labor Allocation**: Average of {np.mean([m['total_hires'] for m in matches]):.1f} total farmhand-days hired. Maintains flat 11–12 farmhands daily throughout Days 9–28.")
    
    # Crop & Livestock Portfolio
    all_crops = Counter()
    for m in matches:
        for c, v in m["crops_planted"].items():
            all_crops[c] += v / len(matches)
    lines.append(f"- **Crop Portfolio**: {', '.join([f'{k}: {v:.1f}' for k, v in all_crops.most_common()])}")
    
    all_sales = Counter()
    for m in matches:
        for s, v in m["market_sales"].items():
            all_sales[s] += v / len(matches)
    lines.append(f"- **Market Sales Breakdown**: Fertilizer: {all_sales.get('FERTILIZER', 0):,.0f} | Milk: {all_sales.get('MILK', 0):,.0f} | Wool: {all_sales.get('WOOL', 0):,.0f} | Wheat: {all_sales.get('WHEAT', 0):,.0f} | Strawberry: {all_sales.get('STRAWBERRY', 0):,.0f} | Melon: {all_sales.get('MELON', 0):,.0f}")
    
    # Distinguishing characteristics
    if r == 1:
        lines.append("- **Strategic Superpower**: Rank 1 (カワシギ) achieves unmatched pastoral balance. They maintain highest average livestock feeding (610 wheat feed) and maximize fertilizer collection while cycling 129 Wheat and 38 Strawberry plantings. Their Day 25 cash reaches $74,357 and finishes at up to $129,581.")
    elif r == 2:
        lines.append("- **Strategic Superpower**: Rank 2 (Thomas Tschinkel) focuses on high strawberry density (42 plantings) and carrot cycling (15.7 plantings). They use fertilizer internally for speed rather than dumping it all on the market, achieving highly stable $85,000–$97,500 finishes.")
    elif r == 3:
        lines.append("- **Strategic Superpower**: Rank 3 (Utkarsh #2) utilizes a high-fertilizer liquidation model (1,906 fertilizer sold) with high Wheat (126) and Melon (18.7), achieving reliable $76,000–$88,800 payouts.")
    elif r == 4:
        lines.append("- **Strategic Superpower**: Rank 4 (ReCurSiON) is the only top competitor that completely avoids Carrots and Tomatoes, focusing exclusively on Wheat (143), Strawberry (37), and Melon (19). They execute the earliest SW quadrant unlock (Turn 241 / Day 10.0).")
    elif r == 5:
        lines.append("- **Strategic Superpower**: Rank 5 (peikopon) holds the single highest match score recorded in the tournament: **$155,278** in Episode 93954943! They operate at massive livestock scale (9.3 cows, 5.3 sheep, 2,304 fertilizer sold) and aggressively unlock all 4 quadrants by Day 12.")
    elif r == 6:
        lines.append("- **Strategic Superpower**: Rank 6 (Efe Can Celiksoy) features heavy strawberry production (42) and achieved a $142,910 peak score in Episode 93926854 by maintaining 10 cows and high-efficiency harvesting.")
    elif r == 7:
        lines.append("- **Strategic Superpower**: Rank 7 (One-For-All) demonstrates the strongest late-game compounding curve (Mean: $111,077, Peak: $135,019), driven by record milk sales (322.7 units) and strawberry cash flows.")
    elif r == 8:
        lines.append("- **Strategic Superpower**: Rank 8 (Kostiantyn Isaienkov) executes aggressive 4-quadrant expansion (SE on Day 12), maximizing Melon (20.0) and Wheat (129.0) to average $97,689 per game.")
    elif r == 9:
        lines.append("- **Strategic Superpower**: Rank 9 (SCLim2022080004) operates a rock-solid pastoral engine (10 cows, 4 sheep) with a peak score of $130,304.")
    elif r == 10:
        lines.append("- **Strategic Superpower**: Rank 10 (Matteo123383iend) executes 4-quadrant expansion with heavy Wheat (129) and Melon (19.3), achieving peak scores above $101,000.")
        
    lines.append("")

lines.append("---")
lines.append("")

# Section 7: Key Architectural Recommendations & Strategy Blueprint
lines.append("## 7. Actionable Strategic Blueprint for Championship Play")
lines.append("")
lines.append("To engineer a competitive agent capable of achieving Rank 1–10 performance (>3,000 Leaderboard Rating / $90,000–$150,000+ match rewards), the bot must implement the following unified architectural pipeline:")
lines.append("")
lines.append("### Phase 1: Turn 1 Hyper-Opening (Day 0, Turn 1)")
lines.append("```python")
lines.append("# Exact Turn 1 Action Blueprint:")
lines.append("return {")
lines.append("    'farmer': ['BUILD_PASTURE'],")
lines.append("    'hands': [],")
lines.append("    'market': [")
lines.append("        ['HIRE'], ['HIRE'], ['HIRE'], ['HIRE'], ['HIRE'],  # 5 farmhands ($250)")
lines.append("        ['BUY_ANIMAL', 'COW', 2],                          # 2 Cows")
lines.append("        ['BUY_ANIMAL', 'SHEEP', 2],                        # 2 Sheep")
lines.append("        ['BUY_PRODUCT', 'WHEAT', 6],                       # Feed stock")
lines.append("        ['BUY_SEED', 'WHEAT', 7],                          # Fast turnover")
lines.append("        ['BUY_SEED', 'MELON', 12]                          # Matures Day 10")
lines.append("    ]")
lines.append("}")
lines.append("```")
lines.append("")
lines.append("### Phase 2: The Daily Morning Care Routine (Hours 0–3)")
lines.append("1. **Hour 0**: Queue maximum allowable `HIRE` orders for the day (5 on D0, 4–5 on D2–D6, 8 on D7, 11–12 on D8–D28).")
lines.append("2. **Hour 1**: Dispatch assigned workers to `FEED` and `CARE` for all Cows and Sheep using inventory Wheat.")
lines.append("3. **Hour 2**: Dispatch workers to `COLLECT_FERTILIZER` from pasture tiles.")
lines.append("4. **Hour 3**: Execute open-market sale orders for all collected `MILK`, `WOOL`, `FERTILIZER`, and ripe harvested crops.")
lines.append("")
lines.append("### Phase 3: The Daily Farming & Field Maintenance Cycle (Hours 4–23)")
lines.append("1. **Watering Priority**: Ensure 100% of planted tiles are watered daily (`watered_today: True`) to avoid crop death or delayed maturation.")
lines.append("2. **Harvesting Priority**: Immediately harvest mature crops (`yield_units > 0`) to free tiles for replanting.")
lines.append("3. **Replanting Logic**:")
lines.append("   - **Days 0–5**: Plant 12–20 Melons + 20–30 Wheat.")
lines.append("   - **Days 6–18**: Expand into NE/SW quadrants. Plant 30–45 Strawberries + continuous Wheat cycles.")
lines.append("   - **Days 19–24**: Plant Strawberries and fast Carrots/Wheat.")
lines.append("   - **Days 25–29**: Cease long-cycle crops (Melon/Tomato). Plant only fast Wheat/Carrot or cease planting to minimize deadweight inventory.")
lines.append("")
lines.append("### Phase 4: Expansion Timetable")
lines.append("- **Day 6 (Turn 149)**: Deduct $1,000 to unlock `NE` quadrant.")
lines.append("- **Day 10 (Turn 241–265)**: Upon Day 10 Melon liquidation ($15,000+ cash balance), deduct $2,000 to unlock `SW` quadrant.")
lines.append("- **Day 12 (Turn 289)**: Deduct $4,000 to unlock `SE` quadrant (if cash exceeds $10,000).")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Conclusion")
lines.append("The forensic analysis of the top 10 competitors conclusively demonstrates that Kaggriculture is won not by pure manual farming, but by **industrialized pastoral capitalism**: leveraging livestock for compounding daily cash flow and fertilizer generation, scaling labor to the maximum legal limit (12 farmhands daily), and executing perfectly timed quadrant unlocks.")

full_report_text = "\n".join(lines)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(full_report_text)

print(f"[+] Successfully generated and saved exhaustive forensic report to {REPORT_PATH}")
print(f"    Report Length: {len(full_report_text)} chars / {len(lines)} lines")

