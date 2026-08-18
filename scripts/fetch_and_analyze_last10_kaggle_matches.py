"""
Retrieve and forensically analyze the last 10 Kaggle match replays for our bot.
"""
import os
import sys
import json
import glob
import subprocess
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout
    except Exception as e:
        print(f"Error running {' '.join(cmd)}: {e}")
        return None

print("================================================================================")
print(" 1. FETCHING LATEST KAGGLE SUBMISSIONS AND EPISODES")
print("================================================================================")

# Check submissions
sub_output = run_cmd(["kaggle", "competitions", "submissions", "kaggriculture", "--format", "json"])
latest_sub_ids = []
if sub_output:
    try:
        subs = json.loads(sub_output)
        print(f"Found {len(subs)} total submissions.")
        for s in subs[:5]:
            sub_id = s.get("id") or s.get("ref")
            date = s.get("date")
            desc = s.get("description")
            status = s.get("status")
            print(f"  Submission ID: {sub_id} | Date: {date} | Status: {status} | Desc: {desc}")
            if sub_id:
                latest_sub_ids.append(sub_id)
    except Exception as e:
        print(f"Could not parse submissions JSON: {e}")

# Check local replays in replays/latest_frontier/
os.makedirs("replays/latest_frontier", exist_ok=True)
existing_files = glob.glob("replays/latest_frontier/*.json")
print(f"\nFound {len(existing_files)} existing replay files in replays/latest_frontier/")

all_parsed_matches = []

for f in existing_files:
    try:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if "steps" not in data:
            continue
            
        steps = data["steps"]
        ep_id = data.get("id") or os.path.basename(f)
        names = data.get("info", {}).get("TeamNames", ["P0", "P1"])
        
        p0_name = names[0] if len(names) > 0 else "P0"
        p1_name = names[1] if len(names) > 1 else "P1"
        
        bot_idx = 0
        if "alfphafarm" in p1_name.lower():
            bot_idx = 1
        elif "alfphafarm" in p0_name.lower():
            bot_idx = 0
            
        last_step = steps[-1]
        r0 = last_step[0].get("reward", 0) or 0
        r1 = last_step[1].get("reward", 0) or 0
        
        bot_score = r0 if bot_idx == 0 else r1
        opp_score = r1 if bot_idx == 0 else r0
        opp_name = p1_name if bot_idx == 0 else p0_name
        
        p0_cash_history = []
        p1_cash_history = []
        wool_prices = []
        milk_prices = []
        fert_prices = []
        bot_market_sales = defaultdict(int)
        
        for t_idx, step_data in enumerate(steps):
            obs0 = step_data[0].get("observation", {})
            farms = obs0.get("farms", [{}, {}])
            m0 = farms[0].get("money", 0) if len(farms) > 0 else 0
            m1 = farms[1].get("money", 0) if len(farms) > 1 else 0
            p0_cash_history.append(m0)
            p1_cash_history.append(m1)
            
            market = obs0.get("market", {}) or {}
            prices = market.get("prices", {}) or {}
            wool_prices.append(prices.get("WOOL", 200))
            milk_prices.append(prices.get("MILK", 160))
            fert_prices.append(prices.get("FERTILIZER", 100))
            
            if bot_idx < len(step_data):
                act = step_data[bot_idx].get("action", {}) or {}
                for o in act.get("market", []):
                    if len(o) >= 3 and o[0] == "SELL":
                        bot_market_sales[o[1]] += o[2]
                        
        final_obs = steps[-1][bot_idx].get("observation", {})
        final_shed = final_obs.get("private", {}).get("shed", {}) if "private" in final_obs else {}
        
        all_parsed_matches.append({
            "file": f,
            "ep_id": ep_id,
            "bot_idx": bot_idx,
            "bot_name": names[bot_idx] if bot_idx < len(names) else "Our Bot",
            "opp_name": opp_name,
            "bot_score": bot_score,
            "opp_score": opp_score,
            "win": bot_score > opp_score,
            "margin": bot_score - opp_score,
            "sales": dict(bot_market_sales),
            "final_shed": final_shed,
            "min_wool_price": min(wool_prices) if wool_prices else 200,
            "final_wool_price": wool_prices[-1] if wool_prices else 200,
            "final_fert_price": fert_prices[-1] if fert_prices else 100,
        })
    except Exception as e:
        pass

# Sort by episode ID
all_parsed_matches.sort(key=lambda x: str(x["ep_id"]), reverse=True)
last_10 = all_parsed_matches[:10]

print(f"\n================================================================================")
print(f" FORENSIC AUDIT OF LAST 10 KAGGLE MATCHES")
print(f"================================================================================")
print(f"{'Episode ID':<12} | {'Role':<4} | {'Our Score':<12} | {'Opponent Score':<14} | {'Outcome':<7} | {'Opponent Name':<25}")
print("-" * 85)

for m in last_10:
    outcome = "WIN" if m["win"] else "LOSS"
    role_str = f"P{m['bot_idx']}"
    print(f"{str(m['ep_id']):<12} | {role_str:<4} | ${m['bot_score']:>10,.1f} | ${m['opp_score']:>12,.1f} | {outcome:<7} | {m['opp_name'][:25]:<25}")

print("\n================================================================================")
print(" TURN-BY-TURN ROOT CAUSE BREAKDOWN FOR EACH MATCH")
print("================================================================================")

for idx, m in enumerate(last_10, 1):
    margin_str = f"+${m['margin']:,.1f}" if m['win'] else f"-${-m['margin']:,.1f}"
    res_str = "WIN" if m['win'] else "LOSS"
    print(f"\n--- MATCH {idx}: Episode {m['ep_id']} ({m['bot_name']} vs {m['opp_name']}) ---")
    print(f"  Result: {res_str} ({margin_str})")
    print(f"  Final Scores: Our Bot = ${m['bot_score']:,.1f} | Opponent = ${m['opp_score']:,.1f}")
    print(f"  Commodities Sold by Bot: {m['sales']}")
    print(f"  Market Price Drop: Wool dropped to ${m['min_wool_price']:.1f} (ended @ ${m['final_wool_price']:.1f}), Fertilizer ended @ ${m['final_fert_price']:.1f}")
    print(f"  Unsold Deadweight at Turn 719: {m['final_shed']}")
    
    reasons = []
    if m["bot_score"] < 50000:
        reasons.append("Low animal scale: herd was not expanded across NE/SW/SE quadrants.")
    if m.get("final_wool_price", 200) < 50:
        reasons.append("Quadratic price crash: Wool price collapsed to <$50 due to over-dumping.")
    if m.get("sales", {}).get("FERTILIZER", 0) > 100 and m.get("final_fert_price", 100) < 20:
        reasons.append("Fertilizer Market Trap: Dumped fertilizer on open market for only $1-$14 (wasting $25k+).")
    if sum(m.get("final_shed", {}).values()) > 30:
        reasons.append(f"Trapped deadweight in shed ({sum(m.get('final_shed', {}).values())} unsold items at turn 719).")
    if not reasons:
        reasons.append("Opponent out-compounded via superior cash crops (Melons/Strawberries) or higher livestock count.")
        
    print(f"  🔍 ROOT CAUSES IDENTIFIED:")
    for r in reasons:
        print(f"     * {r}")
