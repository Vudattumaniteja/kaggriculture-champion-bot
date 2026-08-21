"""
Tournament Runner for Heuristic Specialist Archetypes.
Runs 10 pairwise matches among the 5 archetypes:
1. DeterministicGrandmaster
2. CarrotMonoculture
3. MelonRusher
4. DairySyndicate
5. TownShopSaturator

Saves standalone HTML and JSON replays, and creates an interactive index.html dashboard.
"""

import sys
from pathlib import Path
import json
import time
from itertools import combinations
from typing import Dict, List, Any

# Ensure project imports resolve
prototype_dir = Path(__file__).resolve().parent
grilling_model_dir = prototype_dir.parent
if str(grilling_model_dir) not in sys.path:
    sys.path.insert(0, str(grilling_model_dir))

from kaggle_environments import make
from prototype.specialists import (
    SPECIALIST_CONFIGS,
    get_specialist,
)


def run_specialist_tournament():
    replays_dir = prototype_dir / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    archetypes = [
        "DeterministicGrandmaster",
        "CarrotMonoculture",
        "MelonRusher",
        "DairySyndicate",
        "TownShopSaturator",
    ]

    # 5 choose 2 = 10 unique pairwise pairings
    pairings = list(combinations(archetypes, 2))
    print(f"Starting 10-Match Specialist Tournament among {len(archetypes)} archetypes...")

    stats: Dict[str, Dict[str, Any]] = {
        name: {
            "name": name,
            "description": SPECIALIST_CONFIGS[name].description,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "matches": 0,
            "total_score": 0.0,
            "scores": [],
        }
        for name in archetypes
    }

    match_results: List[Dict[str, Any]] = []

    for idx, (p1_name, p2_name) in enumerate(pairings, start=1):
        match_id = f"match_{idx:02d}_{p1_name}_vs_{p2_name}"
        html_filename = f"{match_id}.html"
        json_filename = f"{match_id}.json"
        html_path = replays_dir / html_filename
        json_path = replays_dir / json_filename

        print(f"\n[{idx}/10] Playing: {p1_name} (P1) vs {p2_name} (P2)...")
        agent1 = get_specialist(p1_name)
        agent2 = get_specialist(p2_name)

        t_start = time.time()
        env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
        env.run([agent1, agent2])
        elapsed = time.time() - t_start

        # Extract final rewards (cash balance)
        step_final = env.steps[-1]
        p1_reward = float(step_final[0].reward if step_final[0].reward is not None else 0.0)
        p2_reward = float(step_final[1].reward if step_final[1].reward is not None else 0.0)

        if p1_reward > p2_reward:
            winner = p1_name
            stats[p1_name]["wins"] += 1
            stats[p2_name]["losses"] += 1
        elif p2_reward > p1_reward:
            winner = p2_name
            stats[p2_name]["wins"] += 1
            stats[p1_name]["losses"] += 1
        else:
            winner = "TIE"
            stats[p1_name]["ties"] += 1
            stats[p2_name]["ties"] += 1

        stats[p1_name]["matches"] += 1
        stats[p2_name]["matches"] += 1
        stats[p1_name]["total_score"] += p1_reward
        stats[p2_name]["total_score"] += p2_reward
        stats[p1_name]["scores"].append(p1_reward)
        stats[p2_name]["scores"].append(p2_reward)

        # Save HTML replay
        html_content = env.render(mode="html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Save JSON replay
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(env.toJSON(), f)

        match_info = {
            "match_num": idx,
            "match_id": match_id,
            "player1": p1_name,
            "player2": p2_name,
            "p1_reward": p1_reward,
            "p2_reward": p2_reward,
            "winner": winner,
            "elapsed_seconds": round(elapsed, 2),
            "html_file": html_filename,
            "json_file": json_filename,
        }
        match_results.append(match_info)
        print(f"    Finished in {elapsed:.2f}s | Result: {p1_name} (${p1_reward:,.0f}) vs {p2_name} (${p2_reward:,.0f}) -> Winner: {winner}")

    # Compute averages
    leaderboard = []
    for name, data in stats.items():
        avg_score = data["total_score"] / max(1, data["matches"])
        win_rate = (data["wins"] / max(1, data["matches"])) * 100.0
        leaderboard.append({
            "name": name,
            "description": data["description"],
            "wins": data["wins"],
            "losses": data["losses"],
            "ties": data["ties"],
            "matches": data["matches"],
            "win_rate": round(win_rate, 1),
            "avg_score": round(avg_score, 1),
            "total_score": round(data["total_score"], 1),
        })

    # Sort leaderboard by wins descending, then avg_score descending
    leaderboard.sort(key=lambda x: (x["wins"], x["avg_score"]), reverse=True)

    # Generate interactive HTML dashboard
    generate_dashboard_html(replays_dir / "index.html", leaderboard, match_results)
    print(f"\nTournament complete! All 10 matches saved.")
    print(f"Replay dashboard generated at: {replays_dir / 'index.html'}")


def generate_dashboard_html(output_file: Path, leaderboard: List[Dict[str, Any]], matches: List[Dict[str, Any]]):
    rows_leaderboard = ""
    for rank, item in enumerate(leaderboard, start=1):
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"#{rank}"))
        rows_leaderboard += f"""
        <tr>
            <td style="text-align:center; font-weight:bold;">{medal}</td>
            <td><strong>{item['name']}</strong><br><small style="color:#666;">{item['description']}</small></td>
            <td style="text-align:center;">{item['wins']} - {item['losses']} - {item['ties']}</td>
            <td style="text-align:center; font-weight:bold; color: #2e7d32;">{item['win_rate']}%</td>
            <td style="text-align:right; font-weight:bold;">${item['avg_score']:,.2f}</td>
            <td style="text-align:right;">${item['total_score']:,.2f}</td>
        </tr>
        """

    rows_matches = ""
    for m in matches:
        p1_win = m['winner'] == m['player1']
        p2_win = m['winner'] == m['player2']
        p1_style = "font-weight:bold; color:#1b5e20;" if p1_win else "color:#333;"
        p2_style = "font-weight:bold; color:#1b5e20;" if p2_win else "color:#333;"

        rows_matches += f"""
        <tr>
            <td style="text-align:center;"><strong>Match {m['match_num']:02d}</strong></td>
            <td style="{p1_style}">{m['player1']}<br><span style="font-size:0.9em; color:#555;">${m['p1_reward']:,.0f}</span></td>
            <td style="text-align:center; font-weight:bold; color:#888;">VS</td>
            <td style="{p2_style}">{m['player2']}<br><span style="font-size:0.9em; color:#555;">${m['p2_reward']:,.0f}</span></td>
            <td style="text-align:center;"><span class="badge {'winner-badge' if m['winner'] != 'TIE' else 'tie-badge'}">{m['winner']}</span></td>
            <td style="text-align:center;">{m['elapsed_seconds']}s</td>
            <td style="text-align:center;">
                <a href="{m['html_file']}" target="_blank" class="btn-replay">Open Replay</a>
            </td>
        </tr>
        """

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kaggriculture Specialists Tournament - 10 Match Replays</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f4f6f9;
            color: #212529;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 32px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 24px;
        }}
        header h1 {{
            margin: 0 0 8px 0;
            font-size: 28px;
        }}
        header p {{
            margin: 0;
            opacity: 0.9;
            font-size: 15px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .card h2 {{
            margin-top: 0;
            margin-bottom: 16px;
            font-size: 20px;
            border-bottom: 2px solid #eef2f7;
            padding-bottom: 10px;
            color: #1e3c72;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
        }}
        th, td {{
            padding: 12px 14px;
            border-bottom: 1px solid #eef2f7;
            vertical-align: middle;
        }}
        th {{
            background-color: #f8fafc;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        tr:hover {{
            background-color: #f8fafc;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        .winner-badge {{
            background-color: #dcfce7;
            color: #166534;
        }}
        .tie-badge {{
            background-color: #f1f5f9;
            color: #475569;
        }}
        .btn-replay {{
            display: inline-block;
            background-color: #2563eb;
            color: white;
            padding: 6px 14px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 500;
            font-size: 13px;
            transition: background-color 0.2s;
        }}
        .btn-replay:hover {{
            background-color: #1d4ed8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Kaggriculture Heuristic Specialists Tournament</h1>
            <p>5 Parameterized Archetypes &bull; 10 Pairwise 720-Turn Matches &bull; Full Browser Replay Viewer</p>
        </header>

        <div class="card">
            <h2>Tournament Standings</h2>
            <table>
                <thead>
                    <tr>
                        <th style="text-align:center; width: 60px;">Rank</th>
                        <th>Archetype</th>
                        <th style="text-align:center;">W - L - T</th>
                        <th style="text-align:center;">Win Rate</th>
                        <th style="text-align:right;">Avg Final Bank</th>
                        <th style="text-align:right;">Total Bank</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_leaderboard}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>All 10 Match Replays</h2>
            <table>
                <thead>
                    <tr>
                        <th style="text-align:center; width: 90px;">Match</th>
                        <th>Player 1 (North)</th>
                        <th style="text-align:center; width: 40px;"></th>
                        <th>Player 2 (South)</th>
                        <th style="text-align:center;">Winner</th>
                        <th style="text-align:center; width: 80px;">Runtime</th>
                        <th style="text-align:center; width: 130px;">Action</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_matches}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)


if __name__ == "__main__":
    run_specialist_tournament()
