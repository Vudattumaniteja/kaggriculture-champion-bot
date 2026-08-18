"""
Local simulation, benchmarking, and tournament evaluation harness for Kaggriculture.
"""

import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from kaggle_environments import make

ITEM_COSTS = {
    "WHEAT_SEED": 10,
    "CARROT_SEED": 20,
    "TOMATO_SEED": 50,
    "STRAWBERRY_SEED": 100,
    "MELON_SEED": 80,
    "GOOSE": 300,
    "COW": 400,
    "SHEEP": 500,
    "FERTILIZER": 100,
    "WHEAT": 25,
    "CARROT": 35,
    "TOMATO": 60,
    "STRAWBERRY": 120,
    "MELON": 250,
    "EGG": 50,
    "MILK": 160,
    "WOOL": 200,
}


def calculate_trapped_deadweight(obs: Dict[str, Any], player_idx: int) -> float:
    """Calculates the dollar value of non-liquidated capital trapped at the end of the match."""
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player_idx] if player_idx < len(farms) else {}
    private = obs.get("private", {}) or {}

    deadweight = 0.0

    # Seeds
    seeds = private.get("seeds", {})
    for crop, count in seeds.items():
        if count > 0:
            seed_key = f"{crop}_SEED"
            deadweight += count * ITEM_COSTS.get(seed_key, 20)

    # Shed inventory
    shed = private.get("shed", {})
    for item, count in shed.items():
        if count > 0:
            deadweight += count * ITEM_COSTS.get(item, 50)

    # Unharvested crops
    tiles = my_farm.get("tiles", [])
    for row in tiles:
        for t in row:
            if isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    crop = t.get("crop", "CARROT")
                    deadweight += ITEM_COSTS.get(crop, 35)

    return deadweight


@dataclass
class MatchResult:
    episode_id: int
    agent0_name: str
    agent1_name: str
    agent0_reward: float
    agent1_reward: float
    agent0_deadweight: float
    agent1_deadweight: float
    agent0_status: str
    agent1_status: str
    winner: Optional[int]
    duration_seconds: float
    replay_html_path: Optional[str] = None
    replay_json_path: Optional[str] = None


@dataclass
class TournamentSummary:
    agent0_name: str
    agent1_name: str
    total_episodes: int
    agent0_wins: int = 0
    agent1_wins: int = 0
    ties: int = 0
    agent0_rewards: List[float] = field(default_factory=list)
    agent1_rewards: List[float] = field(default_factory=list)
    agent0_deadweights: List[float] = field(default_factory=list)
    agent1_deadweights: List[float] = field(default_factory=list)
    matches: List[MatchResult] = field(default_factory=list)

    @property
    def agent0_win_rate(self) -> float:
        return (self.agent0_wins / self.total_episodes) * 100 if self.total_episodes > 0 else 0.0

    @property
    def agent1_win_rate(self) -> float:
        return (self.agent1_wins / self.total_episodes) * 100 if self.total_episodes > 0 else 0.0

    @property
    def tie_rate(self) -> float:
        return (self.ties / self.total_episodes) * 100 if self.total_episodes > 0 else 0.0

    @property
    def agent0_mean_reward(self) -> float:
        return float(np.mean(self.agent0_rewards)) if self.agent0_rewards else 0.0

    @property
    def agent1_mean_reward(self) -> float:
        return float(np.mean(self.agent1_rewards)) if self.agent1_rewards else 0.0

    @property
    def agent0_mean_deadweight(self) -> float:
        return float(np.mean(self.agent0_deadweights)) if self.agent0_deadweights else 0.0

    @property
    def agent1_mean_deadweight(self) -> float:
        return float(np.mean(self.agent1_deadweights)) if self.agent1_deadweights else 0.0

    @property
    def mean_margin(self) -> float:
        return self.agent0_mean_reward - self.agent1_mean_reward

    def format_markdown_table(self) -> str:
        lines = [
            f"# Tournament Evaluation: {self.agent0_name} vs {self.agent1_name}",
            "",
            f"**Total Episodes:** {self.total_episodes} (720 turns/episode)",
            f"**Win Rate ({self.agent0_name}):** {self.agent0_win_rate:.1f}% ({self.agent0_wins}W / {self.agent1_wins}L / {self.ties}T)",
            f"**Mean Final Bank:** ${self.agent0_mean_reward:,.1f} vs ${self.agent1_mean_reward:,.1f} (Margin: ${self.mean_margin:+,.1f})",
            f"**Mean Trapped Deadweight:** ${self.agent0_mean_deadweight:,.1f} ({self.agent0_name}) vs ${self.agent1_mean_deadweight:,.1f} ({self.agent1_name})",
            "",
            "## Aggregate Statistics",
            "",
            "| Agent | Win Rate | Wins | Losses | Ties | Mean Final Bank | Min Bank | Max Bank | Mean Deadweight | Std Dev |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| **{self.agent0_name}** | **{self.agent0_win_rate:.1f}%** | {self.agent0_wins} | {self.agent1_wins} | {self.ties} | ${self.agent0_mean_reward:,.1f} | ${min(self.agent0_rewards):,.1f} | ${max(self.agent0_rewards):,.1f} | ${self.agent0_mean_deadweight:,.1f} | ${float(np.std(self.agent0_rewards)):,.1f} |",
            f"| **{self.agent1_name}** | **{self.agent1_win_rate:.1f}%** | {self.agent1_wins} | {self.agent0_wins} | {self.ties} | ${self.agent1_mean_reward:,.1f} | ${min(self.agent1_rewards):,.1f} | ${max(self.agent1_rewards):,.1f} | ${self.agent1_mean_deadweight:,.1f} | ${float(np.std(self.agent1_rewards)):,.1f} |",
            "",
            "## Match-by-Match Breakdown",
            "",
            "| Ep | Position | " + f"{self.agent0_name} Bank" + " | " + f"{self.agent1_name} Bank" + " | " + f"{self.agent0_name} DW" + " | " + f"{self.agent1_name} DW" + " | Winner | Duration |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for m in self.matches:
            w_str = self.agent0_name if m.winner == 0 else (self.agent1_name if m.winner == 1 else "TIE")
            lines.append(
                f"| #{m.episode_id} | Direct | ${m.agent0_reward:,.1f} | ${m.agent1_reward:,.1f} | ${m.agent0_deadweight:,.1f} | ${m.agent1_deadweight:,.1f} | **{w_str}** | {m.duration_seconds:.2f}s |"
            )

        return "\n".join(lines)


def resolve_agent(agent_spec: Any) -> Tuple[Any, str]:
    """Resolves an agent string (built-in name, file path, module name) or callable into a callable and display name."""
    if callable(agent_spec):
        return agent_spec, getattr(agent_spec, "__name__", "custom_agent")

    if isinstance(agent_spec, str):
        # 1. Built-in agent
        if agent_spec in ["pass", "random", "starter"]:
            return agent_spec, agent_spec

        # 2. File path (.py)
        if os.path.exists(agent_spec):
            name = os.path.splitext(os.path.basename(agent_spec))[0]
            spec = importlib.util.spec_from_file_location(name, os.path.abspath(agent_spec))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
                if hasattr(mod, "agent"):
                    return getattr(mod, "agent"), name
                elif hasattr(mod, "balanced_farmer_agent"):
                    return getattr(mod, "balanced_farmer_agent"), name

        # 3. Dotted module path
        try:
            mod = importlib.import_module(agent_spec)
            name = agent_spec.split(".")[-1]
            if hasattr(mod, "agent"):
                return getattr(mod, "agent"), name
            elif hasattr(mod, "balanced_farmer_agent"):
                return getattr(mod, "balanced_farmer_agent"), name
        except Exception:
            pass

        return agent_spec, agent_spec

    return agent_spec, "custom_agent"


def run_match(
    agent0: Any,
    agent1: Any,
    episode_id: int = 1,
    episode_steps: int = 720,
    debug: bool = False,
    save_html_dir: Optional[str] = None,
    save_json_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> MatchResult:
    """Simulates a single head-to-head match between two agents."""
    resolved_agent0, name0 = resolve_agent(agent0)
    resolved_agent1, name1 = resolve_agent(agent1)

    config = {"episodeSteps": episode_steps}
    if seed is not None:
        config["seed"] = seed

    env = make("kaggriculture", configuration=config, debug=debug)

    start_t = time.time()
    env.run([resolved_agent0, resolved_agent1])
    duration = time.time() - start_t

    final_step = env.steps[-1]
    reward0 = float(final_step[0].reward) if final_step[0].reward is not None else 0.0
    reward1 = float(final_step[1].reward) if final_step[1].reward is not None else 0.0
    status0 = str(final_step[0].status)
    status1 = str(final_step[1].status)

    obs0 = final_step[0].observation
    obs1 = final_step[1].observation

    dw0 = calculate_trapped_deadweight(obs0, player_idx=0)
    dw1 = calculate_trapped_deadweight(obs1, player_idx=1)

    if reward0 > reward1:
        winner = 0
    elif reward1 > reward0:
        winner = 1
    else:
        winner = None

    html_path = None
    if save_html_dir:
        os.makedirs(save_html_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join(save_html_dir, f"replay_ep{episode_id}_{name0}_vs_{name1}_{timestamp}.html")
        try:
            rendered_html = env.render(mode="html", width=1000, height=800)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(rendered_html)
        except Exception as e:
            print(f"Warning: HTML render failed: {e}")

    json_path = None
    if save_json_dir:
        os.makedirs(save_json_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(save_json_dir, f"replay_ep{episode_id}_{name0}_vs_{name1}_{timestamp}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(env.toJSON(), f)
        except Exception as e:
            print(f"Warning: JSON replay dump failed: {e}")

    return MatchResult(
        episode_id=episode_id,
        agent0_name=name0,
        agent1_name=name1,
        agent0_reward=reward0,
        agent1_reward=reward1,
        agent0_deadweight=dw0,
        agent1_deadweight=dw1,
        agent0_status=status0,
        agent1_status=status1,
        winner=winner,
        duration_seconds=duration,
        replay_html_path=html_path,
        replay_json_path=json_path,
    )


def run_tournament(
    agent0: Any,
    agent1: Any,
    episodes: int = 10,
    episode_steps: int = 720,
    swap_positions: bool = True,
    save_replays: bool = False,
    replays_dir: str = "replays",
    base_seed: int = 42,
) -> TournamentSummary:
    """Runs a batch tournament across N episodes, optionally alternating player positions."""
    _, name0 = resolve_agent(agent0)
    _, name1 = resolve_agent(agent1)

    summary = TournamentSummary(
        agent0_name=name0,
        agent1_name=name1,
        total_episodes=episodes,
    )

    html_dir = os.path.join(replays_dir, "html") if save_replays else None
    json_dir = os.path.join(replays_dir, "json") if save_replays else None

    print(f"=== Starting Tournament: {name0} vs {name1} ({episodes} episodes, {episode_steps} steps) ===")

    for ep in range(1, episodes + 1):
        seed = base_seed + ep
        is_swapped = swap_positions and (ep % 2 == 0)

        if is_swapped:
            res = run_match(
                agent0=agent1,
                agent1=agent0,
                episode_id=ep,
                episode_steps=episode_steps,
                save_html_dir=html_dir if ep <= 3 else None,
                save_json_dir=json_dir if ep <= 3 else None,
                seed=seed,
            )
            r0 = res.agent1_reward
            r1 = res.agent0_reward
            dw0 = res.agent1_deadweight
            dw1 = res.agent0_deadweight
        else:
            res = run_match(
                agent0=agent0,
                agent1=agent1,
                episode_id=ep,
                episode_steps=episode_steps,
                save_html_dir=html_dir if ep <= 3 else None,
                save_json_dir=json_dir if ep <= 3 else None,
                seed=seed,
            )
            r0 = res.agent0_reward
            r1 = res.agent1_reward
            dw0 = res.agent0_deadweight
            dw1 = res.agent1_deadweight

        summary.agent0_rewards.append(r0)
        summary.agent1_rewards.append(r1)
        summary.agent0_deadweights.append(dw0)
        summary.agent1_deadweights.append(dw1)

        if r0 > r1:
            summary.agent0_wins += 1
            w_code = f"{name0} won"
        elif r1 > r0:
            summary.agent1_wins += 1
            w_code = f"{name1} won"
        else:
            summary.ties += 1
            w_code = "TIE"

        match_record = MatchResult(
            episode_id=ep,
            agent0_name=name0,
            agent1_name=name1,
            agent0_reward=r0,
            agent1_reward=r1,
            agent0_deadweight=dw0,
            agent1_deadweight=dw1,
            agent0_status="DONE",
            agent1_status="DONE",
            winner=0 if r0 > r1 else (1 if r1 > r0 else None),
            duration_seconds=res.duration_seconds,
            replay_html_path=res.replay_html_path,
            replay_json_path=res.replay_json_path,
        )
        summary.matches.append(match_record)

        print(
            f"  [Ep {ep:2d}/{episodes}] {name0}: ${r0:,.0f} (DW: ${dw0:.0f}) | "
            f"{name1}: ${r1:,.0f} (DW: ${dw1:.0f}) | -> {w_code} ({res.duration_seconds:.2f}s)"
        )

    print("=== Tournament Complete ===")
    print(
        f"Result: {name0} {summary.agent0_wins}W - {summary.agent1_wins}L - {summary.ties}T "
        f"({summary.agent0_win_rate:.1f}% Win Rate) | Mean: ${summary.agent0_mean_reward:,.0f} vs ${summary.agent1_mean_reward:,.0f} | "
        f"Deadweight: ${summary.agent0_mean_deadweight:,.0f} vs ${summary.agent1_mean_deadweight:,.0f}"
    )

    return summary
