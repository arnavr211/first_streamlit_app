"""
exploration_planner.py

Manages a rotation of analysis "lenses" — distinct analytical perspectives on
the 2025 NFL play-by-play data. Each lens focuses the discovery engine on a
specific angle: situations, matchups, players, game-flow, time trends, etc.

Each day, the planner selects 1–2 lenses that haven't been used recently and
runs them, producing findings CSVs that are compatible with interpret.py.

Once all single lenses have been cycled through, the planner enters a "combo"
phase where it intersects two lenses simultaneously for cross-cutting insights
(e.g., player-level analysis restricted to 2-minute drill situations).

Usage
-----
    python exploration_planner.py              # run today's scheduled lenses
    python exploration_planner.py --status     # show lens usage history
    python exploration_planner.py --list       # list all available lenses
    python exploration_planner.py --lens two-minute-drill   # force a lens
    python exploration_planner.py --combo game-flow,player-level  # force a combo
    python exploration_planner.py --dry-run    # show selection, don't execute
    python exploration_planner.py --interpret  # also run interpret.py on results
    python exploration_planner.py --reset      # clear lens usage history
"""

import argparse
import json
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nfl_data_py as nfl
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from nfldiscovery import (
    MIN_PLAYS, Z_THRESHOLD, TOP_N,
    _METRIC_BLACKLIST, _DIM_BLACKLIST, DIMS_DERIVED,
    _auto_detect_metrics, _auto_detect_dims,
    add_derived_columns, load_plays,
    run_split_analysis, run_correlation_analysis,
    composite_score, make_interpretation, _rank_and_interpret,
)

# ── Paths ───────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent
STATE_FILE    = ROOT / "exploration_state.json"
FINDINGS_DIR  = ROOT / "lens_findings"
FINDINGS_DIR.mkdir(exist_ok=True)

# ── Lens dataclass ──────────────────────────────────────────────────────────────

@dataclass
class Lens:
    id: str
    name: str
    description: str
    category: str          # baseline | situational | matchup | player | flow | sequence
                           # | trend | efficiency | scheme | historical

    # prepare_fn(plays) → filtered/enriched DataFrame. None = no filter.
    prepare_fn: Optional[Callable] = None

    # Override analysis. None = standard split + correlation.
    # Signature: (plays, metrics, dims) → list[dict]
    analysis_fn: Optional[Callable] = None

    # Restrict metrics to this list (None = auto-detect)
    metric_focus: Optional[List[str]] = None

    # Restrict dimensions to this list (None = auto-detect + derived)
    dimension_focus: Optional[List[str]] = None

    # Extra dimensions to add beyond auto-detected (e.g. "defteam")
    extra_dims: Optional[List[str]] = None

    # Minimum play count — skip lens if filtered DataFrame is smaller
    min_plays_required: int = 300

    # If True, requires multi-year data load
    requires_multi_year: bool = False

    def __str__(self):
        return f"[{self.id}]  {self.name}  ({self.category})\n    {self.description}"


# ── Lens-specific analysis functions ───────────────────────────────────────────

def _player_split_analysis(plays: pd.DataFrame, metrics: list, dims: list,
                            min_attempts: int = 30) -> list:
    """
    Like run_split_analysis but groups by player name instead of posteam.
    Runs three views: passer, rusher, receiver.
    Returns findings in the standard format (team field = player name).
    """
    findings = []

    player_views = [
        ("passer",   "passer_player_name",   plays[plays["play_type"] == "pass"]),
        ("rusher",   "rusher_player_name",    plays[plays["play_type"] == "run"]),
        ("receiver", "receiver_player_name",  plays[plays["play_type"] == "pass"]),
    ]

    for role, player_col, sub_df in player_views:
        if player_col not in sub_df.columns:
            continue
        sub_df = sub_df.copy()
        sub_df = sub_df[sub_df[player_col].notna()].copy()
        if len(sub_df) < min_attempts * 3:
            continue

        # Replace posteam with player name so run_split_analysis works unchanged
        sub_df["posteam"] = sub_df[player_col].str.split(",").str[0].str.strip()

        # Only include players with enough plays
        counts = sub_df["posteam"].value_counts()
        valid  = counts[counts >= min_attempts].index
        sub_df = sub_df[sub_df["posteam"].isin(valid)]
        if len(sub_df) < min_attempts * 2:
            continue

        # Focus on the most relevant metrics for this role
        if role == "passer":
            role_metrics = [m for m in metrics if m in {
                "epa", "qb_epa", "cpoe", "air_yards", "time_to_throw",
                "pass_oe", "comp_air_epa", "comp_yac_epa", "xpass",
            }]
        elif role == "rusher":
            role_metrics = [m for m in metrics if m in {
                "epa", "yards_gained", "rushing_yards",
                "xyac_epa", "xyac_mean_yardage",
            }]
        else:  # receiver
            role_metrics = [m for m in metrics if m in {
                "epa", "air_yards", "yards_after_catch",
                "comp_air_epa", "comp_yac_epa", "xyac_epa",
            }]

        if not role_metrics:
            role_metrics = metrics[:6]

        role_dims = [d for d in dims if d in {
            "down_str", "pass_length", "pass_location", "score_bucket",
            "shotgun_label", "pressure_label", "qtr_str",
        }]
        if not role_dims:
            role_dims = dims[:5]

        role_findings = run_split_analysis(sub_df, role_metrics, role_dims,
                                           min_plays=min_attempts)
        # Tag each finding with the role
        for f in role_findings:
            f["dimension"] = f"{role}:{f['dimension']}"
        findings.extend(role_findings)

    return findings


def _trend_analysis(plays: pd.DataFrame, metrics: list, dims: list,
                    min_team_weeks: int = 8) -> list:
    """
    Detect teams with statistically anomalous metric trends across weeks.
    For each (team, metric), fits a linear trend over per-week averages and
    compares the slope to the cross-team distribution of slopes.
    Returns findings in the standard format (dimension='week_trend').
    """
    try:
        from scipy import stats as sp_stats
    except ImportError:
        print("  [trend] scipy not available — skipping trend analysis")
        return []

    if "week" not in plays.columns:
        return []

    findings = []

    for metric in metrics:
        sub = plays[["posteam", "game_id", "week", metric]].dropna()
        if len(sub) < 200:
            continue

        # Per-team per-week averages
        team_weekly = (
            sub.groupby(["posteam", "week"])[metric]
               .mean()
               .reset_index()
               .rename(columns={metric: "value"})
        )

        slopes = {}
        for team, grp in team_weekly.groupby("posteam"):
            if grp["week"].nunique() < min_team_weeks:
                continue
            slope, _, _, _, _ = sp_stats.linregress(
                grp["week"].astype(float), grp["value"].astype(float)
            )
            slopes[team] = slope

        if len(slopes) < 8:
            continue

        slope_vals  = np.array(list(slopes.values()))
        slope_mean  = slope_vals.mean()
        slope_std   = slope_vals.std()
        if slope_std < 1e-9:
            continue

        for team, slope in slopes.items():
            z = (slope - slope_mean) / slope_std
            if abs(z) < Z_THRESHOLD:
                continue

            direction = "improving" if z > 0 else "declining"
            n_weeks   = len(team_weekly[team_weekly["posteam"] == team])
            findings.append({
                "team":            team,
                "metric":          metric,
                "dimension":       "week_trend",
                "dimension_value": direction,
                "team_value":      round(float(slope), 5),
                "league_avg":      round(float(slope_mean), 5),
                "delta":           round(float(slope - slope_mean), 5),
                "std_devs_away":   round(float(z), 3),
                "effect_size":     round(float(abs(z)), 3),
                "sample_size":     int(n_weeks),
                "direction":       direction,
                "finding_type":    "trend",
            })

    return findings


def _efficiency_gap_analysis(plays: pd.DataFrame, metrics: list, dims: list) -> list:
    """
    Find teams where EPA/play diverges from actual win rate.
    Uses the plays DataFrame's result column (home - away final score diff)
    to compute team win rates, then cross-compares against mean EPA/play.
    """
    findings = []

    if "result" not in plays.columns or "home_team" not in plays.columns:
        print("  [efficiency-gaps] required columns missing — skipping")
        return findings

    # Build game-level results
    games = (
        plays[["game_id", "home_team", "away_team", "result"]]
        .drop_duplicates("game_id")
        .dropna(subset=["result"])
    )
    if games.empty:
        return findings

    # EPA/play per team (all plays)
    epa_per_team = (
        plays.groupby("posteam")["epa"]
             .agg(epa_per_play="mean", n_plays="count")
             .reset_index()
    )

    # Win rate per team
    records = []
    for _, g in games.iterrows():
        result = float(g["result"])
        home   = g["home_team"]
        away   = g["away_team"]
        records.append({"team": home, "win": int(result > 0), "loss": int(result < 0)})
        records.append({"team": away, "win": int(result < 0), "loss": int(result > 0)})

    rec_df = pd.DataFrame(records)
    win_rates = (
        rec_df.groupby("team")
              .agg(wins=("win", "sum"), games=("win", "count"))
              .assign(win_rate=lambda x: x["wins"] / x["games"])
              .reset_index()
    )

    merged = epa_per_team.merge(win_rates, left_on="posteam", right_on="team")
    if len(merged) < 8:
        return findings

    # League baselines
    lg_epa    = merged["epa_per_play"].mean()
    lg_epa_sd = merged["epa_per_play"].std()
    lg_win    = merged["win_rate"].mean()
    lg_win_sd = merged["win_rate"].std()

    for _, row in merged.iterrows():
        z_epa = (row["epa_per_play"] - lg_epa) / (lg_epa_sd + 1e-9)
        z_win = (row["win_rate"] - lg_win) / (lg_win_sd + 1e-9)

        # Flag teams where EPA and win rate z-scores diverge significantly
        gap = z_epa - z_win
        if abs(gap) < 1.5:
            continue

        direction = "above" if gap > 0 else "below"  # EPA above/below win rate
        label     = "high-EPA-underachiever" if gap > 0 else "low-EPA-overachiever"
        findings.append({
            "team":            str(row["posteam"]),
            "metric":          "epa_vs_win_rate_gap",
            "dimension":       "efficiency_gap",
            "dimension_value": label,
            "team_value":      round(float(row["epa_per_play"]), 4),
            "league_avg":      round(float(lg_epa), 4),
            "delta":           round(float(row["epa_per_play"] - lg_epa), 4),
            "std_devs_away":   round(float(gap), 3),
            "effect_size":     round(float(abs(gap)), 3),
            "sample_size":     int(row["games"]),
            "direction":       direction,
            "finding_type":    "efficiency_gap",
        })

        # Also flag the win_rate dimension separately
        findings.append({
            "team":            str(row["posteam"]),
            "metric":          "win_rate",
            "dimension":       "efficiency_gap",
            "dimension_value": label,
            "team_value":      round(float(row["win_rate"]), 4),
            "league_avg":      round(float(lg_win), 4),
            "delta":           round(float(row["win_rate"] - lg_win), 4),
            "std_devs_away":   round(float(z_win), 3),
            "effect_size":     round(float(abs(z_win)), 3),
            "sample_size":     int(row["games"]),
            "direction":       "above" if z_win > 0 else "below",
            "finding_type":    "efficiency_gap",
        })

    return findings


def _add_sequence_context(plays: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'after_sack', 'after_turnover', 'after_scoring' boolean label columns.
    Requires the full sorted play sequence within each game.
    """
    df = plays.copy().sort_values(["game_id", "play_id"])

    for col, src in [("sack", "sack"), ("interception", "interception"),
                     ("fumble_lost", "fumble_lost"),
                     ("touchdown", "touchdown")]:
        if src in df.columns:
            df[f"prev_{col}"] = (
                df.groupby("game_id")[src]
                  .shift(1)
                  .fillna(0)
                  .astype(int)
            )
        else:
            df[f"prev_{col}"] = 0

    df["prev_turnover"] = ((df["prev_interception"] + df["prev_fumble_lost"]) > 0).astype(int)

    df["after_sack"]     = df["prev_sack"].apply(
        lambda x: "After Sack"     if x == 1 else "Normal Down")
    df["after_turnover"] = df["prev_turnover"].apply(
        lambda x: "After Turnover" if x == 1 else "Normal Down")
    df["after_scoring"]  = df["prev_touchdown"].apply(
        lambda x: "After TD"       if x == 1 else "Normal Down")

    return df


# ── Lens library ────────────────────────────────────────────────────────────────

def _prepare_noop(plays):
    return plays

def _prepare_two_minute(plays):
    return plays[plays["half_seconds_remaining"] <= 120].copy()

def _prepare_goal_line(plays):
    return plays[plays["yardline_100"] <= 5].copy()

def _prepare_third_long(plays):
    return plays[(plays["down"] == 3) & (plays["ydstogo"] >= 7)].copy()

def _prepare_game_flow(plays):
    # Keep all plays; game-flow lens focuses the dimension list on score state
    return plays.copy()

def _prepare_sequence(plays):
    return _add_sequence_context(plays)

def _prepare_personnel(plays):
    return plays.copy()


ALL_LENSES: Dict[str, Lens] = {

    "team-level": Lens(
        id="team-level",
        name="Team-Level Splits",
        category="baseline",
        description=(
            "Baseline team-vs-league analysis across all pass/run plays. "
            "The broadest lens — covers all metrics × all dimensions."
        ),
        prepare_fn=_prepare_noop,
        min_plays_required=5000,
    ),

    "matchup": Lens(
        id="matchup",
        name="Matchup Analysis",
        category="matchup",
        description=(
            "Team performance patterns vs specific opponents. Adds 'defteam' "
            "as a split dimension so we can see how scheme matchups affect metrics."
        ),
        prepare_fn=_prepare_noop,
        extra_dims=["defteam"],
        dimension_focus=["defteam", "score_bucket", "shotgun_label", "pass_location",
                         "pass_length", "defense_coverage_type", "down_str"],
        min_plays_required=5000,
    ),

    "two-minute-drill": Lens(
        id="two-minute-drill",
        name="2-Minute Drill",
        category="situational",
        description=(
            "Plays in the last 2 minutes of each half — the highest-leverage "
            "possession situations. Reveals hurry-up tendencies and clock management."
        ),
        prepare_fn=_prepare_two_minute,
        metric_focus=["epa", "qb_epa", "cpoe", "pass_oe", "time_to_throw",
                      "air_yards", "yards_after_catch", "comp_air_epa"],
        dimension_focus=["down_str", "score_bucket", "shotgun_label",
                         "no_huddle_label", "pass_location", "pass_length"],
        min_plays_required=300,
    ),

    "goal-line": Lens(
        id="goal-line",
        name="Goal Line",
        category="situational",
        description=(
            "Plays inside the opponent's 5-yard line. Tests red-zone execution, "
            "personnel packages, and QB decisiveness under compressed field geometry."
        ),
        prepare_fn=_prepare_goal_line,
        dimension_focus=["down_str", "score_bucket", "shotgun_label",
                         "personnel_group", "pass_location", "qb_scramble_label"],
        min_plays_required=150,
    ),

    "third-and-long": Lens(
        id="third-and-long",
        name="3rd and Long",
        category="situational",
        description=(
            "3rd down with 7+ yards to go — the prototypical 'obvious pass' situation. "
            "Reveals how teams cope with known passing downs and defensive adjustments."
        ),
        prepare_fn=_prepare_third_long,
        dimension_focus=["down_str", "score_bucket", "shotgun_label",
                         "pass_location", "pass_length", "pressure_label",
                         "defense_coverage_type", "defenders_in_box"],
        min_plays_required=300,
    ),

    "player-level": Lens(
        id="player-level",
        name="Player Performance",
        category="player",
        description=(
            "QB, rusher, and receiver splits — groups by individual player name "
            "instead of team. Surfaces outlier passers, runners, and targets."
        ),
        prepare_fn=_prepare_noop,
        analysis_fn=_player_split_analysis,
        min_plays_required=1000,
    ),

    "game-flow": Lens(
        id="game-flow",
        name="Game Flow",
        category="flow",
        description=(
            "How teams behave at various score differentials (blowout deficit, "
            "close game, comfortable lead). Reveals script-departure patterns and "
            "desperation passing vs. run-out-the-clock tendencies."
        ),
        prepare_fn=_prepare_game_flow,
        dimension_focus=["score_bucket", "shotgun_label", "no_huddle_label",
                         "pass_length", "down_str", "qtr_str"],
        metric_focus=["epa", "qb_epa", "pass_oe", "cpoe", "air_yards",
                      "time_to_throw", "comp_yac_epa"],
        min_plays_required=5000,
    ),

    "sequence": Lens(
        id="sequence",
        name="Sequence Analysis",
        category="sequence",
        description=(
            "What happens on the play immediately after a sack, turnover, or "
            "touchdown? Tests whether teams adjust or panic after high-impact events."
        ),
        prepare_fn=_prepare_sequence,
        extra_dims=["after_sack", "after_turnover", "after_scoring"],
        dimension_focus=["after_sack", "after_turnover", "after_scoring",
                         "down_str", "score_bucket", "shotgun_label"],
        metric_focus=["epa", "qb_epa", "pass_oe", "cpoe", "time_to_throw",
                      "air_yards", "yards_after_catch"],
        min_plays_required=1000,
    ),

    "weekly-trends": Lens(
        id="weekly-trends",
        name="Weekly Trends",
        category="trend",
        description=(
            "Which teams improved or declined most across the season? Uses "
            "per-week metric averages and linear regression to detect trajectories."
        ),
        prepare_fn=_prepare_noop,
        analysis_fn=_trend_analysis,
        min_plays_required=2000,
    ),

    "efficiency-gaps": Lens(
        id="efficiency-gaps",
        name="Efficiency vs Outcome Gaps",
        category="efficiency",
        description=(
            "Teams whose EPA/play diverges from their actual win rate. "
            "Identifies 'unlucky' high-EPA teams and 'lucky' low-EPA teams."
        ),
        prepare_fn=_prepare_noop,
        analysis_fn=_efficiency_gap_analysis,
        min_plays_required=2000,
    ),

    "personnel-formation": Lens(
        id="personnel-formation",
        name="Personnel & Formation",
        category="scheme",
        description=(
            "Formation tendencies, personnel grouping usage, and how opponents "
            "adjusted their box counts and coverage types in response."
        ),
        prepare_fn=_prepare_personnel,
        dimension_focus=["personnel_group", "shotgun_label", "no_huddle_label",
                         "offense_formation", "defenders_in_box",
                         "defense_coverage_type", "pass_rushers"],
        metric_focus=["epa", "qb_epa", "pass_oe", "cpoe", "air_yards",
                      "time_to_throw", "yards_after_catch", "comp_yac_epa"],
        min_plays_required=5000,
    ),

    "historical": Lens(
        id="historical",
        name="Historical Comparison",
        category="historical",
        description=(
            "Compares 2025 team metrics to 2023–2024 baselines. Requires "
            "loading multiple seasons — set --seasons flag to activate."
        ),
        prepare_fn=_prepare_noop,
        requires_multi_year=True,
        min_plays_required=10000,
    ),
}


# ── State management ────────────────────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "phase":        "single",   # "single" | "combo"
        "lens_history": {},         # lens_id → {last_used, use_count, findings_files}
        "combo_history": [],        # list of [lens_a_id, lens_b_id] already run
        "created":      str(date.today()),
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return _default_state()


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _days_since(date_str: Optional[str]) -> int:
    if not date_str:
        return 9999
    try:
        d = date.fromisoformat(date_str)
        return (date.today() - d).days
    except ValueError:
        return 9999


# ── Scheduling ──────────────────────────────────────────────────────────────────

def _lens_score(lens_id: str, history: dict) -> float:
    """
    Score = days_since_last_use / (1 + use_count).
    Never-used lenses are treated as 9999 days old and use_count=0.
    Higher score → lens is more overdue.
    """
    h = history.get(lens_id, {})
    days = _days_since(h.get("last_used"))
    count = h.get("use_count", 0)
    return days / (1 + count)


def select_lenses(state: dict, n: int = 2, only_single: bool = False) -> List:
    """
    Select the next 1–2 lenses (or combos) to run.
    Returns a list of:
      - In single phase: list of Lens objects
      - In combo phase:  list of (Lens, Lens) tuples
    """
    single_lenses = [
        lid for lid, lens in ALL_LENSES.items()
        if not lens.requires_multi_year
    ]

    if state["phase"] == "single" or only_single:
        scored = sorted(
            single_lenses,
            key=lambda lid: _lens_score(lid, state["lens_history"]),
            reverse=True,
        )
        return [ALL_LENSES[lid] for lid in scored[:n]]

    # Combo phase — find pairs not yet tried
    done_combos = {tuple(sorted(c)) for c in state["combo_history"]}
    candidates  = []
    for i, a in enumerate(single_lenses):
        for b in single_lenses[i + 1:]:
            pair = tuple(sorted([a, b]))
            if pair in done_combos:
                continue
            # Score the combo by the sum of individual scores
            score = _lens_score(a, state["lens_history"]) + \
                    _lens_score(b, state["lens_history"])
            candidates.append((score, a, b))

    if not candidates:
        print("All combos have been run. Resetting combo history.")
        state["combo_history"] = []
        return select_lenses(state, n=n)

    candidates.sort(reverse=True)
    selected = [(ALL_LENSES[a], ALL_LENSES[b]) for _, a, b in candidates[:n]]
    return selected


def _check_phase_transition(state: dict) -> None:
    """Switch to combo phase if every single lens has been used at least once."""
    if state["phase"] != "single":
        return
    single_ids = [lid for lid, l in ALL_LENSES.items() if not l.requires_multi_year]
    if all(lid in state["lens_history"] for lid in single_ids):
        state["phase"] = "combo"
        print("\n  All single lenses have been cycled through.")
        print("  Switching to COMBO phase — lenses will now be intersected.\n")


# ── Core run logic ──────────────────────────────────────────────────────────────

def _build_dims(plays: pd.DataFrame, lens: Lens) -> list:
    """Build the dimension list for a lens."""
    if lens.dimension_focus is not None:
        # Use exactly the specified dims (filter to those actually present)
        extra = lens.extra_dims or []
        all_focus = lens.dimension_focus + extra
        present = [d for d in all_focus if d in plays.columns]
        return sorted(set(present))

    # Auto-detect + derived, adding any extra dims
    auto = _auto_detect_dims(plays, _DIM_BLACKLIST)
    extra = [d for d in (lens.extra_dims or []) if d in plays.columns]
    return sorted(set(auto + DIMS_DERIVED + extra))


def _build_metrics(plays: pd.DataFrame, lens: Lens) -> list:
    """Build the metric list for a lens."""
    if lens.metric_focus is not None:
        present = [m for m in lens.metric_focus if m in plays.columns
                   and plays[m].notna().sum() / len(plays) > 0.10]
        return present
    return _auto_detect_metrics(plays, _METRIC_BLACKLIST)


def _findings_to_csv(findings: list, lens_id: str, run_date: str) -> Path:
    """Rank findings and write to a lens-specific CSV. Returns the output path."""
    if not findings:
        print(f"  No findings produced for lens '{lens_id}'")
        return None

    df_all = _rank_and_interpret(findings)

    output_cols = [
        "team", "metric", "dimension", "dimension_value",
        "team_value", "league_avg", "std_devs_away",
        "sample_size", "interpretation",
    ]
    top_df = df_all.head(TOP_N).reset_index()[["rank"] + output_cols]

    out_path = FINDINGS_DIR / f"{lens_id}_{run_date}.csv"
    top_df.to_csv(out_path, index=False)
    print(f"  → {len(df_all)} findings | top {len(top_df)} saved to {out_path.name}")
    return out_path


def run_lens(lens: Lens, plays: pd.DataFrame, run_date: str) -> Optional[Path]:
    """
    Execute a single lens against the plays DataFrame.
    Returns the path to the findings CSV, or None if the lens produced no output.
    """
    print(f"\n{'─'*70}")
    print(f"  Lens: {lens.name}  [{lens.id}]")
    print(f"  {lens.description}")
    print(f"{'─'*70}")

    # Apply lens filter / enrichment
    filtered = lens.prepare_fn(plays) if lens.prepare_fn else plays.copy()
    n = len(filtered)

    if n < lens.min_plays_required:
        print(f"  Skipping: only {n:,} plays (minimum {lens.min_plays_required:,})")
        return None

    print(f"  Working dataset: {n:,} plays  |  "
          f"{filtered['game_id'].nunique()} games  |  "
          f"{filtered['posteam'].nunique()} teams")

    metrics = _build_metrics(filtered, lens)
    dims    = _build_dims(filtered, lens)
    print(f"  Metrics: {len(metrics)}  |  Dimensions: {len(dims)}")

    # Run the analysis
    if lens.analysis_fn is not None:
        findings = lens.analysis_fn(filtered, metrics, dims)
    else:
        print("  Running split analysis...")
        split = run_split_analysis(filtered, metrics, dims)
        print(f"    → {len(split)} split findings")
        print("  Running correlation analysis...")
        corr  = run_correlation_analysis(filtered, metrics)
        print(f"    → {len(corr)} correlation findings")
        findings = split + corr

    print(f"  Total findings: {len(findings)}")
    return _findings_to_csv(findings, lens.id, run_date)


def run_combo(lens_a: Lens, lens_b: Lens, plays: pd.DataFrame,
              run_date: str) -> Optional[Path]:
    """
    Run two lenses simultaneously: apply both filters (intersection),
    merge both dimension and metric focus lists, and run analysis.
    Returns the path to the findings CSV.
    """
    combo_id = f"{lens_a.id}+{lens_b.id}"
    print(f"\n{'═'*70}")
    print(f"  COMBO: {lens_a.name}  ×  {lens_b.name}")
    print(f"  [{combo_id}]")
    print(f"{'═'*70}")

    # Intersect filters
    tmp = lens_a.prepare_fn(plays) if lens_a.prepare_fn else plays.copy()
    filtered = lens_b.prepare_fn(tmp) if lens_b.prepare_fn else tmp

    n = len(filtered)
    min_req = max(lens_a.min_plays_required, lens_b.min_plays_required) // 4
    if n < min_req:
        print(f"  Skipping: only {n:,} plays after intersection (minimum {min_req:,})")
        return None

    print(f"  Intersection: {n:,} plays")

    # Merge dimension and metric focus
    mf_a = lens_a.metric_focus or []
    mf_b = lens_b.metric_focus or []
    combined_metric_focus = sorted(set(mf_a + mf_b)) or None

    df_a = lens_a.dimension_focus or []
    df_b = lens_b.dimension_focus or []
    extra = sorted(set((lens_a.extra_dims or []) + (lens_b.extra_dims or [])))
    combined_dim_focus = sorted(set(df_a + df_b + extra)) or None

    # Build a synthetic combo lens for dimension/metric resolution
    combo_lens = Lens(
        id=combo_id, name=f"{lens_a.name} × {lens_b.name}",
        category="combo", description="",
        metric_focus=combined_metric_focus,
        dimension_focus=combined_dim_focus,
        extra_dims=extra,
    )

    metrics = _build_metrics(filtered, combo_lens)
    dims    = _build_dims(filtered, combo_lens)
    print(f"  Metrics: {len(metrics)}  |  Dimensions: {len(dims)}")

    # Prefer a custom analysis fn if either lens has one (player/trend take priority)
    if lens_a.analysis_fn or lens_b.analysis_fn:
        afn = lens_a.analysis_fn or lens_b.analysis_fn
        findings = afn(filtered, metrics, dims)
    else:
        split    = run_split_analysis(filtered, metrics, dims)
        corr     = run_correlation_analysis(filtered, metrics)
        findings = split + corr

    print(f"  Total findings: {len(findings)}")
    return _findings_to_csv(findings, combo_id, run_date)


# ── Status display ──────────────────────────────────────────────────────────────

def show_status(state: dict) -> None:
    print(f"\n{'='*65}")
    print(f"EXPLORATION PLANNER STATUS")
    print(f"{'='*65}")
    print(f"Phase:    {state['phase'].upper()}")
    print(f"State file: {STATE_FILE}")
    print()

    print(f"{'Lens':<22}  {'Uses':>5}  {'Last Used':<12}  {'Score':>6}  Category")
    print("-" * 65)
    for lid, lens in ALL_LENSES.items():
        h     = state["lens_history"].get(lid, {})
        count = h.get("use_count", 0)
        last  = h.get("last_used", "never")
        score = _lens_score(lid, state["lens_history"])
        marker = " *" if lid not in state["lens_history"] else "  "
        print(f"{lid:<22}{marker}{count:>4}  {last:<12}  {score:>6.1f}  {lens.category}")

    print()
    if state["combo_history"]:
        print(f"Combos completed: {len(state['combo_history'])}")
        for a, b in state["combo_history"]:
            print(f"  {a} × {b}")
    else:
        print("No combos run yet.")

    findings_files = sorted(FINDINGS_DIR.glob("*.csv"))
    print(f"\nFindings files: {len(findings_files)} in {FINDINGS_DIR.name}/")
    for f in findings_files[-5:]:
        size = f.stat().st_size // 1024
        print(f"  {f.name}  ({size} KB)")
    if len(findings_files) > 5:
        print(f"  ... and {len(findings_files)-5} older files")


def list_lenses() -> None:
    print(f"\n{'='*65}")
    print("AVAILABLE LENSES")
    print(f"{'='*65}\n")
    for lid, lens in ALL_LENSES.items():
        multi = "  [requires multi-year]" if lens.requires_multi_year else ""
        print(f"  {lid}{multi}")
        print(f"    {lens.description}")
        print()


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NFL analysis lens scheduler and runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--status",    action="store_true",
                        help="Show lens usage history and exit")
    parser.add_argument("--list",      action="store_true",
                        help="List all available lenses and exit")
    parser.add_argument("--lens",      type=str, default=None,
                        help="Force a specific lens ID (comma-separated for multiple)")
    parser.add_argument("--combo",     type=str, default=None,
                        help="Force a combo of two lens IDs: e.g. game-flow,player-level")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show which lenses would run without executing them")
    parser.add_argument("--interpret", action="store_true",
                        help="After generating findings, run interpret.py on each")
    parser.add_argument("--reset",     action="store_true",
                        help="Clear lens usage history")
    parser.add_argument("--seasons",   type=str, default="2025",
                        help="Comma-separated seasons to load (default: 2025)")
    parser.add_argument("--n",         type=int, default=2,
                        help="Number of lenses (or combos) to run today (default: 2)")
    args = parser.parse_args()

    state = load_state()

    # ── Utility commands ────────────────────────────────────────────────────────
    if args.status:
        show_status(state)
        return

    if args.list:
        list_lenses()
        return

    if args.reset:
        print("Resetting exploration state...")
        state = _default_state()
        save_state(state)
        print("Done. All lens history cleared.")
        return

    # ── Determine which lenses / combos to run ──────────────────────────────────
    run_date = date.today().isoformat()

    if args.combo:
        # Force a specific combo
        ids = [s.strip() for s in args.combo.split(",")]
        if len(ids) != 2:
            print("ERROR: --combo requires exactly two lens IDs separated by comma.")
            sys.exit(1)
        for lid in ids:
            if lid not in ALL_LENSES:
                print(f"ERROR: Unknown lens '{lid}'. Use --list to see available lenses.")
                sys.exit(1)
        selected = [(ALL_LENSES[ids[0]], ALL_LENSES[ids[1]])]
        mode = "combo"

    elif args.lens:
        # Force specific single lens(es)
        ids = [s.strip() for s in args.lens.split(",")]
        for lid in ids:
            if lid not in ALL_LENSES:
                print(f"ERROR: Unknown lens '{lid}'. Use --list to see available lenses.")
                sys.exit(1)
        selected = [ALL_LENSES[lid] for lid in ids]
        mode = "single"

    else:
        # Scheduled selection
        _check_phase_transition(state)
        selected = select_lenses(state, n=args.n)
        mode = state["phase"]

    # ── Dry run ─────────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\nDRY RUN — would run {len(selected)} lens(es) in '{mode}' mode:\n")
        if mode == "combo":
            for la, lb in selected:
                print(f"  COMBO: {la.name}  ×  {lb.name}")
        else:
            for lens in selected:
                print(f"  LENS:  {lens.name}  [{lens.id}]")
        return

    # ── Load data ────────────────────────────────────────────────────────────────
    seasons = [int(s.strip()) for s in args.seasons.split(",")]
    plays   = load_plays(seasons)

    # ── Execute ──────────────────────────────────────────────────────────────────
    output_paths = []

    if mode == "combo":
        for la, lb in selected:
            out_path = run_combo(la, lb, plays, run_date)
            if out_path:
                output_paths.append(out_path)
                pair = sorted([la.id, lb.id])
                if pair not in state["combo_history"]:
                    state["combo_history"].append(pair)
                # Update individual lens histories
                for lens in (la, lb):
                    h = state["lens_history"].setdefault(lens.id, {
                        "use_count": 0, "last_used": None, "findings_files": []
                    })
                    h["use_count"]  += 1
                    h["last_used"]   = run_date
                    h["findings_files"].append(str(out_path))
    else:
        for lens in selected:
            if lens.requires_multi_year and len(seasons) < 2:
                print(f"\nSkipping '{lens.id}' — requires --seasons 2023,2024,2025")
                continue
            out_path = run_lens(lens, plays, run_date)
            if out_path:
                output_paths.append(out_path)
            h = state["lens_history"].setdefault(lens.id, {
                "use_count": 0, "last_used": None, "findings_files": []
            })
            h["use_count"]  += 1
            h["last_used"]   = run_date
            if out_path:
                h["findings_files"].append(str(out_path))

    # Phase transition check after this run
    _check_phase_transition(state)
    save_state(state)

    # ── Summary ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Run complete. {len(output_paths)} findings file(s) written:")
    for p in output_paths:
        print(f"  {p}")
    print(f"State saved to {STATE_FILE.name}")

    # ── Optional: pipe to interpret.py ──────────────────────────────────────────
    if args.interpret:
        for p in output_paths:
            print(f"\n{'─'*70}")
            print(f"Running interpret.py on {p.name}...")
            print(f"{'─'*70}")
            result = subprocess.run(
                [sys.executable, str(ROOT / "interpret.py"),
                 "--findings", str(p), "--n-insights", "8"],
                cwd=str(ROOT),
            )
            if result.returncode != 0:
                print(f"  interpret.py exited with code {result.returncode}")


if __name__ == "__main__":
    main()
