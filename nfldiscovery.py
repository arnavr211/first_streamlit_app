"""
nfldiscovery.py

For every NFL team, computes EPA/play splits across six dimensions:
  - Down (1–4)
  - Quarter (1–4, OT)
  - Formation (Shotgun / Under Center / Pistol)
  - Personnel grouping (11, 12, 13, 21, 22 personnel, etc.)
  - Score differential bucket (Blowout Deficit → Blowout Lead)
  - Play type (pass / run)

Flags any team-split where EPA/play is >1.5 std devs from the league
mean for that same split, with at least 20 plays. Ranks findings by
z-score magnitude (most surprising first) and saves to CSV.
"""

import re
import sys
import pandas as pd
import numpy as np
import nfl_data_py as nfl

# ── Configuration ─────────────────────────────────────────────────────────────

MIN_PLAYS = 20          # minimum n to report a finding
Z_THRESHOLD = 1.5       # std devs from league mean to flag
OUTPUT_CSV = "nfl_discovery_findings.csv"

# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading 2025 nflfastR play-by-play data...")
pbp = nfl.import_pbp_data([2025])
plays = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
print(f"  {len(plays):,} pass/run plays loaded across {plays['game_id'].nunique()} games.\n")

# ── Dimension derivation ───────────────────────────────────────────────────────

def score_bucket(diff):
    """Map score_differential to a human-readable bucket."""
    if pd.isna(diff):
        return None
    if diff < -14:
        return "Blowout Deficit (<-14)"
    elif diff < -7:
        return "Deficit (-14 to -8)"
    elif diff < 0:
        return "Close Deficit (-7 to -1)"
    elif diff == 0:
        return "Tied (0)"
    elif diff <= 7:
        return "Close Lead (+1 to +7)"
    elif diff <= 14:
        return "Lead (+8 to +14)"
    else:
        return "Blowout Lead (>+14)"

BUCKET_ORDER = [
    "Blowout Deficit (<-14)", "Deficit (-14 to -8)", "Close Deficit (-7 to -1)",
    "Tied (0)", "Close Lead (+1 to +7)", "Lead (+8 to +14)", "Blowout Lead (>+14)"
]

def extract_personnel(personnel_str):
    """
    Parse the verbose nflverse personnel string into a standard grouping.
    Returns e.g. '11 (1RB,1TE,3WR)' or None if unparseable.
    """
    if pd.isna(personnel_str):
        return None
    rb = int(m.group(1)) if (m := re.search(r"(\d+)\s*RB", personnel_str)) else 0
    te = int(m.group(1)) if (m := re.search(r"(\d+)\s*TE", personnel_str)) else 0
    wr = int(m.group(1)) if (m := re.search(r"(\d+)\s*WR", personnel_str)) else 0
    fb = int(m.group(1)) if (m := re.search(r"(\d+)\s*FB", personnel_str)) else 0
    total_rb = rb + fb   # treat FB as extra RB for grouping purposes
    code = f"{total_rb}{te}"
    label = f"{code} ({total_rb}RB,{te}TE,{wr}WR)"
    return label

# Apply derivations
plays["score_bucket"] = plays["score_differential"].apply(score_bucket)
plays["personnel_group"] = plays["offense_personnel"].apply(extract_personnel)
plays["formation"] = plays["offense_formation"].fillna("Unknown")
plays["down_label"] = plays["down"].apply(
    lambda d: f"Down {int(d)}" if not pd.isna(d) else None
)
plays["qtr_label"] = plays["qtr"].apply(
    lambda q: f"Q{int(q)}" if q <= 4 else "OT"
)

# ── Core computation ───────────────────────────────────────────────────────────

def compute_findings(plays, split_col, split_label, min_plays=MIN_PLAYS, z_thresh=Z_THRESHOLD):
    """
    For a given split column, compute league-wide EPA stats per split value,
    then compare each team's EPA/play to the league baseline.
    Returns a DataFrame of flagged findings (|z| >= z_thresh, n >= min_plays).
    """
    # Drop rows where the split column is null
    df = plays[plays[split_col].notna()].copy()

    # League-wide: mean and std of per-team-game EPA in each split bucket
    # Use per-play raw EPA (not game-aggregated) for simplicity and sample power
    league_stats = (
        df.groupby(split_col)["epa"]
        .agg(league_mean="mean", league_std="std", league_n="count")
        .reset_index()
    )

    # Team-level stats per split bucket
    team_stats = (
        df.groupby(["posteam", split_col])["epa"]
        .agg(team_epa="mean", team_n="count")
        .reset_index()
    )

    # Merge
    merged = team_stats.merge(league_stats, on=split_col)

    # Filter: minimum sample
    merged = merged[merged["team_n"] >= min_plays]

    # Z-score: how many league std devs is this team from the league mean?
    # We use the std of *individual play EPA* as the denominator,
    # then scale by sqrt(n) to get the sampling distribution std.
    merged["sem"] = merged["league_std"] / np.sqrt(merged["team_n"])
    merged["z_score"] = (merged["team_epa"] - merged["league_mean"]) / merged["sem"]
    merged["epa_vs_league"] = merged["team_epa"] - merged["league_mean"]

    # Flag outliers
    flagged = merged[merged["z_score"].abs() >= z_thresh].copy()
    flagged["split_dimension"] = split_label
    flagged["split_value"] = flagged[split_col].astype(str)
    flagged["direction"] = flagged["z_score"].apply(
        lambda z: "ABOVE avg" if z > 0 else "BELOW avg"
    )

    return flagged[[
        "split_dimension", "split_value", "posteam",
        "team_epa", "league_mean", "epa_vs_league",
        "team_n", "league_n", "z_score", "direction"
    ]]

# ── Run all six dimensions ─────────────────────────────────────────────────────

print("Computing splits...")

dimensions = [
    ("down_label",       "Down"),
    ("qtr_label",        "Quarter"),
    ("formation",        "Formation"),
    ("personnel_group",  "Personnel"),
    ("score_bucket",     "Score Differential"),
    ("play_type",        "Play Type"),
]

all_findings = []
for col, label in dimensions:
    findings = compute_findings(plays, col, label)
    all_findings.append(findings)
    print(f"  {label:20s} → {len(findings):3d} flagged findings")

findings_df = pd.concat(all_findings, ignore_index=True)

# ── Rank by surprisingness ─────────────────────────────────────────────────────

# Surprisingness = |z_score|. Higher z = more extreme vs. league.
findings_df = findings_df.sort_values("z_score", key=abs, ascending=False).reset_index(drop=True)
findings_df.index += 1  # 1-based rank
findings_df.index.name = "rank"

# ── Console output ─────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"TOP 30 MOST SURPRISING FINDINGS  (|z| ≥ {Z_THRESHOLD}, n ≥ {MIN_PLAYS})")
print(f"{'='*80}")
print(f"Total flagged findings: {len(findings_df)}\n")

fmt = "{:>4}  {:<9}  {:<22}  {:<30}  {:>6}  {:>8}  {:>8}  {:>7}  {:>6}"
header = fmt.format(
    "Rank", "Team", "Dimension", "Split Value",
    "n", "EPA/play", "Lg Avg", "Diff", "Z-score"
)
print(header)
print("-" * len(header))

for rank, row in findings_df.head(30).iterrows():
    direction_marker = "▲" if row["direction"] == "ABOVE avg" else "▼"
    print(fmt.format(
        f"#{rank}",
        row["posteam"],
        row["split_dimension"],
        row["split_value"][:30],
        int(row["team_n"]),
        f"{row['team_epa']:.3f}",
        f"{row['league_mean']:.3f}",
        f"{direction_marker}{abs(row['epa_vs_league']):.3f}",
        f"{row['z_score']:.2f}",
    ))

# ── Narrative summary of top 10 ────────────────────────────────────────────────

print(f"\n{'='*80}")
print("NARRATIVE: TOP 10 FINDINGS")
print(f"{'='*80}\n")

for rank, row in findings_df.head(10).iterrows():
    direction = "significantly better" if row["direction"] == "ABOVE avg" else "significantly worse"
    print(
        f"#{rank}  [{row['split_dimension']}]  {row['posteam']}  |  {row['split_value']}\n"
        f"    {row['posteam']} is {direction} than league average in this split.\n"
        f"    EPA/play: {row['team_epa']:.3f}  vs  league avg: {row['league_mean']:.3f}  "
        f"({'+' if row['epa_vs_league'] > 0 else ''}{row['epa_vs_league']:.3f})\n"
        f"    Sample: {int(row['team_n'])} plays  |  Z-score: {row['z_score']:.2f}\n"
    )

# ── Dimension summary ──────────────────────────────────────────────────────────

print(f"{'='*80}")
print("FINDINGS SUMMARY BY DIMENSION")
print(f"{'='*80}")
dim_summary = (
    findings_df.reset_index()
    .groupby("split_dimension")
    .agg(
        total_flags=("rank", "count"),
        avg_abs_z=("z_score", lambda x: x.abs().mean()),
        max_abs_z=("z_score", lambda x: x.abs().max()),
        above_avg=("direction", lambda x: (x == "ABOVE avg").sum()),
        below_avg=("direction", lambda x: (x == "BELOW avg").sum()),
    )
    .sort_values("max_abs_z", ascending=False)
)
print(dim_summary.to_string())

# ── Save to CSV ────────────────────────────────────────────────────────────────

output_df = findings_df.reset_index().rename(columns={"index": "rank"})
output_df["team_epa"] = output_df["team_epa"].round(4)
output_df["league_mean"] = output_df["league_mean"].round(4)
output_df["epa_vs_league"] = output_df["epa_vs_league"].round(4)
output_df["z_score"] = output_df["z_score"].round(3)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nAll {len(output_df)} findings saved to: {OUTPUT_CSV}")
