"""
nfldiscovery.py  (refactored — automated metric × dimension discovery)

Automatically detects:
  METRICS   — numeric columns worth measuring (EPA, WPA, CPOE, air_yards, etc.)
  DIMENSIONS — categorical / low-cardinality columns worth splitting by
               (play_type, down, formation, coverage, pressure, etc.)

For every (metric × dimension × team) combination:
  - Computes the team's mean metric value within each dimension bucket
  - Compares against the league mean for that same bucket
  - Flags deviations ≥ 1.5 std devs with n ≥ 30

Also checks metric-to-metric correlations at the team level (game averages),
flagging teams whose metric relationships diverge from the league norm.

All findings are ranked by composite score = effect_size × log1p(n / MIN_PLAYS)
and the top 50 are saved to CSV.
"""

import re
import sys
import warnings
import pandas as pd
import numpy as np
import nfl_data_py as nfl

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────

MIN_PLAYS       = 30      # minimum plays per team-split to report
Z_THRESHOLD     = 1.5     # std devs from league mean to flag
TOP_N           = 50      # findings written to CSV
OUTPUT_CSV      = "nfl_discovery_findings.csv"

# Minimum population rate in pass/run plays for a column to be considered
MIN_METRIC_POP  = 0.20
MIN_DIM_POP     = 0.20

# For correlations: minimum games per team, minimum league |r| to bother testing
MIN_CORR_GAMES  = 8
MIN_LEAGUE_ABS_R = 0.10

# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading 2025 nflfastR play-by-play data...")
pbp = nfl.import_pbp_data([2025])
plays = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
print(f"  {len(plays):,} pass/run plays  |  {plays['game_id'].nunique()} games  |  "
      f"{plays['posteam'].nunique()} teams\n")

# ── Derived columns ────────────────────────────────────────────────────────────

def _score_bucket(diff):
    if pd.isna(diff): return None
    if diff < -14:    return "Blowout Deficit (<-14)"
    if diff < -7:     return "Deficit (-14 to -8)"
    if diff < 0:      return "Close Deficit (-7 to -1)"
    if diff == 0:     return "Tied (0)"
    if diff <= 7:     return "Close Lead (+1 to +7)"
    if diff <= 14:    return "Lead (+8 to +14)"
    return "Blowout Lead (>+14)"

def _personnel_group(s):
    if pd.isna(s): return None
    rb = int(m.group(1)) if (m := re.search(r"(\d+)\s*RB", s)) else 0
    fb = int(m.group(1)) if (m := re.search(r"(\d+)\s*FB", s)) else 0
    te = int(m.group(1)) if (m := re.search(r"(\d+)\s*TE", s)) else 0
    wr = int(m.group(1)) if (m := re.search(r"(\d+)\s*WR", s)) else 0
    total_rb = rb + fb
    return f"{total_rb}{te} ({total_rb}RB,{te}TE,{wr}WR)"

def notna(x):
    try:
        return not pd.isna(x)
    except (TypeError, ValueError):
        return True

plays["score_bucket"]     = plays["score_differential"].apply(_score_bucket)
plays["personnel_group"]  = plays["offense_personnel"].apply(_personnel_group)
plays["down_str"]         = plays["down"].apply(lambda d: f"Down {int(d)}" if notna(d) else None)
plays["qtr_str"]          = plays["qtr"].apply(lambda q: f"Q{int(q)}" if q <= 4 else "OT")
plays["defenders_in_box"] = plays["defenders_in_box"].apply(
    lambda x: f"{int(x)} in box" if notna(x) else None)
plays["pass_rushers"]     = plays["number_of_pass_rushers"].apply(
    lambda x: f"{int(x)} rushers" if notna(x) else None)
plays["shotgun_label"]    = plays["shotgun"].apply(
    lambda x: "Shotgun" if x == 1 else "Under Center" if x == 0 else None)
plays["no_huddle_label"]  = plays["no_huddle"].apply(
    lambda x: "No Huddle" if x == 1 else "Huddle" if x == 0 else None)
plays["qb_scramble_label"]= plays["qb_scramble"].apply(
    lambda x: "Scramble" if x == 1 else "Designed" if x == 0 else None)
plays["pressure_label"]   = plays["was_pressure"].apply(
    lambda x: "Pressure" if str(x) == "True" else "Clean" if str(x) == "False" else None)

# ── Auto-detect METRICS ────────────────────────────────────────────────────────
# Numeric columns that represent per-play outcomes or QB quality signals.
# Excludes: running game totals, scoreboard state, IDs, event flags, clock cols.

_METRIC_BLACKLIST = {
    # IDs / admin
    "play_id", "old_game_id_x", "old_game_id_y", "nfl_api_id", "nflverse_game_id",
    "series", "order_sequence", "fixed_drive", "drive",
    "drive_play_id_started", "drive_play_id_ended",
    # clock / field position (situational context, not outcomes)
    "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
    "yardline_100", "ydstogo", "ydsnet", "quarter_end", "ep",
    # scoreboard running totals
    "posteam_score", "defteam_score", "posteam_score_post", "defteam_score_post",
    "score_differential", "score_differential_post",
    "total_home_score", "total_away_score", "home_score", "away_score",
    "result", "total", "spread_line", "total_line",
    # cumulative game EPA/WPA (not per-play)
    "total_home_epa", "total_away_epa",
    "total_home_rush_epa", "total_away_rush_epa",
    "total_home_pass_epa", "total_away_pass_epa",
    "total_home_comp_air_epa", "total_away_comp_air_epa",
    "total_home_comp_yac_epa", "total_away_comp_yac_epa",
    "total_home_raw_air_epa", "total_away_raw_air_epa",
    "total_home_raw_yac_epa", "total_away_raw_yac_epa",
    "total_home_rush_wpa", "total_away_rush_wpa",
    "total_home_pass_wpa", "total_away_pass_wpa",
    "total_home_comp_air_wpa", "total_away_comp_air_wpa",
    "total_home_comp_yac_wpa", "total_away_comp_yac_wpa",
    "total_home_raw_air_wpa", "total_away_raw_air_wpa",
    "total_home_raw_yac_wpa", "total_away_raw_yac_wpa",
    # redundant WP columns
    "home_wp", "away_wp", "def_wp", "home_wp_post", "away_wp_post",
    "vegas_home_wp", "vegas_home_wpa",
    # jersey numbers / timeouts (not outcomes)
    "passer_jersey_number", "rusher_jersey_number", "receiver_jersey_number",
    "jersey_number", "home_timeouts_remaining", "away_timeouts_remaining",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    # game environment
    "temp", "wind", "week", "season",
    # binary event flags — meaningful as dimensions, not continuous metrics
    "shotgun", "no_huddle", "qb_dropback", "qb_kneel", "qb_spike", "qb_scramble",
    "rush_attempt", "pass_attempt", "sack", "touchdown", "pass_touchdown",
    "rush_touchdown", "return_touchdown", "extra_point_attempt", "two_point_attempt",
    "field_goal_attempt", "kickoff_attempt", "punt_attempt", "fumble", "fumble_lost",
    "fumble_forced", "fumble_not_forced", "fumble_out_of_bounds",
    "complete_pass", "incomplete_pass", "interception", "touchback",
    "first_down_rush", "first_down_pass", "first_down_penalty",
    "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed",
    "punt_blocked", "punt_inside_twenty", "punt_in_endzone",
    "punt_out_of_bounds", "punt_downed", "punt_fair_catch",
    "kickoff_inside_twenty", "kickoff_in_endzone", "kickoff_out_of_bounds",
    "kickoff_downed", "kickoff_fair_catch",
    "solo_tackle", "safety", "penalty", "tackled_for_loss",
    "own_kickoff_recovery", "own_kickoff_recovery_td", "qb_hit",
    "assist_tackle", "lateral_reception", "lateral_rush", "lateral_return",
    "lateral_recovery", "tackle_with_assist",
    "defensive_two_point_attempt", "defensive_two_point_conv",
    "defensive_extra_point_attempt", "defensive_extra_point_conv",
    "aborted_play", "play_deleted", "replay_or_challenge",
    "sp", "out_of_bounds", "home_opening_kickoff", "div_game",
    "pass", "rush", "play", "special", "first_down", "goal_to_go",
    "series_success", "return_yards", "penalty_yards", "kick_distance",
    # drive-level aggregates (redundant with play-level)
    "drive_play_count", "drive_first_downs", "drive_inside20",
    "drive_ended_with_score", "drive_quarter_start", "drive_quarter_end",
    "drive_yards_penalized",
    # probability curves (highly correlated with ep/epa)
    "no_score_prob", "opp_fg_prob", "opp_safety_prob", "opp_td_prob",
    "fg_prob", "safety_prob", "td_prob", "extra_point_prob",
    "two_point_conversion_prob",
    # game-state / expectation metrics (reflect team quality, not per-play performance)
    # Flagging these just reproduces "bad teams have low win probability" trivially
    "wp", "vegas_wp", "xpass",
    # situational integers (used as dimensions, not outcomes)
    "down", "qtr",
    # misc
    "number_of_pass_rushers", "defenders_in_box",  # used as derived string dims
}

def _auto_detect_metrics(df, blacklist, min_pop=MIN_METRIC_POP):
    numeric = df.select_dtypes(include=[np.number]).columns
    metrics = []
    for col in numeric:
        if col in blacklist:
            continue
        s = df[col].dropna()
        pop = len(s) / len(df)
        if pop < min_pop:
            continue
        if s.std() < 0.05:          # near-constant
            continue
        if s.nunique() <= 2:        # binary — use as dimension
            continue
        metrics.append(col)
    return sorted(metrics)

METRICS = _auto_detect_metrics(plays, _METRIC_BLACKLIST)
print(f"Auto-detected {len(METRICS)} metrics:")
print("  " + ", ".join(METRICS) + "\n")


# ── Auto-detect DIMENSIONS ─────────────────────────────────────────────────────
# Object columns with 2–20 unique values, plus derived categorical columns,
# plus select low-cardinality numeric columns treated as categories.

_DIM_BLACKLIST = {
    "game_id", "old_game_id_x", "old_game_id_y", "nflverse_game_id",
    "nfl_api_id", "stadium_id", "home_team", "away_team", "posteam",
    "defteam", "possession_team", "posteam_type",
    # player names / IDs
    "passer_player_name", "passer_player_id", "passer", "passer_id",
    "rusher_player_name", "rusher_player_id", "rusher", "rusher_id",
    "receiver_player_name", "receiver_player_id", "receiver", "receiver_id",
    "name", "id", "fantasy", "fantasy_player_name", "fantasy_player_id",
    "fantasy_id", "td_player_name", "td_player_id",
    "kicker_player_name", "kicker_player_id",
    "interception_player_name", "interception_player_id",
    "sack_player_name", "sack_player_id",
    "tackle_for_loss_1_player_name", "qb_hit_1_player_name",
    "forced_fumble_player_1_player_name",
    "solo_tackle_1_player_name", "assist_tackle_1_player_name",
    "punter_player_name", "punt_returner_player_name",
    "kickoff_returner_player_name",
    "pass_defense_1_player_name", "fumbled_1_player_name",
    "fumble_recovery_1_player_name",
    "tackle_with_assist_1_player_name", "tackle_with_assist_2_player_name",
    "safety_player_name",
    # game-level ids
    "home_coach", "away_coach", "game_stadium",
    # high-cardinality / text
    "desc", "play_type_nfl", "weather", "time", "yrdln",
    "time_of_day", "start_time", "end_clock_time", "play_clock",
    "drive_real_start_time", "drive_game_clock_start", "drive_game_clock_end",
    "drive_start_yard_line", "drive_end_yard_line",
    "drive_time_of_possession",
    "game_half",       # redundant with qtr_str
    "penalty_type",    # too many distinct values
    "series_result",   # drive outcome, not play context
    "fixed_drive_result", "drive_start_transition", "drive_end_transition",
    "offense_personnel",  # raw; use personnel_group instead
    "offense_players", "defense_players", "offense_names", "defense_names",
    "offense_positions", "defense_positions", "offense_numbers",
    "defense_numbers", "players_on_play",
    "replay_or_challenge_result",
    "td_team", "return_team", "penalty_team",
    # team-level IDs on specific events
    "timeout_team", "fumbled_1_team", "fumbled_2_team",
    "fumble_recovery_1_team", "fumble_recovery_2_team",
    "forced_fumble_player_1_team", "forced_fumble_player_2_team",
    "solo_tackle_1_team", "solo_tackle_2_team",
    "assist_tackle_1_team", "assist_tackle_2_team",
    "assist_tackle_3_team", "assist_tackle_4_team",
    "tackle_with_assist_1_team", "tackle_with_assist_2_team",
}

DIMS_DERIVED = [
    "score_bucket", "personnel_group", "down_str", "qtr_str",
    "shotgun_label", "no_huddle_label", "pressure_label",
    "defenders_in_box", "pass_rushers",
]

def _auto_detect_dims(df, blacklist, min_pop=MIN_DIM_POP, max_unique=20):
    obj_cols = df.select_dtypes(include=["object"]).columns
    dims = []
    for col in obj_cols:
        if col in blacklist:
            continue
        s = df[col].dropna()
        pop = len(s) / len(df)
        nu  = s.nunique()
        if pop < min_pop or nu < 2 or nu > max_unique:
            continue
        dims.append(col)
    return sorted(dims)

DIMS_AUTO = _auto_detect_dims(plays, _DIM_BLACKLIST)
DIMS = sorted(set(DIMS_AUTO + DIMS_DERIVED))
print(f"Auto-detected {len(DIMS)} dimensions:")
print("  " + ", ".join(DIMS) + "\n")


# ── Split analysis ─────────────────────────────────────────────────────────────

def run_split_analysis(df, metrics, dims, min_plays=MIN_PLAYS, z_thresh=Z_THRESHOLD):
    """
    For each (metric × dimension), compute league-wide stats per bucket,
    compare each team, flag |z| >= z_thresh with n >= min_plays.
    Returns a list of dicts (one per flagged finding).
    """
    findings = []
    combos = [(m, d) for m in metrics for d in dims]
    total = len(combos)

    for i, (metric, dim) in enumerate(combos):
        if i % 50 == 0:
            print(f"  Split analysis: {i}/{total} combinations...", end="\r")

        sub = df[[metric, dim, "posteam"]].dropna()
        if len(sub) < min_plays * 2:
            continue

        # League-wide stats per dimension bucket
        league = (
            sub.groupby(dim)[metric]
               .agg(league_mean="mean", league_std="std", league_n="count")
               .reset_index()
        )
        # Guard against zero std (all same value in bucket)
        league = league[league["league_std"] > 1e-6]

        # Team stats per bucket
        team = (
            sub.groupby(["posteam", dim])[metric]
               .agg(team_value="mean", n="count")
               .reset_index()
        )
        merged = team.merge(league, on=dim)
        merged = merged[merged["n"] >= min_plays]
        if merged.empty:
            continue

        # Standard error of the sampling distribution
        merged["sem"]          = merged["league_std"] / np.sqrt(merged["n"])
        merged["z_score"]      = (merged["team_value"] - merged["league_mean"]) / merged["sem"]
        merged["effect_size"]  = (merged["team_value"] - merged["league_mean"]) / merged["league_std"]

        flagged = merged[merged["z_score"].abs() >= z_thresh]
        for _, row in flagged.iterrows():
            direction = "above" if row["z_score"] > 0 else "below"
            findings.append({
                "team":            row["posteam"],
                "metric":          metric,
                "dimension":       dim,
                "dimension_value": str(row[dim]),
                "team_value":      round(float(row["team_value"]), 4),
                "league_avg":      round(float(row["league_mean"]), 4),
                "delta":           round(float(row["team_value"] - row["league_mean"]), 4),
                "std_devs_away":   round(float(row["z_score"]), 3),
                "effect_size":     round(float(row["effect_size"]), 3),
                "sample_size":     int(row["n"]),
                "direction":       direction,
                "finding_type":    "split",
            })

    print(f"  Split analysis: {total}/{total} combinations done.   ")
    return findings


# ── Correlation analysis ────────────────────────────────────────────────────────

def run_correlation_analysis(df, metrics, min_games=MIN_CORR_GAMES,
                              min_abs_r=MIN_LEAGUE_ABS_R, z_thresh=Z_THRESHOLD):
    """
    For each pair of metrics, compute per-game averages per team.
    Compare each team's pairwise correlation to the league-wide correlation
    using Fisher's z transform. Flag teams that differ significantly.
    """
    # Game-level metric averages (team × game)
    game_avgs = (
        df.groupby(["posteam", "game_id"])[metrics]
          .mean()
          .reset_index()
    )

    # Only keep metrics with decent game-level coverage
    metric_coverage = {
        m: (game_avgs[m].notna().sum() / len(game_avgs))
        for m in metrics
    }
    good_metrics = [m for m, cov in metric_coverage.items() if cov > 0.4]

    pairs = [(good_metrics[i], good_metrics[j])
             for i in range(len(good_metrics))
             for j in range(i + 1, len(good_metrics))]

    findings = []
    total = len(pairs)
    print(f"  Correlation analysis: {len(good_metrics)} metrics → {total} pairs...")

    for m1, m2 in pairs:
        sub = game_avgs[["posteam", m1, m2]].dropna()
        if len(sub) < 20:
            continue

        # League-wide correlation
        r_league = sub[m1].corr(sub[m2])
        if abs(r_league) < min_abs_r or pd.isna(r_league):
            continue  # no meaningful league signal to deviate from

        z_league = np.arctanh(np.clip(r_league, -0.9999, 0.9999))

        for team, grp in sub.groupby("posteam"):
            if len(grp) < min_games:
                continue
            r_team = grp[m1].corr(grp[m2])
            if pd.isna(r_team):
                continue
            z_team = np.arctanh(np.clip(r_team, -0.9999, 0.9999))
            se = 1.0 / np.sqrt(len(grp) - 3)
            z_score = (z_team - z_league) / se

            if abs(z_score) < z_thresh:
                continue

            direction = "stronger" if z_score > 0 else "weaker"
            findings.append({
                "team":            team,
                "metric":          f"corr({m1}, {m2})",
                "dimension":       "team_correlation",
                "dimension_value": f"league_r={r_league:.3f}",
                "team_value":      round(float(r_team), 4),
                "league_avg":      round(float(r_league), 4),
                "delta":           round(float(r_team - r_league), 4),
                "std_devs_away":   round(float(z_score), 3),
                "effect_size":     round(float(abs(r_team - r_league)), 3),
                "sample_size":     int(len(grp)),
                "direction":       direction,
                "finding_type":    "correlation",
            })

    return findings


# ── Ranking ─────────────────────────────────────────────────────────────────────

def composite_score(row):
    """
    Reward large effects AND large samples.
    effect_size is Cohen's d-like (league-std-normalised delta, sample-independent).
    log1p(n / MIN_PLAYS) rewards having more data beyond the minimum.
    """
    return abs(row["effect_size"]) * np.log1p(row["sample_size"] / MIN_PLAYS)


# ── Run everything ─────────────────────────────────────────────────────────────

print("Running split analysis...")
split_findings = run_split_analysis(plays, METRICS, DIMS)
print(f"  → {len(split_findings)} flagged split findings\n")

print("Running correlation analysis...")
corr_findings = run_correlation_analysis(plays, METRICS)
print(f"  → {len(corr_findings)} flagged correlation findings\n")

all_findings = split_findings + corr_findings
df_all = pd.DataFrame(all_findings)
df_all["composite_score"] = df_all.apply(composite_score, axis=1)
df_all = df_all.sort_values("composite_score", ascending=False).reset_index(drop=True)
df_all.index += 1
df_all.index.name = "rank"


# ── Interpretation strings ─────────────────────────────────────────────────────

def make_interpretation(row):
    if row["finding_type"] == "split":
        sign = "+" if row["delta"] > 0 else ""
        return (
            f"{row['team']} [{row['dimension']}={row['dimension_value']}]: "
            f"{row['metric']}={row['team_value']:.3f} vs league {row['league_avg']:.3f} "
            f"({sign}{row['delta']:.3f}), n={row['sample_size']}, "
            f"z={row['std_devs_away']:.2f} — "
            f"{abs(row['std_devs_away']):.1f} std devs {row['direction']} average"
        )
    else:
        return (
            f"{row['team']} correlation {row['metric']}: "
            f"team r={row['team_value']:.3f} vs league r={row['league_avg']:.3f} "
            f"(Δ={row['delta']:+.3f}), games={row['sample_size']}, "
            f"Fisher z={row['std_devs_away']:.2f} — "
            f"relationship is {row['direction']} than league norm"
        )

df_all["interpretation"] = df_all.apply(make_interpretation, axis=1)


# ── Console output ─────────────────────────────────────────────────────────────

type_counts = df_all["finding_type"].value_counts()
print(f"{'='*90}")
print(f"TOTAL FINDINGS: {len(df_all)}  "
      f"(split: {type_counts.get('split',0)}, "
      f"correlation: {type_counts.get('correlation',0)})")
print(f"{'='*90}\n")

print(f"TOP {min(30, len(df_all))} FINDINGS (ranked by effect size × sample weight)\n")

hdr = f"{'#':>4}  {'Team':<5}  {'Type':<6}  {'Metric':<28}  {'Dim':<20}  "
hdr += f"{'Split Value':<26}  {'n':>5}  {'Team':>7}  {'Lg Avg':>7}  {'Δ':>7}  {'z':>6}"
print(hdr)
print("-" * len(hdr))

for rank, row in df_all.head(30).iterrows():
    dim_short = row["dimension"][:20]
    val_short = str(row["dimension_value"])[:26]
    metric_short = row["metric"][:28]
    delta_str = f"{'+' if row['delta'] > 0 else ''}{row['delta']:.3f}"
    print(
        f"{rank:>4}  {row['team']:<5}  {row['finding_type'][:6]:<6}  {metric_short:<28}  "
        f"{dim_short:<20}  {val_short:<26}  {row['sample_size']:>5}  "
        f"{row['team_value']:>7.3f}  {row['league_avg']:>7.3f}  {delta_str:>7}  "
        f"{row['std_devs_away']:>6.2f}"
    )

# Per-dimension breakdown
print(f"\n{'='*60}")
print("FINDINGS COUNT BY DIMENSION")
print(f"{'='*60}")
dim_summary = (
    df_all.reset_index()
          .groupby("dimension")
          .agg(
              findings=("rank", "count"),
              avg_effect=("effect_size", lambda x: x.abs().mean()),
              max_effect=("effect_size", lambda x: x.abs().max()),
              above=("direction", lambda x: (x.isin(["above","stronger"])).sum()),
              below=("direction", lambda x: (x.isin(["below","weaker"])).sum()),
          )
          .sort_values("findings", ascending=False)
)
print(dim_summary.to_string())

# Top team summary
print(f"\n{'='*60}")
print("MOST FLAGGED TEAMS")
print(f"{'='*60}")
team_counts = df_all.reset_index().groupby("team")["rank"].count().sort_values(ascending=False)
print(team_counts.head(10).to_string())


# ── Save CSV ───────────────────────────────────────────────────────────────────

output_cols = [
    "team", "metric", "dimension", "dimension_value",
    "team_value", "league_avg", "std_devs_away",
    "sample_size", "interpretation",
]

top_df = df_all.head(TOP_N).reset_index()[["rank"] + output_cols]
top_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nTop {TOP_N} findings saved → {OUTPUT_CSV}")
print(f"Total findings available: {len(df_all)} "
      f"({type_counts.get('split',0)} splits + {type_counts.get('correlation',0)} correlations)")
