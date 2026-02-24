"""
main.py — Daily NFL insights pipeline orchestrator.

Pipeline steps
--------------
1. Check if new nflverse data is available.
   → If yes: download it, flag fresh data, reset exploration to team-level,
             then also run a weekly delta report comparing the new week against
             the full season and previous insights (confirms / contradicts / new).
2. Select today's analysis lens via exploration_planner.py.
3. Run nfldiscovery with that lens → findings CSV in lens_findings/.
4. Load past insights from insights_memory/.
5. Send findings + memory context to Claude API → parse new insights.
6. Log new insights to insights_memory/insights_log.jsonl.
7. Write daily report to reports/YYYY-MM-DD-{lens}.md.
8. Send notification via notify.py.

Usage
-----
    python main.py                          # full pipeline
    python main.py --dry-run                # skip API calls; still writes findings + report
    python main.py --force-lens goal-line   # override lens selection
    python main.py --skip-api               # generate findings but skip Claude calls
    python main.py --n-insights 12          # number of insights to request (default 10)
    python main.py --seasons 2023,2024,2025 # multi-year data load
    python main.py --show-report            # print most recent report and exit
"""

import argparse
import json
import os
import sys
import textwrap
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from exploration_planner import (
    ALL_LENSES,
    FINDINGS_DIR,
    _check_phase_transition,
    load_state as load_planner_state,
    run_combo,
    run_lens,
    save_state as save_planner_state,
    select_lenses,
)
from insights_memory.memory_manager import (
    build_seen_sets,
    generate_memory_summary,
    load_insights,
)
from interpret import (
    DEFAULT_FINDINGS_CSV,
    build_prompt,
    call_claude,
    display_insights,
    filter_findings,
    log_insights_from_response,
    parse_insights,
)
from nfldiscovery import (
    _METRIC_BLACKLIST,
    _auto_detect_metrics,
    add_derived_columns,
    load_plays,
    run_correlation_analysis,
    run_split_analysis,
    _rank_and_interpret,
)
from roster_validation import (
    enrich_and_validate_players,
    load_rosters,
)
import notify

# ── Paths & constants ───────────────────────────────────────────────────────────

DATA_STATE_FILE = ROOT / "data_state.json"
REPORTS_DIR     = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DEFAULT_N_INSIGHTS = 10
# Minimum new-week plays to bother with a delta report
MIN_WEEK_PLAYS = 200
# How many season findings to include in delta comparison
DELTA_TOP_N = 40
# Max memory refs per finding in the delta prompt
MAX_MEMORY_REFS_PER_FINDING = 2


# ── Player enrichment ────────────────────────────────────────────────────────────

# Extra prompt instructions when roster validation could not be performed
_UNVALIDATED_PLAYER_WARNING = """\

⚠️  ROSTER VALIDATION COULD NOT BE PERFORMED  ⚠️

The player-level findings below have NOT been cross-checked against official
NFL rosters. Some player_id or posteam values may be stale, incorrect, or refer
to practice-squad / cut players. Apply extra scrutiny:

1. Cross-check any player name against your own knowledge before highlighting.
2. Be skeptical of findings for players you don't recognize — they may be
   practice-squad elevations whose sample size inflates their metrics.
3. Note sample size aggressively; small-sample outliers are more likely to be
   noise when roster validation is missing.
"""


def _is_player_findings(df: pd.DataFrame) -> bool:
    """Return True if the findings DataFrame contains player-level data."""
    if "entity_type" in df.columns:
        if (df["entity_type"] == "player").any():
            return True
    if "player_id" in df.columns:
        if df["player_id"].notna().any():
            return True
    return False


def _enrich_player_findings(
    findings_df: pd.DataFrame,
    findings_path: Path,
    seasons: list[int],
) -> tuple[pd.DataFrame, Path, dict, bool]:
    """
    Attempt to load rosters and enrich player-level findings.

    Returns:
        (enriched_df, enriched_path, validation_summary, roster_loaded)

    If roster loading fails (network error), returns:
        (original_df, original_path, {}, False)
    so the pipeline can proceed with a warning.
    """
    print("  Detected player-level findings — loading rosters for validation...")

    # ── Load rosters ─────────────────────────────────────────────────────────────
    rosters_df = None
    try:
        rosters_df = load_rosters(seasons)
        print(f"    Loaded {len(rosters_df):,} roster entries for seasons {seasons}")
    except Exception as e:
        print(f"    ⚠️  ROSTER LOAD FAILED: {e}")
        print("    Proceeding WITHOUT roster enrichment — interpret prompt will include warning")
        return findings_df, findings_path, {}, False

    # ── Enrich and validate ─────────────────────────────────────────────────────
    enriched_df, summary = enrich_and_validate_players(
        findings_df,
        rosters_df,
        player_id_col="player_id",
        posteam_col="posteam",
        drop_missing=False,   # keep findings even if roster match missing
        drop_mismatch=False,  # keep findings even if team mismatch
    )

    # ── Print validation summary ─────────────────────────────────────────────────
    print(f"    Validation summary:")
    print(f"      Total findings:        {summary.get('total_rows', len(findings_df))}")
    print(f"      Roster matched:        {summary.get('matched_rows', 0)}")
    print(f"      Missing roster match:  {summary.get('missing_rows', 0)}")
    print(f"      Team mismatch:         {summary.get('mismatch_rows', 0)}")

    # ── Write enriched CSV ───────────────────────────────────────────────────────
    stem = findings_path.stem  # e.g. "player-level_2026-02-24"
    enriched_path = findings_path.parent / f"{stem}_ENRICHED.csv"
    enriched_df.to_csv(enriched_path, index=False)
    print(f"    Enriched findings → {enriched_path.name}")

    return enriched_df, enriched_path, summary, True


# ── Data state ──────────────────────────────────────────────────────────────────

def load_data_state() -> dict:
    if DATA_STATE_FILE.exists():
        with open(DATA_STATE_FILE) as f:
            return json.load(f)
    return {
        "last_week":     -1,
        "last_games":    -1,
        "last_n_plays":  -1,
        "last_checked":  None,
        "last_download": None,
    }


def save_data_state(state: dict) -> None:
    with open(DATA_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def check_for_new_data(data_state: dict, seasons: list) -> tuple[bool, int, int]:
    """
    Determine whether new completed-game data is available.

    Primary:  compare play count in a fresh (non-cached) PBP load against the
              last known count stored in data_state.json.
    Fallback: compare schedule completed-game counts via nfl_data_py.import_schedules.

    Returns (has_new_data, latest_week, n_completed_games).
    """
    import nfl_data_py as nfl
    import warnings
    warnings.filterwarnings("ignore")

    last_week  = data_state.get("last_week", -1)
    last_plays = data_state.get("last_n_plays", -1)

    # ── Try schedule check first (lightweight) ──────────────────────────────────
    try:
        schedule  = nfl.import_schedules(seasons)
        completed = schedule[schedule["result"].notna()].copy()
        if "gameday" in completed.columns:
            today_str = str(date.today())
            completed = completed[
                completed["gameday"].notna() &
                (completed["gameday"] <= today_str)
            ]
        latest_week  = int(completed["week"].max()) if len(completed) > 0 else 0
        latest_games = len(completed)
        last_games   = data_state.get("last_games", -1)
        has_new      = (latest_week > last_week) or (latest_games > last_games)
        return has_new, latest_week, latest_games

    except Exception:
        pass  # network failure — fall through to play-count check

    # ── Fallback: compare cached play counts ────────────────────────────────────
    # This is cheap because nfl_data_py caches locally after the first pull.
    try:
        pbp = nfl.import_pbp_data(seasons)
        n_plays     = len(pbp)
        latest_week = int(pbp["week"].max()) if "week" in pbp.columns else last_week
        has_new     = n_plays > last_plays and latest_week > last_week
        return has_new, latest_week, n_plays
    except Exception as e:
        print(f"  [data-check] could not determine freshness: {e}")
        return False, last_week, last_plays


# ── Weekly delta helpers ────────────────────────────────────────────────────────

def _compute_week_values(
    week_plays: pd.DataFrame,
    season_findings: pd.DataFrame,
    min_week_plays: int = 5,
) -> pd.DataFrame:
    """
    For each finding in season_findings, compute the team's metric value for
    just the new week. Returns season_findings extended with:
      week_value  (float | None)
      n_week      (int)
      week_delta  (float | None)  week_value - team_value (season)
    """
    rows = []
    for _, row in season_findings.iterrows():
        team    = str(row["team"])
        metric  = str(row["metric"])
        dim     = str(row["dimension"])
        dim_v   = str(row["dimension_value"])

        week_value = None
        n_week     = 0

        if (dim in week_plays.columns and
                metric in week_plays.columns and
                "posteam" in week_plays.columns):
            mask = (
                (week_plays["posteam"] == team) &
                (week_plays[dim].astype(str) == dim_v) &
                week_plays[metric].notna()
            )
            n = int(mask.sum())
            if n >= min_week_plays:
                week_value = float(week_plays.loc[mask, metric].mean())
                n_week     = n

        season_val = row.get("team_value")
        week_delta = (
            round(week_value - float(season_val), 4)
            if (week_value is not None and season_val is not None)
            else None
        )

        rows.append({
            **row.to_dict(),
            "week_value": week_value,
            "n_week":     n_week,
            "week_delta": week_delta,
        })

    return pd.DataFrame(rows)


def _tag_memory_overlap(
    delta_rows: list[dict],
    past_insights: list[dict],
) -> list[dict]:
    """
    For each delta finding, attach any past insight texts that reference the
    same team+metric combination. Caps at MAX_MEMORY_REFS_PER_FINDING per row.
    """
    tagged = []
    for row in delta_rows:
        team   = row.get("team", "")
        metric = row.get("metric", "")
        refs   = []
        for insight in past_insights:
            if (team in (insight.get("teams") or []) and
                    metric in (insight.get("metrics") or [])):
                text = (insight.get("insight_text") or "").strip()
                if text and text not in refs:
                    refs.append(text)
                if len(refs) >= MAX_MEMORY_REFS_PER_FINDING:
                    break
        tagged.append({**row, "memory_refs": refs})
    return tagged


_DELTA_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an NFL weekly delta analyst. Your job is to compare a single week's
    patterns against the full-season baseline AND previously logged insights,
    then classify each finding as CONFIRMS, CONTRADICTS, or NEW SIGNAL.

    Classification rules:
      CONFIRMS    — The new week continues the same directional trend as the season
                    AND is consistent with a past logged insight. Quote the insight.
      CONTRADICTS — The new week REVERSES the season trend OR contradicts a past
                    logged insight. This is the highest-value category — something
                    has genuinely changed. Explain what likely caused the reversal.
      NEW SIGNAL  — No past insight covers this team+metric combination. Assess
                    whether this looks like a meaningful one-week spike or a real
                    emerging pattern.

    Additional rules:
      1. Rank CONTRADICTS entries highest overall — reversals are most informative.
      2. Every insight must cite week_value, season_value, and the delta.
      3. For CONTRADICTS, speculate on causation (scheme change, opponent, injury).
      4. For CONFIRMS, add second-order context beyond what the past insight said.
      5. For NEW SIGNAL, state your confidence level (1-week spike vs real trend).
      6. Output ONLY a JSON array — no prose outside the JSON.
""")

_DELTA_USER_TEMPLATE = """\
## Week {week} vs Full-Season Delta

Format: team | metric | dim=val | season_avg | week_{week} | delta | n_week | memory_refs

{findings_block}

---

{memory_section}

---

## Your Task

Produce {n_insights} insights from the delta above. Return a JSON array where each
element has exactly these keys:

  "delta_type"        : "confirms" | "contradicts" | "new_signal"
  "rank"              : integer, 1 = most analytically interesting
  "teams"             : list of team abbreviations
  "metrics"           : list of metric names
  "dimensions"        : list of dimension names
  "dimension_values"  : list of dimension values
  "week_value"        : the team's Week {week} value (float)
  "season_value"      : the team's full-season value (float)
  "delta"             : week_value minus season_value (float, + = improved this week)
  "insight_text"      : 2–4 sentences. Lead with the delta number. Explain the
                        football implication. Note sample size caveats.
  "referenced_insight": for confirms/contradicts, first sentence of the relevant
                        past logged insight; empty string for new_signal.
  "novelty_reason"    : 1 sentence on why this is analytically interesting THIS week.

Rank CONTRADICTS entries highest. DO NOT generate insights where week_value is None.
"""


def _build_delta_prompt(
    tagged_rows: list[dict],
    memory_summary: str,
    latest_week: int,
    n_insights: int,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the weekly delta Claude call."""
    lines = []
    for row in tagged_rows:
        if row.get("week_value") is None:
            continue  # skip findings with no new-week data
        team      = str(row["team"])
        metric    = str(row["metric"])
        dim       = str(row["dimension"])
        dim_v     = str(row["dimension_value"])
        s_val     = row.get("team_value")
        w_val     = row.get("week_value")
        delta     = row.get("week_delta")
        n_week    = row.get("n_week", 0)
        refs      = row.get("memory_refs", [])

        s_str  = f"{float(s_val):+.3f}" if s_val is not None else "N/A"
        w_str  = f"{float(w_val):+.3f}" if w_val is not None else "N/A"
        d_str  = f"{float(delta):+.3f}" if delta is not None else "N/A"
        ref_str = " | ".join(r[:80] for r in refs) if refs else "no_prior_insight"

        lines.append(
            f"  {team:<5} | {metric[:24]:<24} | {dim[:16]:<16}={str(dim_v)[:18]:<18} | "
            f"season={s_str} | wk{latest_week}={w_str} | Δ={d_str} | "
            f"n={n_week} | {ref_str}"
        )

    findings_block = "\n".join(lines) if lines else "  (no findings with week-level data)"

    memory_section = memory_summary if memory_summary else (
        "## No Previous Insights\n\nThis is the first analysis session."
    )

    user_prompt = _DELTA_USER_TEMPLATE.format(
        week=latest_week,
        findings_block=findings_block,
        memory_section=memory_section,
        n_insights=n_insights,
    )
    return _DELTA_SYSTEM_PROMPT, user_prompt


def generate_weekly_delta_report(
    plays: pd.DataFrame,
    season_findings_path: Path,
    latest_week: int,
    past_insights: list,
    memory_summary: str,
    session_id: str,
    run_date: str,
    dry_run: bool = False,
    skip_api: bool = False,
    n_insights: int = DEFAULT_N_INSIGHTS,
) -> Optional[Path]:
    """
    Run the weekly delta pipeline:
      1. Filter plays to the new week.
      2. Load season findings (already computed for the standard pipeline).
      3. For each season finding, compute the team's metric in the new week.
      4. Tag memory overlaps.
      5. Build delta-specific Claude prompt.
      6. Call API (or skip on --dry-run / --skip-api).
      7. Write weekly delta report to reports/.

    Returns the report path, or None on failure.
    """
    print(f"\n{'═'*65}")
    print(f"  WEEKLY DELTA REPORT — Week {latest_week}")
    print(f"{'═'*65}")

    # ── Filter to new week ───────────────────────────────────────────────────────
    if "week" not in plays.columns:
        print("  [delta] 'week' column not found — skipping delta report")
        return None

    week_plays = plays[plays["week"] == latest_week].copy()
    n_week     = len(week_plays)
    print(f"  Week {latest_week}: {n_week:,} plays  "
          f"({week_plays['game_id'].nunique()} games)")

    if n_week < MIN_WEEK_PLAYS:
        print(f"  Only {n_week} plays in Week {latest_week} "
              f"(min {MIN_WEEK_PLAYS}) — skipping delta report")
        return None

    # ── Load season findings ────────────────────────────────────────────────────
    if not season_findings_path or not season_findings_path.exists():
        print("  [delta] season findings file not found — skipping")
        return None

    season_df = pd.read_csv(season_findings_path).head(DELTA_TOP_N)
    print(f"  Season findings loaded: {len(season_df)} rows from {season_findings_path.name}")

    # ── Compute week-level values for each season finding ─────────────────────
    print(f"  Computing Week {latest_week} values for each finding...")
    delta_df  = _compute_week_values(week_plays, season_df)
    has_data  = delta_df["week_value"].notna().sum()
    print(f"  {has_data}/{len(delta_df)} findings have Week {latest_week} data")

    # ── Tag memory overlaps ──────────────────────────────────────────────────────
    delta_rows = _tag_memory_overlap(delta_df.to_dict("records"), past_insights)
    n_with_memory = sum(1 for r in delta_rows if r.get("memory_refs"))
    print(f"  {n_with_memory} findings overlap with logged memory")

    # ── Build prompt ─────────────────────────────────────────────────────────────
    system_prompt, user_prompt = _build_delta_prompt(
        delta_rows, memory_summary, latest_week, n_insights
    )

    if dry_run or skip_api:
        print(f"\n{'─'*65}")
        print("  [dry-run] DELTA USER PROMPT PREVIEW (first 1500 chars)")
        print(f"{'─'*65}")
        print(user_prompt[:1500])
        print(f"{'─'*65}")

        # Write a partial delta report marked as dry-run
        report_path = _write_delta_report(
            run_date, latest_week, delta_rows, insights=None,
            n_season_plays=len(plays), n_week_plays=n_week,
            n_memory=len(past_insights), session_id=session_id,
            dry_run=True,
        )
        return report_path

    # ── Call API ──────────────────────────────────────────────────────────────────
    print(f"  Calling Claude API for delta analysis...")
    try:
        raw = call_claude(system_prompt, user_prompt)
    except (EnvironmentError, Exception) as e:
        print(f"  [delta] API call failed: {e}")
        return None

    # ── Parse response ────────────────────────────────────────────────────────────
    try:
        delta_insights = parse_insights(raw)
    except (ValueError, Exception) as e:
        print(f"  [delta] failed to parse API response: {e}")
        fallback = REPORTS_DIR / f"delta_raw_{run_date}.txt"
        fallback.write_text(raw)
        print(f"  Raw response saved to {fallback.name}")
        return None

    print(f"  Parsed {len(delta_insights)} delta insights")

    # Augment each delta insight with delta_type metadata for the logger
    for item in delta_insights:
        if "finding_type" not in item:
            item["finding_type"] = f"delta:{item.get('delta_type', 'unknown')}"

    # Log delta insights to memory (marked with delta finding_type)
    n_logged = log_insights_from_response(delta_insights, session_id=session_id)
    print(f"  {n_logged} delta insights logged to memory")

    # ── Write report ──────────────────────────────────────────────────────────────
    report_path = _write_delta_report(
        run_date, latest_week, delta_rows, insights=delta_insights,
        n_season_plays=len(plays), n_week_plays=n_week,
        n_memory=len(past_insights), session_id=session_id,
        dry_run=False,
    )
    return report_path


# ── Report writers ──────────────────────────────────────────────────────────────

def _write_delta_report(
    run_date: str,
    latest_week: int,
    delta_rows: list[dict],
    insights: Optional[list],
    n_season_plays: int,
    n_week_plays: int,
    n_memory: int,
    session_id: str,
    dry_run: bool = False,
) -> Path:
    """Write the weekly delta markdown report."""
    report_path = REPORTS_DIR / f"{run_date}-week-{latest_week}-delta.md"

    confirms    = [i for i in (insights or []) if i.get("delta_type") == "confirms"]
    contradicts = [i for i in (insights or []) if i.get("delta_type") == "contradicts"]
    new_signals = [i for i in (insights or []) if i.get("delta_type") == "new_signal"]

    dry_banner = "\n> ⚠️  **DRY RUN** — Claude API was not called.\n" if dry_run else ""

    lines = [
        f"# Weekly Delta Report — Week {latest_week} ({run_date})",
        "",
        f"> New data detected: **Week {latest_week}** added to the 2025 season.",
        f"> Comparing Week {latest_week} patterns against the full season to date,",
        f"> cross-referencing against **{n_memory} previously logged insights**.",
        dry_banner,
        "---",
        "",
        "## Delta Summary",
        "",
        "| Category | Count |",
        "|----------|-------|",
        f"| ✓ Confirms past insight | {len(confirms)} |",
        f"| ✗ Contradicts past insight | {len(contradicts)} |",
        f"| ★ New signal | {len(new_signals)} |",
        f"| — No week data | "
        f"{sum(1 for r in delta_rows if r.get('week_value') is None)} |",
        "",
    ]

    def _insight_block(item: dict) -> list[str]:
        rank    = item.get("rank", "?")
        teams   = ", ".join(item.get("teams") or [])
        metrics = ", ".join(f"`{m}`" for m in (item.get("metrics") or []))
        dims    = ", ".join(f"`{d}`" for d in (item.get("dimensions") or []))
        text    = item.get("insight_text", "")
        ref     = item.get("referenced_insight", "")
        novelty = item.get("novelty_reason", "")
        w_val   = item.get("week_value")
        s_val   = item.get("season_value")
        delta   = item.get("delta")

        stats_parts = []
        if w_val is not None:
            stats_parts.append(f"**Week {latest_week}:** {float(w_val):+.3f}")
        if s_val is not None:
            stats_parts.append(f"**Season:** {float(s_val):+.3f}")
        if delta is not None:
            stats_parts.append(f"**Δ:** {float(delta):+.3f}")
        stats_line = " | ".join(stats_parts)

        block = [
            f"### {rank} · [{teams}] · {metrics}",
            "",
        ]
        if stats_line:
            block += [stats_line, ""]
        block += [text, ""]
        if ref:
            block += [f"> **Previously logged:** *{ref.rstrip('.')}.*", ""]
        if novelty:
            block += [f"*{novelty}*", ""]
        block.append("---")
        block.append("")
        return block

    # ── Contradictions first (most interesting) ─────────────────────────────────
    if contradicts:
        lines += [
            "## ✗ Contradicted Patterns — Something Changed This Week",
            "",
            "_These are the highest-value findings: Week "
            f"{latest_week} reversed an established season trend._",
            "",
        ]
        for item in sorted(contradicts, key=lambda x: x.get("rank", 99)):
            lines.extend(_insight_block(item))

    # ── Confirmations ────────────────────────────────────────────────────────────
    if confirms:
        lines += [
            "## ✓ Confirmed Patterns — Still Holding True",
            "",
            "_Week "
            f"{latest_week} continues patterns we've already logged. "
            "Look for second-order implications._",
            "",
        ]
        for item in sorted(confirms, key=lambda x: x.get("rank", 99)):
            lines.extend(_insight_block(item))

    # ── New signals ──────────────────────────────────────────────────────────────
    if new_signals:
        lines += [
            f"## ★ New Week-{latest_week} Signals — Not Seen Before",
            "",
            "_Patterns with no corresponding entry in insights memory._",
            "",
        ]
        for item in sorted(new_signals, key=lambda x: x.get("rank", 99)):
            lines.extend(_insight_block(item))

    if insights is None and not dry_run:
        lines += ["## Insights\n\n_(API call was skipped or failed.)_\n"]

    # ── Footer ───────────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines += [
        "---",
        "",
        "## Session Notes",
        "",
        f"- **Session ID:** `{session_id}`",
        f"- **Season plays:** {n_season_plays:,}",
        f"- **Week {latest_week} plays:** {n_week_plays:,}",
        f"- **Season findings analyzed:** {len(delta_rows)}",
        f"- **Findings with week-level data:** "
        f"{sum(1 for r in delta_rows if r.get('week_value') is not None)}",
        f"- **Memory loaded:** {n_memory} past insights",
        f"- **Generated:** {ts}",
        "",
        "*Pipeline: main.py → weekly delta → nfldiscovery → interpret → notify*",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Delta report → {report_path.name}")
    return report_path


def write_daily_report(
    run_date: str,
    lens,
    findings_df: pd.DataFrame,
    insights: Optional[list],
    counts: dict,
    data_status: dict,
    session_id: str,
    dry_run: bool = False,
) -> Path:
    """Write the standard daily insights report to reports/YYYY-MM-DD-{lens}.md."""
    lens_id   = lens.id if hasattr(lens, "id") else str(lens)
    lens_name = lens.name if hasattr(lens, "name") else lens_id
    lens_desc = lens.description if hasattr(lens, "description") else ""
    lens_cat  = lens.category if hasattr(lens, "category") else ""

    report_path = REPORTS_DIR / f"{run_date}-{lens_id}.md"
    dry_banner  = "\n> ⚠️  **DRY RUN** — Claude API was not called.\n" if dry_run else ""

    # ── Header ───────────────────────────────────────────────────────────────────
    fresh_line = ""
    if data_status.get("fresh_data"):
        fresh_line = (
            f"\n> 🆕 **New data detected** — "
            f"Week {data_status.get('latest_week', '?')} "
            f"({data_status.get('n_completed', '?')} completed games).\n"
        )

    lines = [
        f"# NFL Insights Report — {run_date}",
        "",
        f"**Lens:** {lens_name} (`{lens_id}`) — {lens_cat}  ",
        f"**Description:** {lens_desc}",
        fresh_line,
        dry_banner,
        "---",
        "",
        "## Pipeline Status",
        "",
        "| Step | Value |",
        "|------|-------|",
        f"| Season | {', '.join(str(s) for s in data_status.get('seasons', [2025]))} |",
        f"| New data | {'Yes — Week ' + str(data_status.get('latest_week','?')) if data_status.get('fresh_data') else 'No'} |",
        f"| Plays analyzed | {data_status.get('n_plays', 'N/A'):,} |"
        if isinstance(data_status.get("n_plays"), int)
        else f"| Plays analyzed | {data_status.get('n_plays', 'N/A')} |",
        f"| Findings generated | {counts.get('total', len(findings_df))} |",
        f"| Sent to Claude | {counts.get('sent', len(findings_df))} "
        f"({counts.get('new', 0)} new · "
        f"{counts.get('related', 0)} related · "
        f"{counts.get('duplicate', 0)} duplicate) |",
        f"| Insights generated | {len(insights) if insights else 0} |",
        f"| Memory loaded | {counts.get('n_memory', 0)} past insights |",
        "",
        "---",
        "",
    ]

    # ── Top findings table ────────────────────────────────────────────────────────
    lines += [
        "## Top Statistical Findings",
        "",
        "| # | Team | Metric | Dimension = Value | Team | League | z | n |",
        "|---|------|--------|------------------|------|--------|---|---|",
    ]
    for _, row in findings_df.head(20).iterrows():
        direction = "▲" if row.get("team_value", 0) > row.get("league_avg", 0) else "▼"
        lines.append(
            f"| {int(row.get('rank', 0)) if 'rank' in row.index else ''} "
            f"| {row['team']} "
            f"| `{row['metric']}` "
            f"| `{row['dimension']}`=`{str(row['dimension_value'])[:20]}` "
            f"| {direction}{float(row['team_value']):.3f} "
            f"| {float(row['league_avg']):.3f} "
            f"| {float(row['std_devs_away']):.2f} "
            f"| {int(row['sample_size'])} |"
        )

    lines += ["", "---", ""]

    # ── Insights ──────────────────────────────────────────────────────────────────
    if insights:
        lines += [
            f"## AI-Interpreted Insights ({len(insights)} new)",
            "",
        ]
        for item in sorted(insights, key=lambda x: x.get("rank", 99)):
            rank    = item.get("rank", "?")
            teams   = ", ".join(item.get("teams") or [])
            metrics = ", ".join(f"`{m}`" for m in (item.get("metrics") or []))
            text    = item.get("insight_text", "")
            novelty = item.get("novelty_reason", "")

            lines += [
                f"### {rank} · [{teams}] · {metrics}",
                "",
                text,
                "",
            ]
            if novelty:
                lines += [f"> **New angle:** {novelty}", ""]
            lines += ["---", ""]
    elif dry_run:
        lines += [
            "## Insights",
            "",
            "_(Dry run — Claude API was not called.)_",
            "",
        ]
    else:
        lines += [
            "## Insights",
            "",
            "_(API call was skipped or returned no parseable insights.)_",
            "",
        ]

    # ── Footer ────────────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines += [
        "---",
        "",
        "## Session",
        "",
        f"- **ID:** `{session_id}`",
        f"- **Generated:** {ts}",
        f"- **Findings file:** `lens_findings/{findings_df.get('_source_file', run_date + '-' + lens_id + '.csv') if hasattr(findings_df, 'attrs') else run_date + '-' + lens_id + '.csv'}`",
        "",
        "*Pipeline: exploration_planner → nfldiscovery → interpret → notify*",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Daily report → {report_path.name}")
    return report_path


# ── Utilities ───────────────────────────────────────────────────────────────────

def _show_latest_report() -> None:
    reports = sorted(REPORTS_DIR.glob("*.md"))
    if not reports:
        print("No reports found in reports/")
        return
    latest = reports[-1]
    print(f"\n{'='*65}")
    print(f"  {latest.name}")
    print(f"{'='*65}\n")
    print(latest.read_text(encoding="utf-8"))


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily NFL insights pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run",     action="store_true",
                        help="Skip API calls; still writes findings and a partial report")
    parser.add_argument("--skip-api",    action="store_true",
                        help="Generate findings but do not call Claude API")
    parser.add_argument("--force-lens",  type=str, default=None,
                        help="Override lens selection (e.g. goal-line)")
    parser.add_argument("--n-insights",  type=int, default=DEFAULT_N_INSIGHTS,
                        help=f"Number of insights to request (default {DEFAULT_N_INSIGHTS})")
    parser.add_argument("--seasons",     type=str, default="2025",
                        help="Comma-separated seasons (default: 2025)")
    parser.add_argument("--show-report", action="store_true",
                        help="Print the most recent report and exit")
    args = parser.parse_args()

    if args.show_report:
        _show_latest_report()
        return

    seasons     = [int(s.strip()) for s in args.seasons.split(",")]
    run_date    = date.today().isoformat()
    session_id  = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-"
                   f"{uuid.uuid4().hex[:6]}")

    print(f"\n{'='*65}")
    print(f"  NFL DAILY INSIGHTS PIPELINE")
    print(f"  {run_date}  |  session {session_id}")
    print(f"{'='*65}\n")

    # ── Step 1: Check for new data ───────────────────────────────────────────────
    print("Step 1 — Checking for new nflverse data...")
    data_state = load_data_state()
    has_new, latest_week, n_completed = check_for_new_data(data_state, seasons)
    data_state["last_checked"] = run_date

    if has_new:
        print(f"  ✓ NEW DATA: Week {latest_week}  ({n_completed} completed games)")
        data_state["last_week"]  = latest_week
        data_state["last_games"] = n_completed
        data_state["last_download"] = run_date
        # Reset exploration priority to broadest lens
        planner_state = load_planner_state()
        planner_state["lens_history"].pop("team-level", None)
        save_planner_state(planner_state)
        print("  → Exploration priority reset to team-level analysis")
    else:
        print(f"  No new data (last known: Week {data_state.get('last_week','?')}, "
              f"{data_state.get('last_games','?')} games)")

    data_status = {
        "fresh_data":  has_new,
        "latest_week": latest_week,
        "n_completed": n_completed,
        "seasons":     seasons,
    }

    # ── Step 2: Select lens ──────────────────────────────────────────────────────
    print("\nStep 2 — Selecting analysis lens...")
    planner_state = load_planner_state()

    if args.force_lens:
        if args.force_lens not in ALL_LENSES:
            print(f"ERROR: Unknown lens '{args.force_lens}'. "
                  f"Run with --help or see exploration_planner.py --list")
            sys.exit(1)
        selected = ALL_LENSES[args.force_lens]
        is_combo = False
        print(f"  Forced lens: {selected.name} [{selected.id}]")
    elif has_new:
        selected = ALL_LENSES["team-level"]
        is_combo = False
        print(f"  Fresh data → team-level lens selected")
    else:
        _check_phase_transition(planner_state)
        selection = select_lenses(planner_state, n=1)
        first     = selection[0]
        is_combo  = isinstance(first, tuple)
        selected  = first
        if is_combo:
            la, lb = selected
            print(f"  Combo: {la.name} × {lb.name}")
        else:
            print(f"  Lens: {selected.name} [{selected.id}]")

    # ── Step 3: Load data + run lens ─────────────────────────────────────────────
    print("\nStep 3 — Loading play-by-play data and running discovery...")
    plays = load_plays(seasons)
    data_status["n_plays"] = len(plays)
    data_state["last_n_plays"] = len(plays)

    if is_combo:
        la, lb       = selected
        findings_path = run_combo(la, lb, plays, run_date)
        lens_for_report = la  # use first lens for report metadata
    else:
        findings_path = run_lens(selected, plays, run_date)
        lens_for_report = selected

    if findings_path is None or not findings_path.exists():
        print("ERROR: lens produced no findings — aborting pipeline")
        sys.exit(1)

    findings_df = pd.read_csv(findings_path)
    print(f"  Findings loaded: {len(findings_df)} rows from {findings_path.name}")

    # ── Step 3b: Enrich player-level findings with roster data ───────────────────
    roster_validated = True  # assume True; set False if player findings + roster load fails
    validation_summary = {}

    if _is_player_findings(findings_df):
        findings_df, findings_path, validation_summary, roster_validated = \
            _enrich_player_findings(findings_df, findings_path, seasons)

    # Update planner state
    if is_combo:
        la, lb = selected
        pair   = sorted([la.id, lb.id])
        if pair not in planner_state["combo_history"]:
            planner_state["combo_history"].append(pair)
        for lens in (la, lb):
            h = planner_state["lens_history"].setdefault(lens.id, {
                "use_count": 0, "last_used": None, "findings_files": []
            })
            h["use_count"] += 1
            h["last_used"]  = run_date
            h["findings_files"].append(str(findings_path))
    else:
        h = planner_state["lens_history"].setdefault(selected.id, {
            "use_count": 0, "last_used": None, "findings_files": []
        })
        h["use_count"] += 1
        h["last_used"]  = run_date
        h["findings_files"].append(str(findings_path))
    _check_phase_transition(planner_state)
    save_planner_state(planner_state)

    # ── Step 4: Load past insights ────────────────────────────────────────────────
    print("\nStep 4 — Loading insights memory...")
    past_insights   = load_insights()
    seen_fp, seen_dk = build_seen_sets(past_insights)
    memory_summary  = generate_memory_summary(past_insights)
    print(f"  {len(past_insights)} past insights  "
          f"({len(seen_fp)} unique fingerprints)")

    # ── Step 5: Filter + call Claude API ─────────────────────────────────────────
    print("\nStep 5 — Filtering findings and calling Claude API...")
    filtered_df, filter_counts = filter_findings(
        findings_df, seen_fp, seen_dk, top_n=30
    )
    print(f"  {filter_counts['new']} new | "
          f"{filter_counts['related']} related | "
          f"{filter_counts['duplicate']} duplicate")
    print(f"  Sending {len(filtered_df)} findings to Claude")

    counts = {
        "total":     len(findings_df),
        "sent":      len(filtered_df),
        "n_memory":  len(past_insights),
        **filter_counts,
        **validation_summary,  # includes roster_matched, missing_roster_match, team_mismatch
    }

    insights = None
    if not filtered_df.empty and not args.dry_run and not args.skip_api:
        system_prompt, user_prompt = build_prompt(
            filtered_df, memory_summary, n_insights=args.n_insights
        )
        # Inject unvalidated-player warning if roster load failed for player findings
        if _is_player_findings(filtered_df) and not roster_validated:
            print("  ⚠️  Injecting unvalidated-player warning into prompt")
            user_prompt = _UNVALIDATED_PLAYER_WARNING + user_prompt
        try:
            raw      = call_claude(system_prompt, user_prompt)
            insights = parse_insights(raw)
            print(f"  Received {len(insights)} insights from Claude")
        except EnvironmentError as e:
            print(f"  WARNING: {e}")
            print("  Continuing without insights (set ANTHROPIC_API_KEY to enable)")
        except Exception as e:
            print(f"  WARNING: API call failed: {e}")
    elif args.dry_run:
        print("  [dry-run] skipping API call")
    elif filtered_df.empty:
        print("  All findings are duplicates — no API call needed")

    # ── Step 6: Log insights ──────────────────────────────────────────────────────
    n_logged = 0
    if insights:
        print("\nStep 6 — Logging new insights to memory...")
        n_logged = log_insights_from_response(insights, session_id=session_id)
        print(f"  {n_logged} insights logged")
    else:
        print("\nStep 6 — No new insights to log")

    # ── Step 7: Write daily report ────────────────────────────────────────────────
    print("\nStep 7 — Writing daily report...")
    report_path = write_daily_report(
        run_date       = run_date,
        lens           = lens_for_report,
        findings_df    = findings_df,
        insights       = insights,
        counts         = counts,
        data_status    = data_status,
        session_id     = session_id,
        dry_run        = args.dry_run,
    )

    # ── Weekly delta report (when new data arrived) ───────────────────────────────
    delta_report_path = None
    if has_new:
        print("\nStep 7b — Generating weekly delta report...")
        # Reload memory (may have new insights from step 6 logged)
        past_insights_updated  = load_insights()
        memory_summary_updated = generate_memory_summary(past_insights_updated)

        delta_report_path = generate_weekly_delta_report(
            plays             = plays,
            season_findings_path = findings_path,
            latest_week       = latest_week,
            past_insights     = past_insights_updated,
            memory_summary    = memory_summary_updated,
            session_id        = session_id,
            run_date          = run_date,
            dry_run           = args.dry_run,
            skip_api          = args.skip_api,
            n_insights        = args.n_insights,
        )

    # Persist data state
    save_data_state(data_state)

    # ── Step 8: Send notification ─────────────────────────────────────────────────
    print("\nStep 8 — Sending notification...")
    top_insight = ""
    if insights:
        top = sorted(insights, key=lambda x: x.get("rank", 99))
        top_insight = top[0].get("insight_text", "")[:200] if top else ""

    notify_report = {
        "date":        run_date,
        "lens":        lens_for_report.id if hasattr(lens_for_report, "id") else "combo",
        "lens_name":   lens_for_report.name if hasattr(lens_for_report, "name") else "Combo",
        "n_plays":     data_status.get("n_plays", 0),
        "n_findings":  counts["total"],
        "n_sent":      counts["sent"],
        "n_insights":  len(insights) if insights else 0,
        "n_memory":    len(past_insights),
        "fresh_data":  has_new,
        "latest_week": latest_week,
        "dry_run":     args.dry_run,
        "report_path": str(report_path),
        "top_insight": top_insight,
        "session_id":  session_id,
    }
    notify.send(notify_report)

    if delta_report_path:
        delta_notify = {**notify_report,
                        "lens": f"week-{latest_week}-delta",
                        "lens_name": f"Week {latest_week} Delta",
                        "report_path": str(delta_report_path),
                        "top_insight": ""}
        notify.send(delta_notify)

    # ── Final summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Pipeline complete")
    print(f"  Session: {session_id}")
    print(f"  Insights logged: {n_logged}")
    print(f"  Daily report:  {report_path.name}")
    if delta_report_path:
        print(f"  Delta report:  {delta_report_path.name}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
