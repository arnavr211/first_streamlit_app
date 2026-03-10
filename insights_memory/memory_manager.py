"""
insights_memory/memory_manager.py

Append-only insight logging and duplicate detection for the NFL analysis agent.

Fingerprint scheme
------------------
  Split findings:       "{team}|{metric}|{dimension}|{dimension_value}"
  Correlation findings: "{team}|{metric}|team_correlation"

"Substantially similar" = exact fingerprint match.
"Related" = same (team, metric, dimension) but different value — reported
  in the memory summary so Claude can look for second-order angles.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Path to the append-only log (relative to this file's directory)
_LOG_PATH = Path(__file__).parent / "insights_log.jsonl"


# ── Fingerprinting ─────────────────────────────────────────────────────────────

def get_fingerprint(team: str, metric: str, dimension: str,
                    dimension_value: str = "") -> str:
    """
    Return a stable, human-readable dedup key for a finding.
    Normalised to lowercase with inner whitespace collapsed.
    """
    parts = [
        team.strip().upper(),
        metric.strip().lower(),
        dimension.strip().lower(),
        dimension_value.strip().lower(),
    ]
    raw = "|".join(parts)
    # Short MD5 suffix so two near-identical strings don't accidentally collide
    md5 = hashlib.md5(raw.encode()).hexdigest()[:6]
    return f"{raw}#{md5}"


def get_dimension_key(team: str, metric: str, dimension: str) -> str:
    """
    Looser key: same team × metric × dimension regardless of bucket value.
    Used to detect related-but-not-identical findings.
    """
    return f"{team.upper()}|{metric.lower()}|{dimension.lower()}"


# ── Logging ────────────────────────────────────────────────────────────────────

def log_insight(
    insight_text: str,
    teams: list[str],
    metrics: list[str],
    dimensions: list[str],
    dimension_values: list[str],
    finding_type: str = "split",
    stats: dict[str, Any] | None = None,
    session_id: str = "",
    log_path: Path = _LOG_PATH,
) -> str:
    """
    Append one insight to the JSONL log. Returns the fingerprint written.

    Parameters
    ----------
    insight_text     : Human-readable description of the finding.
    teams            : Team abbreviations involved (e.g. ["LV"]).
    metrics          : Metric column names (e.g. ["epa"]).
    dimensions       : Dimension column names (e.g. ["play_type"]).
    dimension_values : Dimension bucket values (e.g. ["run"]).
    finding_type     : "split" or "correlation".
    stats            : Dict of supporting numbers (team_value, league_avg, etc.).
    session_id       : Optional tag to group entries from one run.
    """
    now = datetime.now(timezone.utc)
    fp = get_fingerprint(
        teams[0] if teams else "",
        metrics[0] if metrics else "",
        dimensions[0] if dimensions else "",
        dimension_values[0] if dimension_values else "",
    )

    record = {
        "id": fp,
        "date": now.strftime("%Y-%m-%d"),
        "timestamp": now.isoformat(),
        "session_id": session_id,
        "finding_type": finding_type,
        "insight_text": insight_text,
        "teams": teams,
        "metrics": metrics,
        "dimensions": dimensions,
        "dimension_values": dimension_values,
        "fingerprint": fp,
        "stats": stats or {},
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    return fp


# ── Loading ────────────────────────────────────────────────────────────────────

def load_insights(log_path: Path = _LOG_PATH) -> list[dict]:
    """
    Return all records from the JSONL log, oldest first.
    Returns an empty list if the file does not exist.
    """
    if not log_path.exists() or log_path.stat().st_size == 0:
        return []

    records = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[memory_manager] Warning: malformed JSON on line {line_no}: {exc}")
    return records


# ── Duplicate detection ────────────────────────────────────────────────────────

def build_seen_sets(
    insights: list[dict],
) -> tuple[set[str], set[str]]:
    """
    Return two sets built from the loaded insights:
      seen_fingerprints  — exact duplicates (same team+metric+dim+value)
      seen_dim_keys      — related findings (same team+metric+dim, any value)
    """
    seen_fingerprints: set[str] = set()
    seen_dim_keys: set[str] = set()

    for rec in insights:
        seen_fingerprints.add(rec.get("fingerprint", ""))
        teams  = rec.get("teams", [])
        mets   = rec.get("metrics", [])
        dims   = rec.get("dimensions", [])
        if teams and mets and dims:
            seen_dim_keys.add(get_dimension_key(teams[0], mets[0], dims[0]))

    return seen_fingerprints, seen_dim_keys


def is_duplicate(
    team: str,
    metric: str,
    dimension: str,
    dimension_value: str,
    seen_fingerprints: set[str],
) -> bool:
    """Return True if this exact finding has been logged before."""
    fp = get_fingerprint(team, metric, dimension, dimension_value)
    return fp in seen_fingerprints


def classify_finding(
    team: str,
    metric: str,
    dimension: str,
    dimension_value: str,
    seen_fingerprints: set[str],
    seen_dim_keys: set[str],
) -> str:
    """
    Returns one of:
      "new"       — never seen before
      "duplicate" — exact match in log
      "related"   — same team×metric×dimension, different value
    """
    fp  = get_fingerprint(team, metric, dimension, dimension_value)
    dk  = get_dimension_key(team, metric, dimension)
    if fp in seen_fingerprints:
        return "duplicate"
    if dk in seen_dim_keys:
        return "related"
    return "new"


# ── Memory summary for prompts ─────────────────────────────────────────────────

def generate_memory_summary(
    insights: list[dict] | None = None,
    max_per_team: int = 8,
    max_total: int = 60,
) -> str:
    """
    Build a concise, prompt-ready summary of previously logged insights,
    grouped by team and capped to avoid bloating the context window.

    Returns an empty string if there are no past insights.
    """
    if insights is None:
        insights = load_insights()
    if not insights:
        return ""

    # De-duplicate by fingerprint in case the log has redundant entries
    seen: set[str] = set()
    unique: list[dict] = []
    for rec in reversed(insights):          # most-recent first for the cap
        fp = rec.get("fingerprint", "")
        if fp and fp not in seen:
            seen.add(fp)
            unique.append(rec)

    # Re-sort oldest-first for readability
    unique = list(reversed(unique))

    # Group by team
    by_team: dict[str, list[dict]] = {}
    for rec in unique:
        teams = rec.get("teams", ["UNKNOWN"])
        team  = teams[0] if teams else "UNKNOWN"
        by_team.setdefault(team, []).append(rec)

    lines: list[str] = [
        "## Previously Identified Insights",
        f"(Total logged: {len(unique)} distinct findings across "
        f"{len(by_team)} teams)\n",
    ]

    total_written = 0
    for team in sorted(by_team.keys()):
        recs = by_team[team][:max_per_team]
        lines.append(f"**{team}**")
        for rec in recs:
            if total_written >= max_total:
                lines.append("  ... (additional findings omitted for brevity)")
                break
            text  = rec.get("insight_text", "")
            stats = rec.get("stats", {})
            dim   = rec.get("dimensions", ["?"])[0]
            dval  = rec.get("dimension_values", ["?"])[0]
            met   = rec.get("metrics", ["?"])[0]
            tv    = stats.get("team_value", "?")
            la    = stats.get("league_avg", "?")
            z     = stats.get("std_devs_away", "?")
            n     = stats.get("sample_size", "?")

            if isinstance(tv, float): tv = f"{tv:.3f}"
            if isinstance(la, float): la = f"{la:.3f}"
            if isinstance(z,  float): z  = f"{z:.2f}"

            lines.append(
                f"  • [{met} | {dim}={dval}] "
                f"team={tv} vs league={la} (z={z}, n={n})"
            )
            total_written += 1
        lines.append("")

    return "\n".join(lines)


# ── Convenience: log findings from a DataFrame row ────────────────────────────

def log_from_finding_row(row: dict, insight_text: str, session_id: str = "") -> str:
    """
    Helper to log a finding directly from a nfl_discovery_findings.csv row dict.
    Returns the fingerprint.
    """
    metric    = row.get("metric", "")
    dimension = row.get("dimension", "")
    dim_val   = str(row.get("dimension_value", ""))
    team      = row.get("team", "")

    # Correlation rows have metric like "corr(a, b)" — extract both metric names
    metrics = [metric]

    return log_insight(
        insight_text   = insight_text,
        teams          = [team],
        metrics        = metrics,
        dimensions     = [dimension],
        dimension_values=[dim_val],
        finding_type   = row.get("finding_type", "split"),
        stats          = {
            "team_value":     row.get("team_value"),
            "league_avg":     row.get("league_avg"),
            "std_devs_away":  row.get("std_devs_away"),
            "sample_size":    row.get("sample_size"),
        },
        session_id     = session_id,
    )
