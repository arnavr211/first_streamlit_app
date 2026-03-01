"""
interpret.py

Reads the top findings from nfl_discovery_findings.csv, checks them against
the insights memory to avoid repeating past observations, then calls the
Claude API for novel analytical interpretation.

Each interpreted insight is logged to insights_memory/insights_log.jsonl.

Usage
-----
    python interpret.py                          # uses nfl_discovery_findings.csv
    python interpret.py --findings my_file.csv   # custom findings file
    python interpret.py --top 20                 # send top N findings (default 30)
    python interpret.py --dry-run                # build prompt but don't call API
    python interpret.py --show-memory            # print memory summary and exit
"""

import argparse
import json
import os
import sys
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pandas as pd

# Add project root to path so we can import memory_manager
sys.path.insert(0, str(Path(__file__).parent))
from insights_memory.memory_manager import (
    build_seen_sets,
    classify_finding,
    generate_memory_summary,
    load_insights,
    log_from_finding_row,
)

# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_FINDINGS_CSV = Path(__file__).parent / "nfl_discovery_findings.csv"
DEFAULT_TOP_N        = 30
MODEL                = "claude-opus-4-5-20251101"
MAX_TOKENS           = 4096

# How many findings to include in a single prompt group
# (keeps prompts focused; run again to process the next batch)
FINDINGS_PER_PROMPT  = 30

# ── Prompt assembly ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an elite NFL analytics expert. You receive statistically flagged
    findings from a 2025 play-by-play analysis engine and your job is to
    translate them into concise, insightful observations about team tendencies,
    scheme design, QB decision-making, and coaching patterns.

    ════════════════════════════════════════════════════════════════════════════
    GROUNDING RULES — STRICTLY ENFORCED
    ════════════════════════════════════════════════════════════════════════════

    You are a statistical analysis engine. Your task is to interpret the
    findings table strictly from a quantitative standpoint.

    Player Identity:
    • Player identifiers may be abbreviated (e.g., "T.Thornton").
    • You MUST NOT expand abbreviations into full names UNLESS roster_full_name
      is provided in the findings table — then use that EXACT value only.
    • You MUST NOT infer team membership, position, or background beyond what
      appears in the table. If roster_team or roster_position columns exist,
      use those EXACT values. Otherwise, do not mention team or position.
    • If roster fields are MISSING, refer to the player exactly as player_label
      (or team column) and do not speculate on identity.

    CRITICAL CONSTRAINTS (MUST FOLLOW):
    1. Do NOT infer or guess any player team.
    2. Do NOT infer or guess any player identifiers.
    3. Only reference player names exactly as they appear in the table.
    4. Only reference team values exactly as they appear in the roster_team field.
    5. Do NOT speculate on strategic implications (scheme, coaching, etc.).
    6. Do NOT introduce any player or team not present in the table.
    7. If a team is not explicitly listed in the row, do not mention a team.

    ════════════════════════════════════════════════════════════════════════════

    General Rules:
    1. DO NOT repeat or rephrase any insight already listed in the memory section.
    2. For teams that appear in the memory, look for second-order implications
       or deeper connections — not restatements of what is already known.
    3. Every claim must be grounded in at least one specific statistic from the
       findings provided (cite team, metric, value, and sample size).
    4. Rank insights by analytical surprise: findings that are counterintuitive,
       that suggest hidden structure, or that connect multiple patterns together
       should rank higher than findings that simply confirm existing narratives.
    5. For correlation findings (where the metric is "corr(X, Y)"), explain what
       it means that the relationship between X and Y is stronger or weaker than
       the league norm for this team.
    6. Output ONLY a JSON array — no prose outside the JSON.
""")

_USER_PROMPT_TEMPLATE = """\
## New Statistical Findings ({n_new} findings, filtered from {n_total} total)

Each finding is a team-level or player-level deviation from league average.
Format: team | metric | dimension=value | team_avg vs league_avg | z-score | n
For player findings, identity fields appear on a continuation line (→).

{findings_block}

---

{memory_section}

---

## Your Task

Analyze the new findings above. Produce {n_insights} distinct insights.
Return a JSON array where each element has exactly these keys:

  "rank"           : integer, 1 = most surprising
  "teams"          : list of team abbreviations referenced (use roster_team if present,
                     otherwise posteam; leave empty if neither is available)
  "metrics"        : list of metric names referenced
  "dimensions"     : list of dimension names referenced
  "dimension_values": list of dimension values referenced
  "evidence"       : object quoting EXACT row values that support this insight:
                     {{
                       "team": "<team column value or null>",
                       "player_label": "<player_label if player finding, else null>",
                       "roster_full_name": "<roster_full_name if present, else null>",
                       "metric": "<metric name>",
                       "dimension": "<dimension name>",
                       "dimension_value": "<dimension value>"
                     }}
  "insight_text"   : 2–4 sentences. Lead with the key number. Explain the
                     football implication. Note any caveats (small n, etc.).
                     For player findings: use roster_full_name if available,
                     otherwise use player_label EXACTLY as shown. Do NOT expand
                     abbreviations or infer team/position if not in the table.
  "novelty_reason" : 1 sentence explaining why this is NOT a repeat of
                     anything in the memory section.

CRITICAL: The "evidence" object MUST contain values copied exactly from the table.
This proves your insight is grounded in the data, not invented.

DO NOT repeat or rephrase any previous insight. Find genuinely new angles,
deeper connections, or second-order implications of patterns already identified.
"""


def _format_findings_block(df: pd.DataFrame) -> str:
    """Format the filtered findings DataFrame into a compact prompt block.

    For player-level findings (entity_type == 'player'), includes additional
    identity columns when present: player_label, player_id, posteam, and
    roster-grounded fields (roster_full_name, roster_team, roster_position).
    """
    # Detect if this is player-level data
    is_player_data = (
        ("entity_type" in df.columns and (df["entity_type"] == "player").any()) or
        ("player_id" in df.columns and df["player_id"].notna().any())
    )

    # Detect which roster columns are present and populated
    roster_cols = []
    for col in ["roster_full_name", "roster_team", "roster_position"]:
        if col in df.columns and df[col].notna().any():
            roster_cols.append(col)

    lines = []

    # Add header comment for player findings
    if is_player_data:
        if roster_cols:
            lines.append(f"# ROSTER-GROUNDED columns available: {', '.join(roster_cols)}")
            lines.append("# Use these EXACT values for player identity. Do not expand or infer.")
        else:
            lines.append("# WARNING: No roster columns present. Use player_label exactly as shown.")
            lines.append("# Do NOT expand abbreviations or infer team/position.")
        lines.append("")

    for _, row in df.iterrows():
        direction = "▲" if row["team_value"] > row["league_avg"] else "▼"
        delta = row["team_value"] - row["league_avg"]

        # Base line: team | metric | dimension=value | stats
        base = (
            f"  {str(row['team'])[:12]:<12} | {row['metric'][:24]:<24} | "
            f"{row['dimension'][:18]:<18}={str(row['dimension_value'])[:18]:<18} | "
            f"team={row['team_value']:>8.3f} vs lg={row['league_avg']:>8.3f} "
            f"({direction}{abs(delta):.3f}) | z={row['std_devs_away']:>6.2f} | "
            f"n={int(row['sample_size'])}"
        )

        # For player findings, add identity fields on a continuation line
        if is_player_data:
            identity_parts = []

            # Player label (abbreviated name from PBP)
            if "player_label" in row.index and pd.notna(row.get("player_label")):
                identity_parts.append(f"player_label={row['player_label']}")

            # Player ID (stable GSIS ID)
            if "player_id" in row.index and pd.notna(row.get("player_id")):
                identity_parts.append(f"player_id={row['player_id']}")

            # Team from PBP (posteam)
            if "posteam" in row.index and pd.notna(row.get("posteam")):
                identity_parts.append(f"posteam={row['posteam']}")

            # Roster-grounded fields (authoritative)
            for col in roster_cols:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    identity_parts.append(f"{col}={val}")

            if identity_parts:
                base += "\n      → " + " | ".join(identity_parts)

        lines.append(base)

    return "\n".join(lines)


def build_prompt(
    findings_df: pd.DataFrame,
    memory_summary: str,
    n_insights: int = 10,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) ready for the API call."""
    findings_block = _format_findings_block(findings_df)

    memory_section = memory_summary if memory_summary else (
        "## No Previous Insights\n\n"
        "This is the first analysis session — there is no prior memory to avoid."
    )

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        n_new=len(findings_df),
        n_total=len(findings_df),
        findings_block=findings_block,
        memory_section=memory_section,
        n_insights=n_insights,
    )
    return _SYSTEM_PROMPT, user_prompt


# ── API call ───────────────────────────────────────────────────────────────────

def call_claude(system_prompt: str, user_prompt: str) -> str:
    """Call the Claude API and return the raw response text."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running:\n  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


def parse_insights(raw_response: str) -> list[dict]:
    """
    Extract the JSON array from the API response.
    Strips markdown code fences if present.
    """
    text = raw_response.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    # Find the outermost JSON array
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(
            f"No JSON array found in API response.\nRaw response:\n{raw_response[:500]}"
        )

    return json.loads(text[start:end])


# ── Filtering ──────────────────────────────────────────────────────────────────

def filter_findings(
    df: pd.DataFrame,
    seen_fingerprints: set[str],
    seen_dim_keys: set[str],
    top_n: int = FINDINGS_PER_PROMPT,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Classify each finding, remove exact duplicates, return top_n non-duplicates.
    Also returns a count dict: {"new": X, "related": Y, "duplicate": Z}.
    """
    counts = {"new": 0, "related": 0, "duplicate": 0}
    rows = []

    for _, row in df.iterrows():
        status = classify_finding(
            team            = str(row["team"]),
            metric          = str(row["metric"]),
            dimension       = str(row["dimension"]),
            dimension_value = str(row["dimension_value"]),
            seen_fingerprints = seen_fingerprints,
            seen_dim_keys   = seen_dim_keys,
        )
        counts[status] += 1
        if status != "duplicate":
            rows.append(row)

    filtered = pd.DataFrame(rows).head(top_n)
    return filtered, counts


# ── Logging interpreted insights ───────────────────────────────────────────────

def log_insights_from_response(
    insights: list[dict],
    session_id: str,
) -> int:
    """Log each insight to the JSONL. Returns count of insights logged."""
    logged = 0
    for item in insights:
        teams           = item.get("teams", [])
        metrics         = item.get("metrics", [])
        dimensions      = item.get("dimensions", [])
        dimension_values= item.get("dimension_values", [])
        insight_text    = item.get("insight_text", "")

        if not insight_text:
            continue

        log_from_finding_row(
            row = {
                "team":            teams[0] if teams else "MULTI",
                "metric":          metrics[0] if metrics else "mixed",
                "dimension":       dimensions[0] if dimensions else "mixed",
                "dimension_value": dimension_values[0] if dimension_values else "",
                "finding_type":    "interpreted",
                "team_value":      None,
                "league_avg":      None,
                "std_devs_away":   None,
                "sample_size":     None,
            },
            insight_text = insight_text,
            session_id   = session_id,
        )
        logged += 1

    return logged


# ── Display ────────────────────────────────────────────────────────────────────

def display_insights(insights: list[dict]) -> None:
    """Print insights to stdout in a clean, readable format."""
    print(f"\n{'='*80}")
    print(f"INTERPRETED INSIGHTS  ({len(insights)} total)")
    print(f"{'='*80}\n")

    for item in sorted(insights, key=lambda x: x.get("rank", 99)):
        rank   = item.get("rank", "?")
        teams  = ", ".join(item.get("teams", []))
        mets   = ", ".join(item.get("metrics", []))
        text   = item.get("insight_text", "")
        novelty= item.get("novelty_reason", "")

        print(f"#{rank}  [{teams}]  {mets}")
        print("-" * 70)
        # Word-wrap at 78 chars
        for para in text.split("\n"):
            print(textwrap.fill(para, width=78, initial_indent="  ",
                                subsequent_indent="  "))
        if novelty:
            print(f"\n  New angle: {novelty}")
        print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Interpret nfl_discovery findings via Claude API.")
    parser.add_argument("--findings",    default=str(DEFAULT_FINDINGS_CSV),
                        help="Path to findings CSV (default: nfl_discovery_findings.csv)")
    parser.add_argument("--top",         type=int, default=DEFAULT_TOP_N,
                        help=f"Max findings to include in prompt (default: {DEFAULT_TOP_N})")
    parser.add_argument("--n-insights",  type=int, default=10,
                        help="Number of insights to request from Claude (default: 10)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print the prompt but do not call the API")
    parser.add_argument("--show-memory", action="store_true",
                        help="Print the current memory summary and exit")
    args = parser.parse_args()

    # ── Memory ──────────────────────────────────────────────────────────────────
    print("Loading insight memory...")
    past_insights = load_insights()
    seen_fp, seen_dk = build_seen_sets(past_insights)
    memory_summary = generate_memory_summary(past_insights)
    print(f"  {len(past_insights)} past insights loaded  "
          f"({len(seen_fp)} unique fingerprints)\n")

    if args.show_memory:
        if memory_summary:
            print(memory_summary)
        else:
            print("(No insights logged yet.)")
        return

    # ── Load findings ────────────────────────────────────────────────────────────
    findings_path = Path(args.findings)
    if not findings_path.exists():
        print(f"ERROR: findings file not found: {findings_path}")
        print("Run nfldiscovery.py first to generate it.")
        sys.exit(1)

    df = pd.read_csv(findings_path)
    print(f"Loaded {len(df)} findings from {findings_path.name}")

    # ── Filter duplicates ────────────────────────────────────────────────────────
    filtered_df, counts = filter_findings(df, seen_fp, seen_dk, top_n=args.top)
    print(f"  Classified: {counts['new']} new | "
          f"{counts['related']} related | {counts['duplicate']} duplicate")
    print(f"  Sending {len(filtered_df)} non-duplicate findings to Claude\n")

    if filtered_df.empty:
        print("All findings are duplicates of previously logged insights.")
        print("Run nfldiscovery.py to generate fresh findings, or clear the memory log.")
        return

    # ── Build prompt ─────────────────────────────────────────────────────────────
    session_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    system_prompt, user_prompt = build_prompt(
        filtered_df, memory_summary, n_insights=args.n_insights
    )

    if args.dry_run:
        print("=" * 80)
        print("SYSTEM PROMPT")
        print("=" * 80)
        print(system_prompt)
        print("\n" + "=" * 80)
        print("USER PROMPT")
        print("=" * 80)
        print(user_prompt)
        return

    # ── Call API ─────────────────────────────────────────────────────────────────
    print(f"Calling Claude API (model={MODEL}, session={session_id})...")
    try:
        raw_response = call_claude(system_prompt, user_prompt)
    except EnvironmentError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except anthropic.APIError as exc:
        print(f"\nAPI Error: {exc}")
        sys.exit(1)

    # ── Parse & display ──────────────────────────────────────────────────────────
    try:
        insights = parse_insights(raw_response)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"\nFailed to parse API response as JSON: {exc}")
        print("Raw response saved to interpret_raw_response.txt")
        Path("interpret_raw_response.txt").write_text(raw_response)
        sys.exit(1)

    display_insights(insights)

    # ── Log to memory ─────────────────────────────────────────────────────────────
    n_logged = log_insights_from_response(insights, session_id=session_id)
    print(f"\n{n_logged} insights logged to insights_memory/insights_log.jsonl")
    print(f"Session ID: {session_id}")


if __name__ == "__main__":
    main()
