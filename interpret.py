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
    You are an NFL analytics interpreter. You receive statistically flagged
    findings from a 2025 play-by-play analysis engine. Each finding has a
    canonical entity_key that uniquely identifies the player or team.

    ════════════════════════════════════════════════════════════════════════════
    CRITICAL: ENTITY-KEY BASED RESPONSES
    ════════════════════════════════════════════════════════════════════════════

    You are NOT the source of truth for identity fields.

    Each finding is tagged with an entity_key in brackets: [Name|Team|Position]
    You MUST return this exact entity_key in your response. The post-processor
    will use it to look up canonical identity fields from the authoritative
    registry. You do NOT need to include player name, team, or position in
    your response — they will be filled in automatically.

    WHAT TO RETURN:
    • entity_key: EXACT string from the finding (e.g., "Tyquan Thornton|KC|WR")
    • interpretation: Your analytical insight about this finding

    WHAT NOT TO RETURN:
    • Do NOT return player name, team, or position separately
    • Do NOT infer or expand any identity information
    • Do NOT mention teams or players not present in the findings

    ════════════════════════════════════════════════════════════════════════════

    General Rules:
    1. DO NOT repeat or rephrase any insight already listed in the memory section.
    2. Every claim must be grounded in at least one specific statistic from the
       findings provided (cite metric, value, and sample size).
    3. Rank insights by analytical surprise: counterintuitive findings rank higher.
    4. Output ONLY a JSON array — no prose outside the JSON.
""")

_USER_PROMPT_TEMPLATE = """\
## New Statistical Findings ({n_new} findings, filtered from {n_total} total)

Each finding is keyed by entity_key in brackets: [Name|Team|Position]
You MUST return this exact entity_key — identity fields will be filled in later.

{findings_block}

---

{memory_section}

---

## Your Task

Analyze the findings above. Produce {n_insights} distinct insights.
Return a JSON array where each element has EXACTLY these keys:

  "rank"           : integer, 1 = most surprising
  "entity_key"     : EXACT string from brackets, e.g. "Tyquan Thornton|KC|WR"
                     For team-level findings: "|KC|" or similar
  "metrics"        : list of metric names referenced
  "interpretation" : 2–4 sentences. Lead with the key statistic.
                     Do NOT include player name, team, or position — these will
                     be filled from the canonical registry using entity_key.
                     Focus purely on what the numbers mean analytically.
  "novelty_reason" : 1 sentence explaining why this is NOT a repeat of
                     anything in the memory section.

CRITICAL RULES:
1. The entity_key MUST be copied EXACTLY from the finding (including case and pipes)
2. Do NOT invent entity_keys or modify them in any way
3. Do NOT mention player names or teams in interpretation — use "this player" or
   "this entity" since identity will be injected from the registry
4. Do NOT repeat or rephrase any previous insight

Example response format:
[
  {{
    "rank": 1,
    "entity_key": "Tyquan Thornton|KC|WR",
    "metrics": ["air_yards"],
    "interpretation": "This receiver averages 27.1 air yards in shotgun formations, dramatically higher than the league average of 8.1 (z=11.55, n=36). This suggests a deep-threat role with high average depth of target.",
    "novelty_reason": "First observation of this specific shotgun air yards pattern."
  }}
]
"""


def _format_findings_block(df: pd.DataFrame, registry: dict[str, dict] = None) -> str:
    """Format the filtered findings DataFrame into a compact prompt block.

    For player-level findings, includes the entity_key for each row.
    The entity_key is the canonical identifier that Claude must return.
    """
    # Detect if this is player-level data
    is_player_data = (
        ("entity_type" in df.columns and (df["entity_type"] == "player").any()) or
        ("player_id" in df.columns and df["player_id"].notna().any())
    )

    lines = []

    # Add header explaining entity_key system
    lines.append("# ENTITY-KEYED FINDINGS")
    lines.append("# Each finding has an entity_key you MUST return in your response.")
    lines.append("# Do NOT invent or modify identity fields - use entity_key to reference findings.")
    lines.append("")

    for _, row in df.iterrows():
        direction = "▲" if row["team_value"] > row["league_avg"] else "▼"
        delta = row["team_value"] - row["league_avg"]

        # Build entity_key for this row
        roster_name = row.get("roster_full_name") if "roster_full_name" in row.index else None
        roster_team = row.get("roster_team") if "roster_team" in row.index else None
        roster_pos = row.get("roster_position") if "roster_position" in row.index else None

        roster_name = str(roster_name).strip() if pd.notna(roster_name) else ""
        roster_team = str(roster_team).strip().upper() if pd.notna(roster_team) else ""
        roster_pos = str(roster_pos).strip().upper() if pd.notna(roster_pos) else ""

        # Fallback to posteam
        if not roster_team and "posteam" in row.index and pd.notna(row.get("posteam")):
            roster_team = str(row["posteam"]).strip().upper()

        if roster_name:
            entity_key = f"{roster_name}|{roster_team}|{roster_pos}"
        else:
            team_col = row.get("team") if "team" in row.index else None
            if pd.notna(team_col):
                team_val = str(team_col).strip().upper()
                entity_key = f"|{team_val}|" if len(team_val) <= 4 else f"|UNKNOWN|"
            else:
                entity_key = "|UNKNOWN|"

        # Base line: entity_key | metric | dimension=value | stats
        base = (
            f"  [{entity_key}]\n"
            f"    metric={row['metric']} | "
            f"{row['dimension']}={row['dimension_value']} | "
            f"value={row['team_value']:.3f} vs league={row['league_avg']:.3f} "
            f"({direction}{abs(delta):.3f}) | z={row['std_devs_away']:.2f} | "
            f"n={int(row['sample_size'])}"
        )

        lines.append(base)

    return "\n".join(lines)


def build_prompt(
    findings_df: pd.DataFrame,
    memory_summary: str,
    registry: dict[str, dict] = None,
    n_insights: int = 10,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) ready for the API call."""
    findings_block = _format_findings_block(findings_df, registry)

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


# ── Entity Registry & Identity Lock ───────────────────────────────────────────

def build_entity_registry(df: pd.DataFrame) -> dict[str, dict]:
    """
    Build a canonical entity registry from the enriched findings DataFrame.

    Each entity is keyed by: roster_full_name|roster_team|roster_position
    For team-level findings (no player), key is: |team|

    Returns a dict mapping entity_key -> canonical row data.
    """
    registry = {}

    for idx, row in df.iterrows():
        # Determine if this is a player-level or team-level finding
        roster_name = row.get("roster_full_name") if "roster_full_name" in row.index else None
        roster_team = row.get("roster_team") if "roster_team" in row.index else None
        roster_pos = row.get("roster_position") if "roster_position" in row.index else None

        # Normalize values
        roster_name = str(roster_name).strip() if pd.notna(roster_name) else ""
        roster_team = str(roster_team).strip().upper() if pd.notna(roster_team) else ""
        roster_pos = str(roster_pos).strip().upper() if pd.notna(roster_pos) else ""

        # Fallback to posteam if roster_team missing
        if not roster_team and "posteam" in row.index and pd.notna(row.get("posteam")):
            roster_team = str(row["posteam"]).strip().upper()

        # For team-level findings (no player name)
        if not roster_name:
            team_col = row.get("team") if "team" in row.index else None
            if pd.notna(team_col):
                team_val = str(team_col).strip().upper()
                if len(team_val) <= 4:  # Looks like team abbrev
                    entity_key = f"|{team_val}|"
                else:
                    continue  # Skip rows we can't key
            else:
                continue
        else:
            entity_key = f"{roster_name}|{roster_team}|{roster_pos}"

        # Store canonical data for this entity
        canonical = {
            "entity_key": entity_key,
            "roster_full_name": roster_name if roster_name else None,
            "roster_team": roster_team if roster_team else None,
            "roster_position": roster_pos if roster_pos else None,
            "player_label": row.get("player_label") if "player_label" in row.index else None,
            "metric": row.get("metric"),
            "dimension": row.get("dimension"),
            "dimension_value": row.get("dimension_value"),
            "team_value": row.get("team_value"),
            "league_avg": row.get("league_avg"),
            "std_devs_away": row.get("std_devs_away"),
            "sample_size": row.get("sample_size"),
            "_row_idx": idx,
        }

        # Keep first occurrence if duplicate key
        if entity_key not in registry:
            registry[entity_key] = canonical

    return registry


def _normalize_entity_key(key: str) -> list[str]:
    """
    Generate normalized forms of an entity_key for fuzzy matching.
    Returns candidate keys to try in the registry.
    """
    if not key:
        return []

    candidates = [key]

    # Try lowercase version
    candidates.append(key.lower())

    # Parse and normalize components
    parts = key.split("|")
    if len(parts) == 3:
        name, team, pos = parts
        # Try with uppercase team/pos
        candidates.append(f"{name}|{team.upper()}|{pos.upper()}")
        # Try with stripped whitespace
        candidates.append(f"{name.strip()}|{team.strip()}|{pos.strip()}")

    return candidates


def apply_identity_lock(
    insights: list[dict],
    registry: dict[str, dict],
    verbose: bool = False,
) -> tuple[list[dict], list[dict], dict]:
    """
    Post-process insights to enforce canonical identity from the entity registry.

    For each insight:
    - Extract entity_key (or infer from evidence fields)
    - Look up in registry
    - Overwrite all identity fields from canonical data
    - Flag as identity_locked=True/False

    Returns:
        (valid_insights, quarantined_insights, stats_dict)

    stats_dict contains: {"locked": N, "quarantined": M, "overwritten": K}
    """
    valid = []
    quarantined = []
    stats = {"locked": 0, "quarantined": 0, "overwritten": 0}

    for insight in insights:
        entity_key = insight.get("entity_key")
        matched_entry = None

        # Try to find in registry
        if entity_key:
            for candidate in _normalize_entity_key(entity_key):
                if candidate in registry:
                    matched_entry = registry[candidate]
                    break

        # Fallback: try to match from evidence fields
        if not matched_entry:
            evidence = insight.get("evidence", {})
            if isinstance(evidence, dict):
                # Try to reconstruct entity_key from evidence
                ev_name = evidence.get("roster_full_name") or ""
                ev_team = evidence.get("roster_team") or evidence.get("team") or ""
                ev_pos = ""  # Position rarely in evidence

                if ev_name and ev_name != "null":
                    # Try exact match first
                    reconstructed = f"{ev_name}|{ev_team.upper()}|{ev_pos}"
                    for reg_key, reg_val in registry.items():
                        if reg_val.get("roster_full_name") == ev_name:
                            matched_entry = reg_val
                            break

                # Fallback for team-level findings
                if not matched_entry and not ev_name:
                    team_key = f"|{ev_team.upper()}|"
                    if team_key in registry:
                        matched_entry = registry[team_key]

        # Apply identity lock
        if matched_entry:
            # Track if we're overwriting anything
            old_teams = insight.get("teams", [])
            new_team = matched_entry.get("roster_team")

            if new_team:
                new_teams = [new_team]
                if old_teams and set(t.upper() for t in old_teams if t) != {new_team}:
                    stats["overwritten"] += 1
                    if verbose:
                        print(f"  IDENTITY LOCK: teams {old_teams} → {new_teams}")
            else:
                new_teams = old_teams  # Keep existing if no canonical team

            # Overwrite all identity fields from canonical registry
            insight["entity_key"] = matched_entry["entity_key"]
            insight["roster_full_name"] = matched_entry.get("roster_full_name")
            insight["roster_team"] = matched_entry.get("roster_team")
            insight["roster_position"] = matched_entry.get("roster_position")
            insight["teams"] = new_teams
            insight["canonical_metric"] = matched_entry.get("metric")
            insight["canonical_sample_size"] = matched_entry.get("sample_size")
            insight["identity_locked"] = True

            stats["locked"] += 1
            valid.append(insight)

        else:
            # Quarantine: entity_key not in registry
            insight["identity_locked"] = False
            insight["quarantine_reason"] = "entity_key not found in registry"
            stats["quarantined"] += 1

            if verbose:
                print(f"  QUARANTINED: entity_key={entity_key!r}, "
                      f"evidence={insight.get('evidence', {})}")

            quarantined.append(insight)

    return valid, quarantined, stats


def validate_insights(
    insights: list[dict],
    registry: dict[str, dict],
    verbose: bool = False,
) -> tuple[list[dict], list[str]]:
    """
    Final validation before report writing.

    Checks:
    - Every insight has a valid entity_key
    - Every rendered player/team/position matches canonical registry values

    Returns: (validated_insights, list of validation_errors)
    """
    validated = []
    errors = []

    for i, insight in enumerate(insights):
        entity_key = insight.get("entity_key")
        is_valid = True

        # Check 1: entity_key exists
        if not entity_key:
            errors.append(f"Insight #{i+1}: missing entity_key")
            is_valid = False
            continue

        # Check 2: entity_key in registry
        if entity_key not in registry:
            # Try normalized forms
            found = False
            for candidate in _normalize_entity_key(entity_key):
                if candidate in registry:
                    found = True
                    break
            if not found:
                errors.append(f"Insight #{i+1}: entity_key '{entity_key}' not in registry")
                is_valid = False
                continue

        # Check 3: identity fields match registry
        canonical = registry.get(entity_key)
        if canonical:
            for field in ["roster_full_name", "roster_team", "roster_position"]:
                insight_val = insight.get(field)
                canonical_val = canonical.get(field)
                if insight_val and canonical_val and insight_val != canonical_val:
                    errors.append(
                        f"Insight #{i+1}: {field} mismatch: "
                        f"'{insight_val}' vs canonical '{canonical_val}'"
                    )
                    # Auto-correct
                    insight[field] = canonical_val

        if is_valid:
            validated.append(insight)
        elif verbose:
            print(f"  VALIDATION FAILED: Insight #{i+1}")

    return validated, errors


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
    """Log each insight to the JSONL. Uses identity-locked fields from registry."""
    logged = 0
    for item in insights:
        # Use LOCKED identity fields (from canonical registry)
        team = item.get("roster_team") or (item.get("teams", [None])[0] if item.get("teams") else "MULTI")
        metrics = item.get("metrics", [])

        # Use interpretation field (new schema) or fall back to insight_text
        insight_text = item.get("interpretation") or item.get("insight_text", "")

        if not insight_text:
            continue

        log_from_finding_row(
            row = {
                "team":            team,
                "metric":          metrics[0] if metrics else "mixed",
                "dimension":       "entity_key",
                "dimension_value": item.get("entity_key", ""),
                "finding_type":    "interpreted",
                "team_value":      None,
                "league_avg":      None,
                "std_devs_away":   None,
                "sample_size":     item.get("canonical_sample_size"),
            },
            insight_text = insight_text,
            session_id   = session_id,
        )
        logged += 1

    return logged


# ── Display ────────────────────────────────────────────────────────────────────

def display_insights(insights: list[dict], quarantined: list[dict] = None) -> None:
    """Print insights to stdout in a clean, readable format.

    Uses identity-locked fields (roster_full_name, roster_team, roster_position)
    from the canonical registry, NOT Claude-authored values.
    """
    print(f"\n{'='*80}")
    print(f"INTERPRETED INSIGHTS  ({len(insights)} valid)")
    if quarantined:
        print(f"  ({len(quarantined)} quarantined - invalid entity_key)")
    print(f"{'='*80}\n")

    for item in sorted(insights, key=lambda x: x.get("rank", 99)):
        rank = item.get("rank", "?")

        # Use LOCKED identity fields from registry (NOT Claude-authored)
        player = item.get("roster_full_name") or ""
        team = item.get("roster_team") or ""
        position = item.get("roster_position") or ""

        # Build identity string from locked values
        if player:
            identity = f"{player}"
            if team:
                identity += f" ({team}"
                if position:
                    identity += f", {position}"
                identity += ")"
        elif team:
            identity = team
        else:
            identity = item.get("entity_key", "UNKNOWN")

        mets = ", ".join(item.get("metrics", []))

        # Use interpretation field (new schema) or fall back to insight_text
        text = item.get("interpretation") or item.get("insight_text", "")
        novelty = item.get("novelty_reason", "")

        # Header shows locked identity
        lock_status = "LOCKED" if item.get("identity_locked") else "UNLOCKED"
        print(f"#{rank}  [{identity}]  {mets}  ({lock_status})")
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
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed debug output (e.g., team lock operations)")
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

    # ── Build entity registry ─────────────────────────────────────────────────────
    entity_registry = build_entity_registry(filtered_df)
    print(f"  Entity registry built: {len(entity_registry)} canonical entities")

    # ── Build prompt ─────────────────────────────────────────────────────────────
    session_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    system_prompt, user_prompt = build_prompt(
        filtered_df, memory_summary, registry=entity_registry, n_insights=args.n_insights
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
        print("\n" + "=" * 80)
        print("ENTITY REGISTRY")
        print("=" * 80)
        for key, val in list(entity_registry.items())[:10]:
            print(f"  {key}")
            print(f"    → name={val.get('roster_full_name')}, team={val.get('roster_team')}, pos={val.get('roster_position')}")
        if len(entity_registry) > 10:
            print(f"  ... and {len(entity_registry) - 10} more")
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

    # ── Apply Identity Lock ───────────────────────────────────────────────────────
    # Merge insights with canonical entity registry - overwrites ALL identity fields
    valid_insights, quarantined, lock_stats = apply_identity_lock(
        insights, entity_registry, verbose=args.verbose
    )

    print(f"\n  IDENTITY LOCK: {lock_stats['locked']} locked, "
          f"{lock_stats['quarantined']} quarantined, "
          f"{lock_stats['overwritten']} had identity corrected")

    # ── Validate before report ────────────────────────────────────────────────────
    validated_insights, validation_errors = validate_insights(
        valid_insights, entity_registry, verbose=args.verbose
    )

    if validation_errors:
        print(f"\n  VALIDATION: {len(validation_errors)} error(s)")
        if args.verbose:
            for err in validation_errors:
                print(f"    - {err}")

    # ── Display & Log ─────────────────────────────────────────────────────────────
    display_insights(validated_insights, quarantined=quarantined)

    # Log quarantined insights separately if verbose
    if quarantined and args.verbose:
        print(f"\n{'='*80}")
        print(f"QUARANTINED INSIGHTS ({len(quarantined)} total)")
        print(f"{'='*80}")
        for q in quarantined:
            print(f"  entity_key: {q.get('entity_key')}")
            print(f"  reason: {q.get('quarantine_reason')}")
            print()

    n_logged = log_insights_from_response(validated_insights, session_id=session_id)
    print(f"\n{n_logged} insights logged to insights_memory/insights_log.jsonl")
    print(f"Session ID: {session_id}")


if __name__ == "__main__":
    main()
