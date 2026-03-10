"""
roster_validation.py — Load nflverse rosters and validate player-level findings.

Functions
---------
load_rosters(seasons)
    Load seasonal rosters via nfl_data_py, normalize columns to:
        roster_player_id, roster_full_name, roster_team, roster_position, roster_status

enrich_and_validate_players(findings_df, rosters_df, ...)
    Left-join findings to rosters on player ID, add validation flags, and return
    the cleaned DataFrame plus a summary dict.

CLI
---
    python roster_validation.py --season 2025
    python roster_validation.py --season 2025 --sample-findings lens_findings/player-level_2025-01-15.csv
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore")

# ── Column mappings (adapt to actual nflverse roster schema) ────────────────────

# Map source column names → normalized names.
# We try multiple candidate names for each field in case the schema changes.
_ROSTER_COLUMN_MAP = {
    "roster_player_id": ["player_id", "gsis_id", "nfl_id"],
    "roster_full_name": ["player_name", "full_name", "display_name", "name"],
    "roster_team":      ["team", "team_abbr", "club_code"],
    "roster_position":  ["position", "pos"],
    "roster_status":    ["status", "roster_status", "status_short"],
}


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first candidate column present in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_roster_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename roster columns to the standardized names:
        roster_player_id, roster_full_name, roster_team, roster_position, roster_status

    Handles schema variations gracefully: if a source column is missing, the
    normalized column is set to None.
    """
    out = pd.DataFrame(index=df.index)
    for norm_name, candidates in _ROSTER_COLUMN_MAP.items():
        src = _find_column(df, candidates)
        if src is not None:
            out[norm_name] = df[src]
        else:
            out[norm_name] = None
    return out


# ── load_rosters ────────────────────────────────────────────────────────────────

def load_rosters(seasons: list[int]) -> pd.DataFrame:
    """
    Load seasonal rosters from nflverse via nfl_data_py.

    Parameters
    ----------
    seasons : list[int]
        List of NFL seasons (e.g., [2025] or [2023, 2024, 2025]).

    Returns
    -------
    pd.DataFrame
        Columns: roster_player_id, roster_full_name, roster_team,
                 roster_position, roster_status
        One row per (player, season). Duplicates on player_id are possible if
        a player changed teams mid-season.
    """
    import nfl_data_py as nfl

    # Try import_seasonal_rosters first (preferred — one row per player per season)
    try:
        raw = nfl.import_seasonal_rosters(seasons)
    except Exception as e:
        # Fallback: try weekly rosters
        try:
            raw = nfl.import_weekly_rosters(seasons)
        except Exception:
            raise RuntimeError(
                f"Could not load rosters for seasons {seasons}. "
                f"Original error: {e}"
            ) from e

    if raw.empty:
        raise ValueError(f"Rosters for seasons {seasons} are empty.")

    # Normalize columns
    rosters = _normalize_roster_columns(raw)

    # Keep only rows with a valid player_id
    rosters = rosters[rosters["roster_player_id"].notna() &
                      (rosters["roster_player_id"].astype(str).str.len() > 0)]

    # Deduplicate: keep latest entry per player_id (in case of mid-season moves)
    # If a player changed teams, this keeps the most recent team.
    rosters = rosters.drop_duplicates(subset=["roster_player_id"], keep="last")

    return rosters.reset_index(drop=True)


# ── enrich_and_validate_players ─────────────────────────────────────────────────

def enrich_and_validate_players(
    findings_df: pd.DataFrame,
    rosters_df: pd.DataFrame,
    *,
    player_id_col: str,
    posteam_col: Optional[str] = None,
    drop_missing: bool = True,
    drop_mismatch: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Left-join findings to rosters on player ID and add validation flags.

    Parameters
    ----------
    findings_df : pd.DataFrame
        Player-level findings with at least the `player_id_col` column.
    rosters_df : pd.DataFrame
        Output of load_rosters() with normalized columns.
    player_id_col : str
        Name of the column in findings_df containing the player ID
        (will be joined to roster_player_id).
    posteam_col : str | None
        Name of the column in findings_df containing the offensive team.
        If provided, team_mismatch flag compares this to roster_team.
    drop_missing : bool
        If True (default), drop rows with missing_roster_match == True.
    drop_mismatch : bool
        If True, drop rows with team_mismatch == True. Default is False.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        clean_df : enriched findings with roster columns + flags applied
        summary  : dict with counts:
            total_rows, matched_rows, missing_rows, missing_rows_dropped,
            mismatch_rows, mismatch_rows_dropped
    """
    if findings_df.empty:
        return findings_df.copy(), {
            "total_rows": 0,
            "matched_rows": 0,
            "missing_rows": 0,
            "missing_rows_dropped": 0,
            "mismatch_rows": 0,
            "mismatch_rows_dropped": 0,
        }

    if player_id_col not in findings_df.columns:
        raise ValueError(f"player_id_col '{player_id_col}' not in findings_df")

    # Left join
    merged = findings_df.merge(
        rosters_df,
        left_on=player_id_col,
        right_on="roster_player_id",
        how="left",
    )

    total_rows = len(merged)

    # Flag: missing_roster_match
    merged["missing_roster_match"] = (
        merged["roster_full_name"].isna() |
        (merged["roster_full_name"].astype(str).str.strip() == "")
    )
    missing_rows = int(merged["missing_roster_match"].sum())

    # Flag: team_mismatch
    if posteam_col and posteam_col in merged.columns and "roster_team" in merged.columns:
        # Both must be non-null and different to count as mismatch
        has_both = (
            merged[posteam_col].notna() &
            merged["roster_team"].notna() &
            (merged[posteam_col].astype(str).str.strip() != "") &
            (merged["roster_team"].astype(str).str.strip() != "")
        )
        merged["team_mismatch"] = (
            has_both &
            (merged[posteam_col].astype(str).str.upper() !=
             merged["roster_team"].astype(str).str.upper())
        )
    else:
        merged["team_mismatch"] = False

    mismatch_rows = int(merged["team_mismatch"].sum())
    matched_rows  = total_rows - missing_rows

    # Apply drops
    missing_dropped  = 0
    mismatch_dropped = 0

    if drop_missing:
        before = len(merged)
        merged = merged[~merged["missing_roster_match"]]
        missing_dropped = before - len(merged)

    if drop_mismatch:
        before = len(merged)
        merged = merged[~merged["team_mismatch"]]
        mismatch_dropped = before - len(merged)

    summary = {
        "total_rows":           total_rows,
        "matched_rows":         matched_rows,
        "missing_rows":         missing_rows,
        "missing_rows_dropped": missing_dropped,
        "mismatch_rows":        mismatch_rows,
        "mismatch_rows_dropped": mismatch_dropped,
    }

    return merged.reset_index(drop=True), summary


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _print_roster_summary(rosters: pd.DataFrame) -> None:
    """Print a summary of the loaded rosters."""
    print(f"\n{'='*60}")
    print("ROSTER SUMMARY")
    print(f"{'='*60}")
    print(f"  Total players: {len(rosters):,}")
    print(f"  Unique teams:  {rosters['roster_team'].nunique()}")

    if "roster_position" in rosters.columns:
        pos_counts = rosters["roster_position"].value_counts().head(10)
        print(f"\n  Top positions:")
        for pos, n in pos_counts.items():
            print(f"    {pos}: {n}")

    if "roster_status" in rosters.columns:
        status_counts = rosters["roster_status"].value_counts()
        print(f"\n  Status breakdown:")
        for status, n in status_counts.items():
            print(f"    {status}: {n}")

    print(f"\n  Sample rows:")
    print(rosters.head(5).to_string(index=False))


def _print_validation_summary(summary: dict, merged: pd.DataFrame) -> None:
    """Print the validation summary."""
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows:              {summary['total_rows']}")
    print(f"  Matched to roster:       {summary['matched_rows']}")
    print(f"  Missing roster match:    {summary['missing_rows']}")
    print(f"    → Dropped:             {summary['missing_rows_dropped']}")
    print(f"  Team mismatch:           {summary['mismatch_rows']}")
    print(f"    → Dropped:             {summary['mismatch_rows_dropped']}")
    print(f"  Final rows:              {len(merged)}")

    if summary["mismatch_rows"] > 0:
        print(f"\n  Team mismatches (first 5):")
        mismatches = merged[merged["team_mismatch"]].head(5)
        cols = ["team", "posteam", "roster_team", "roster_full_name"]
        cols = [c for c in cols if c in mismatches.columns]
        if cols:
            print(mismatches[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load rosters and validate player-level findings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--season", type=int, default=2025,
                        help="Season to load rosters for (default: 2025)")
    parser.add_argument("--seasons", type=str, default=None,
                        help="Comma-separated seasons (overrides --season)")
    parser.add_argument("--sample-findings", type=str, default=None,
                        help="Path to a findings CSV to validate")
    parser.add_argument("--player-id-col", type=str, default="player_id",
                        help="Column name for player ID (default: player_id)")
    parser.add_argument("--posteam-col", type=str, default="posteam",
                        help="Column name for offensive team (default: posteam)")
    parser.add_argument("--keep-missing", action="store_true",
                        help="Keep rows with no roster match (default: drop them)")
    args = parser.parse_args()

    # Parse seasons
    if args.seasons:
        seasons = [int(s.strip()) for s in args.seasons.split(",")]
    else:
        seasons = [args.season]

    print(f"Loading rosters for seasons: {seasons}")

    try:
        rosters = load_rosters(seasons)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    _print_roster_summary(rosters)

    # If a sample findings file is provided, validate it
    if args.sample_findings:
        findings_path = Path(args.sample_findings)
        if not findings_path.exists():
            print(f"\nERROR: findings file not found: {findings_path}", file=sys.stderr)
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"VALIDATING: {findings_path.name}")
        print(f"{'='*60}")

        findings = pd.read_csv(findings_path)
        print(f"  Loaded {len(findings)} findings")
        print(f"  Columns: {list(findings.columns)}")

        # Check if player_id_col exists
        if args.player_id_col not in findings.columns:
            print(f"\n  WARNING: '{args.player_id_col}' not in findings columns.")
            print(f"  Available: {[c for c in findings.columns if 'player' in c.lower() or 'id' in c.lower()]}")
            # Try to find an alternative
            for alt in ["player_id", "passer_player_id", "rusher_player_id", "receiver_player_id"]:
                if alt in findings.columns:
                    args.player_id_col = alt
                    print(f"  Using '{alt}' instead.")
                    break
            else:
                print(f"\n  ERROR: No suitable player ID column found.")
                sys.exit(1)

        posteam_col = args.posteam_col if args.posteam_col in findings.columns else None

        merged, summary = enrich_and_validate_players(
            findings,
            rosters,
            player_id_col=args.player_id_col,
            posteam_col=posteam_col,
            drop_missing=not args.keep_missing,
        )

        _print_validation_summary(summary, merged)

        # Show sample of enriched rows
        print(f"\n  Sample enriched rows:")
        show_cols = [args.player_id_col, "team", "roster_full_name", "roster_team",
                     "roster_position", "missing_roster_match", "team_mismatch"]
        show_cols = [c for c in show_cols if c in merged.columns]
        print(merged[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
