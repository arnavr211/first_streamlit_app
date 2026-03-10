"""
notify.py — Notification backends for the daily NFL insights pipeline.

Configure by creating notify_config.json in the project root, or via
environment variables. All channels fail silently so they never break
the main pipeline.

notify_config.json schema
--------------------------
{
    "channels": ["stdout", "file"],
    "webhook_url": null,
    "webhook_format": "slack",
    "log_file": "notifications.log",
    "desktop": false,
    "quiet": false
}

Supported channels
------------------
  stdout   — formatted console summary (default on, disable with "quiet": true)
  file     — one-liner appended to notifications.log (or custom log_file path)
  webhook  — HTTP POST body compatible with Slack, Discord, or generic JSON
             Set webhook_url or $NFL_NOTIFY_WEBHOOK env var.
  desktop  — system notification (macOS via osascript; Linux via notify-send)

Environment variable overrides
-------------------------------
  NFL_NOTIFY_CHANNELS   — comma-separated list, e.g. "stdout,webhook"
  NFL_NOTIFY_WEBHOOK    — webhook URL
  NFL_NOTIFY_QUIET      — "1" to suppress stdout
"""

import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).parent
CONFIG_FILE = ROOT / "notify_config.json"
LOG_FILE    = ROOT / "notifications.log"

# ── Config loading ──────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load notify_config.json, falling back to safe defaults."""
    defaults = {
        "channels":       ["stdout"],
        "webhook_url":    None,
        "webhook_format": "slack",
        "log_file":       str(LOG_FILE),
        "desktop":        False,
        "quiet":          False,
    }

    cfg = dict(defaults)

    # JSON config file
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                file_cfg = json.load(f)
            cfg.update(file_cfg)
        except (json.JSONDecodeError, OSError):
            pass

    # Environment variable overrides
    if os.environ.get("NFL_NOTIFY_CHANNELS"):
        cfg["channels"] = [c.strip() for c in
                           os.environ["NFL_NOTIFY_CHANNELS"].split(",")]
    if os.environ.get("NFL_NOTIFY_WEBHOOK"):
        cfg["webhook_url"] = os.environ["NFL_NOTIFY_WEBHOOK"]
        if "webhook" not in cfg["channels"]:
            cfg["channels"].append("webhook")
    if os.environ.get("NFL_NOTIFY_QUIET") == "1":
        cfg["quiet"] = True

    return cfg


# ── Message formatting ──────────────────────────────────────────────────────────

def _format_summary(report: dict) -> str:
    """
    Plain-text one-liner summary for file log and desktop notifications.
    report keys: date, lens, lens_name, n_findings, n_insights, n_memory,
                 report_path, top_insight, fresh_data, dry_run, session_id
    """
    prefix    = "[DRY RUN] " if report.get("dry_run") else ""
    fresh_tag = " [NEW DATA]" if report.get("fresh_data") else ""
    return (
        f"{prefix}NFL {report.get('date', '?')} | "
        f"Lens: {report.get('lens_name', report.get('lens', '?'))}{fresh_tag} | "
        f"{report.get('n_findings', 0)} findings → {report.get('n_insights', 0)} insights | "
        f"Memory: {report.get('n_memory', 0)} past | "
        f"Report: {report.get('report_path', 'N/A')}"
    )


def _format_stdout(report: dict) -> str:
    """Multi-line console banner."""
    lines = [
        "",
        "=" * 65,
        "  NFL DAILY INSIGHTS PIPELINE — COMPLETE",
        "=" * 65,
    ]
    if report.get("dry_run"):
        lines.append("  *** DRY RUN — no API call was made ***")
    if report.get("fresh_data"):
        lines.append(f"  *** NEW DATA detected (Week {report.get('latest_week', '?')}) ***")

    lines += [
        f"  Date:       {report.get('date', '?')}",
        f"  Lens:       {report.get('lens_name', '?')} [{report.get('lens', '?')}]",
        f"  Plays:      {report.get('n_plays', 'N/A'):,}" if isinstance(
            report.get('n_plays'), int) else f"  Plays:      {report.get('n_plays', 'N/A')}",
        f"  Findings:   {report.get('n_findings', 0)} total → {report.get('n_sent', 0)} sent to Claude",
        f"  Insights:   {report.get('n_insights', 0)} new  ({report.get('n_memory', 0)} in memory)",
        f"  Report:     {report.get('report_path', 'N/A')}",
        f"  Session:    {report.get('session_id', '?')}",
    ]

    if report.get("top_insight"):
        lines += [
            "",
            "  Top insight:",
            *[f"    {ln}" for ln in report["top_insight"][:200].split("\n")],
        ]

    lines.append("=" * 65)
    return "\n".join(lines)


def _format_slack(report: dict) -> dict:
    """Slack Block Kit payload."""
    fresh_tag = " :new:" if report.get("fresh_data") else ""
    dry_tag   = " _(dry run)_" if report.get("dry_run") else ""
    header    = f"NFL Insights — {report.get('date', '?')} · {report.get('lens_name', '?')}{fresh_tag}{dry_tag}"

    top = report.get("top_insight", "")
    top_block = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*Top insight:*\n{top[:600]}" if top else "_No insights generated._",
        },
    } if top else None

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": header}},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Findings:*\n{report.get('n_findings', 0)}"},
                {"type": "mrkdwn", "text": f"*Insights:*\n{report.get('n_insights', 0)} new"},
                {"type": "mrkdwn", "text": f"*Memory:*\n{report.get('n_memory', 0)} past"},
                {"type": "mrkdwn", "text": f"*Lens:*\n{report.get('lens', '?')}"},
            ],
        },
    ]
    if top_block:
        blocks.append(top_block)
    if report.get("report_path"):
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn",
                          "text": f"Report: `{report['report_path']}`"}],
        })

    return {"text": header, "blocks": blocks}


def _format_discord(report: dict) -> dict:
    """Discord webhook payload."""
    fresh = " 🆕" if report.get("fresh_data") else ""
    dry   = " (dry run)" if report.get("dry_run") else ""
    top   = report.get("top_insight", "")

    embed = {
        "title": f"NFL Insights — {report.get('date', '?')} · {report.get('lens_name', '?')}{fresh}{dry}",
        "color": 0x003594,
        "fields": [
            {"name": "Findings", "value": str(report.get("n_findings", 0)), "inline": True},
            {"name": "New Insights", "value": str(report.get("n_insights", 0)), "inline": True},
            {"name": "Memory", "value": f"{report.get('n_memory', 0)} past", "inline": True},
        ],
    }
    if top:
        embed["description"] = top[:500]

    return {"embeds": [embed]}


def _format_generic(report: dict) -> dict:
    """Generic JSON payload for custom webhooks."""
    return {
        "event":       "nfl_insights_daily",
        "date":        report.get("date"),
        "lens":        report.get("lens"),
        "lens_name":   report.get("lens_name"),
        "n_findings":  report.get("n_findings", 0),
        "n_insights":  report.get("n_insights", 0),
        "n_memory":    report.get("n_memory", 0),
        "fresh_data":  report.get("fresh_data", False),
        "dry_run":     report.get("dry_run", False),
        "report_path": report.get("report_path"),
        "top_insight": report.get("top_insight", ""),
        "session_id":  report.get("session_id"),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }


# ── Channel senders ─────────────────────────────────────────────────────────────

def _send_stdout(report: dict, cfg: dict) -> None:
    if not cfg.get("quiet"):
        print(_format_stdout(report))


def _send_file(report: dict, cfg: dict) -> None:
    try:
        log_path = Path(cfg.get("log_file", str(LOG_FILE)))
        with open(log_path, "a", encoding="utf-8") as f:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write(f"[{ts}] {_format_summary(report)}\n")
    except OSError as e:
        print(f"[notify] file channel error: {e}", file=sys.stderr)


def _send_webhook(report: dict, cfg: dict) -> None:
    url = cfg.get("webhook_url")
    if not url:
        print("[notify] webhook channel enabled but no webhook_url configured",
              file=sys.stderr)
        return

    fmt = cfg.get("webhook_format", "slack").lower()
    if fmt == "discord":
        payload = _format_discord(report)
    elif fmt == "generic":
        payload = _format_generic(report)
    else:
        payload = _format_slack(report)

    try:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            if status not in (200, 204):
                print(f"[notify] webhook returned HTTP {status}", file=sys.stderr)
    except (urllib.error.URLError, OSError) as e:
        print(f"[notify] webhook error: {e}", file=sys.stderr)


def _send_desktop(report: dict, cfg: dict) -> None:
    title   = "NFL Insights"
    message = _format_summary(report)[:200]
    try:
        if sys.platform == "darwin":
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{message}" with title "{title}"'],
                timeout=5, check=False, capture_output=True,
            )
        else:
            # Linux — requires libnotify-bin / notify-send
            subprocess.run(
                ["notify-send", title, message],
                timeout=5, check=False, capture_output=True,
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        print(f"[notify] desktop channel error: {e}", file=sys.stderr)


_SENDERS = {
    "stdout":  _send_stdout,
    "file":    _send_file,
    "webhook": _send_webhook,
    "desktop": _send_desktop,
}


# ── Public API ──────────────────────────────────────────────────────────────────

def send(report: dict) -> None:
    """
    Dispatch a pipeline completion notification to all configured channels.

    Parameters
    ----------
    report : dict with any subset of:
        date          str   "2025-01-15"
        lens          str   "goal-line"
        lens_name     str   "Goal Line"
        n_plays       int   47_823
        n_findings    int   50
        n_sent        int   30
        n_insights    int   8
        n_memory      int   47
        fresh_data    bool  True if new nflverse data was downloaded
        latest_week   int   14
        dry_run       bool  True if --dry-run was used
        report_path   str   "reports/2025-01-15-goal-line.md"
        top_insight   str   first sentence of top insight
        session_id    str   "20250115T143022-a3f2b1"
    """
    cfg      = _load_config()
    channels = cfg.get("channels", ["stdout"])

    for channel in channels:
        sender = _SENDERS.get(channel)
        if sender is None:
            print(f"[notify] unknown channel '{channel}'", file=sys.stderr)
            continue
        try:
            sender(report, cfg)
        except Exception as e:  # never break the pipeline
            print(f"[notify] channel '{channel}' raised {type(e).__name__}: {e}",
                  file=sys.stderr)


# ── CLI for testing ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick sanity test with a dummy report
    test_report = {
        "date":        "2025-01-15",
        "lens":        "goal-line",
        "lens_name":   "Goal Line",
        "n_plays":     1847,
        "n_findings":  50,
        "n_sent":      30,
        "n_insights":  8,
        "n_memory":    47,
        "fresh_data":  False,
        "dry_run":     False,
        "report_path": "reports/2025-01-15-goal-line.md",
        "top_insight": "KC's EPA/play at the goal line is +0.31 vs league average of -0.04, "
                       "making them the most efficient red-zone team in 2025.",
        "session_id":  "20250115T143022-a3f2b1",
    }
    send(test_report)
