# NFL Insights Report — 2026-02-24

**Lens:** Team-Level Splits (`team-level`) — baseline  
**Description:** Baseline team-vs-league analysis across all pass/run plays. The broadest lens — covers all metrics × all dimensions.


> ⚠️  **DRY RUN** — Claude API was not called.

---

## Pipeline Status

| Step | Value |
|------|-------|
| Season | 2025 |
| New data | No |
| Plays analyzed | 34,628 |
| Findings generated | 50 |
| Sent to Claude | 30 (50 new · 0 related · 0 duplicate) |
| Insights generated | 0 |
| Memory loaded | 0 past insights |

---

## Top Statistical Findings

| # | Team | Metric | Dimension = Value | Team | League | z | n |
|---|------|--------|------------------|------|--------|---|---|
| 1 | DEN | `comp_yac_epa` | `pass_rushers`=`0 rushers` | ▲0.019 | 0.001 | 11.61 | 473 |
| 2 | LA | `pass_oe` | `defenders_in_box`=`8 in box` | ▲39.754 | -1.939 | 7.62 | 74 |
| 3 | CHI | `time_to_throw` | `pass_length`=`deep` | ▲3.859 | 3.248 | 7.14 | 160 |
| 4 | NYJ | `pass_oe` | `play_type`=`pass` | ▼20.441 | 28.011 | -8.09 | 551 |
| 5 | TEN | `time_to_throw` | `was_pressure`=`True` | ▲4.256 | 3.408 | 6.88 | 134 |
| 6 | TEN | `time_to_throw` | `pressure_label`=`Pressure` | ▲4.256 | 3.408 | 6.88 | 134 |
| 7 | CHI | `time_to_throw` | `down_str`=`Down 1` | ▲3.227 | 2.757 | 7.02 | 261 |
| 8 | NE | `pass_oe` | `play_type`=`run` | ▲-32.672 | -41.961 | 7.84 | 572 |
| 9 | CHI | `time_to_throw` | `season_type`=`REG` | ▲3.052 | 2.694 | 7.78 | 576 |
| 10 | CHI | `time_to_throw` | `qb_scramble_label`=`Designed` | ▲3.037 | 2.698 | 7.94 | 667 |
| 11 | CHI | `time_to_throw` | `play_type`=`pass` | ▲3.037 | 2.698 | 7.94 | 667 |
| 12 | CHI | `time_to_throw` | `pressure_label`=`Pressure` | ▲4.119 | 3.408 | 6.65 | 178 |
| 13 | CHI | `time_to_throw` | `was_pressure`=`True` | ▲4.119 | 3.408 | 6.65 | 178 |
| 14 | NYJ | `pass_oe` | `offense_formation`=`SHOTGUN` | ▼-5.352 | 6.240 | -7.97 | 695 |
| 15 | CHI | `time_to_throw` | `no_huddle_label`=`Huddle` | ▲3.049 | 2.704 | 7.75 | 623 |
| 16 | CHI | `time_to_throw` | `location`=`Home` | ▲3.037 | 2.702 | 7.83 | 667 |
| 17 | DEN | `pass_oe` | `pass_rushers`=`0 rushers` | ▲-40.818 | -47.540 | 7.28 | 472 |
| 18 | NYJ | `pass_oe` | `pass_length`=`short` | ▼21.310 | 28.798 | -7.02 | 419 |
| 19 | NYJ | `pass_oe` | `defense_man_zone_type`=`nan` | ▼-54.548 | -47.731 | -6.83 | 393 |
| 20 | CHI | `time_to_throw` | `defense_man_zone_type`=`ZONE_COVERAGE` | ▲3.088 | 2.729 | 7.09 | 494 |

---

## Insights

_(Dry run — Claude API was not called.)_

---

## Session

- **ID:** `20260224T033945-8e8811`
- **Generated:** 2026-02-24 03:40:11 UTC
- **Findings file:** `lens_findings/2026-02-24-team-level.csv`

*Pipeline: exploration_planner → nfldiscovery → interpret → notify*