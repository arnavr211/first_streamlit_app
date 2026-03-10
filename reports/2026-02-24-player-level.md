# NFL Insights Report — 2026-02-24

**Lens:** Player Performance (`player-level`) — player  
**Description:** QB, rusher, and receiver splits — groups by individual player name instead of team. Surfaces outlier passers, runners, and targets.


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
| 1 | T.Thornton | `air_yards` | `receiver:shotgun_label`=`Shotgun` | ▲27.111 | 8.092 | 11.55 | 36 |
| 2 | A.Jeanty | `air_yards` | `receiver:pass_length`=`short` | ▼-2.084 | 4.278 | -9.75 | 71 |
| 3 | A.Pierce | `air_yards` | `receiver:pressure_label`=`Clean` | ▲19.552 | 7.956 | 9.51 | 67 |
| 4 | K.Gainwell | `xyac_epa` | `receiver:pressure_label`=`Clean` | ▲1.220 | 0.714 | 8.59 | 75 |
| 5 | K.Gainwell | `comp_air_epa` | `receiver:pressure_label`=`Clean` | ▼-0.889 | 0.132 | -8.54 | 75 |
| 6 | A.Pierce | `air_yards` | `receiver:shotgun_label`=`Shotgun` | ▲18.290 | 8.092 | 8.58 | 69 |
| 7 | K.Gainwell | `air_yards` | `receiver:pass_length`=`short` | ▼-0.489 | 4.278 | -8.23 | 90 |
| 8 | T.Spears | `air_yards` | `receiver:pass_length`=`short` | ▼-2.300 | 4.278 | -8.46 | 50 |
| 9 | K.Gainwell | `comp_air_epa` | `receiver:pass_length`=`short` | ▼-0.802 | -0.036 | -7.98 | 90 |
| 10 | K.Gainwell | `xyac_epa` | `receiver:shotgun_label`=`Shotgun` | ▲1.187 | 0.721 | 8.01 | 81 |
| 11 | T.Spears | `comp_air_epa` | `receiver:shotgun_label`=`Shotgun` | ▼-1.127 | 0.142 | -8.35 | 48 |
| 12 | K.Gainwell | `comp_air_epa` | `receiver:shotgun_label`=`Shotgun` | ▼-0.767 | 0.142 | -7.86 | 83 |
| 13 | B.Robinson | `yards_after_catch` | `receiver:pass_length`=`short` | ▲10.848 | 5.120 | 7.85 | 79 |
| 14 | T.Spears | `comp_air_epa` | `receiver:pass_length`=`short` | ▼-1.092 | -0.036 | -8.20 | 50 |
| 15 | J.Warren | `air_yards` | `receiver:pass_length`=`short` | ▼-2.370 | 4.278 | -8.20 | 46 |
| 16 | K.Gainwell | `air_yards` | `receiver:pressure_label`=`Clean` | ▼-1.000 | 7.956 | -7.77 | 75 |
| 17 | T.Etienne | `air_yards` | `receiver:pass_length`=`short` | ▼-1.589 | 4.278 | -7.99 | 56 |
| 18 | D.Achane | `air_yards` | `receiver:pass_length`=`short` | ▼-0.305 | 4.278 | -7.55 | 82 |
| 19 | K.Shakir | `xyac_epa` | `receiver:pressure_label`=`Clean` | ▲1.095 | 0.714 | 7.45 | 99 |
| 20 | R.White | `air_yards` | `receiver:pass_length`=`short` | ▼-2.267 | 4.278 | -7.99 | 45 |

---

## Insights

_(Dry run — Claude API was not called.)_

---

## Session

- **ID:** `20260224T033912-c18286`
- **Generated:** 2026-02-24 03:39:32 UTC
- **Findings file:** `lens_findings/2026-02-24-player-level.csv`

*Pipeline: exploration_planner → nfldiscovery → interpret → notify*