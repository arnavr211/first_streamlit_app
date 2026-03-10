# NFL_ANALYSIS_AGENT.md

This document defines how an analysis agent should understand, query, and present findings from the 2025 nflfastR play-by-play dataset.

---

## 1. Dataset Overview

**Source:** `nfl_data_py.import_pbp_data([2025])`
**Size:** ~48,578 rows × 372 columns
**Coverage:** Weeks 1–21 (regular season + playoffs), all 32 teams, 284 games
**Load pattern:** Always filter to `play_type.isin(["pass", "run"])` before EPA analysis. Non-play rows (kickoffs, punts, no_play, quarter_end markers) will dilute averages and skew results.

```python
import nfl_data_py as nfl
pbp = nfl.import_pbp_data([2025])
plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
```

---

## 2. Column Reference

Columns are grouped by analytical value. **Completeness** is the % of rows with non-null values.

### 2.1 Game & Play Identity (100% complete)
| Column | Description |
|--------|-------------|
| `game_id` | Unique game identifier: `2025_WK_AWAY_HOME` |
| `week` | Week number (1–18 regular season, 19–22 postseason) |
| `season_type` | `REG` or `POST` |
| `home_team` / `away_team` | Team abbreviations |
| `posteam` | Team with possession (offense) |
| `defteam` | Team on defense |
| `desc` | Full text description of the play |
| `play_type` | `pass`, `run`, `kickoff`, `punt`, `field_goal`, `extra_point`, `no_play`, `qb_kneel`, `qb_spike` |
| `div_game` | 1 if divisional matchup, 0 otherwise |
| `roof` | Stadium roof type: `outdoors`, `dome`, `retractable` |
| `surface` | Field surface: `grass`, `turf`, etc. |
| `spread_line` | Vegas point spread |
| `total_line` | Vegas over/under |

### 2.2 Situational Context (84–100% complete)
| Column | Completeness | Description |
|--------|-------------|-------------|
| `qtr` | 100% | Quarter (1–5, 5 = OT) |
| `down` | 84% | Down number (1–4). Null on non-scrimmage plays |
| `ydstogo` | 100% | Yards needed for first down |
| `yardline_100` | 93% | Yards from own end zone (1–99) |
| `score_differential` | 94% | `posteam_score` minus `defteam_score` |
| `game_seconds_remaining` | 100% | Seconds left in game |
| `quarter_seconds_remaining` | 100% | Seconds left in quarter |
| `posteam_timeouts_remaining` | 94% | Offensive team's timeouts |
| `defteam_timeouts_remaining` | 94% | Defensive team's timeouts |
| `shotgun` | 100% | 1 if QB in shotgun |
| `no_huddle` | 100% | 1 if no-huddle play |
| `goal_to_go` | 100% | 1 if defense is inside the 10 |
| `xpass` | 76% | Expected pass rate given situation (0–1) |
| `pass_oe` | 74% | Pass rate over expected (`pass` flag minus `xpass`) |

### 2.3 ⭐ Expected Points (EPA) — Highest Analytical Value
EPA measures how much a play changed the expected point total for the offensive team. League average on pass/run plays is approximately 0.0 by construction.

| Column | Completeness | Description |
|--------|-------------|-------------|
| `ep` | 99% | Expected points **before** the play |
| `epa` | 99% | Expected points **added** by the play |
| `qb_epa` | 99% | EPA credited to the QB (sacks count against QB, not the run game) |
| `air_epa` | 37% | EPA attributable to where the ball was thrown (air yards portion) |
| `yac_epa` | 37% | EPA attributable to yards after catch |
| `comp_air_epa` | 97% | Expected air EPA on all pass attempts (completed or not) |
| `comp_yac_epa` | 97% | Expected YAC EPA |

**Reference values:**
- Elite QB season: >0.20 EPA/play
- Average QB: 0.05–0.15 EPA/play
- Below replacement: <0.00 EPA/play
- A single play >2.0 EPA is a big gain; >4.0 EPA is a near-game-changing play

### 2.4 ⭐ Win Probability (WPA) — Highest Analytical Value
| Column | Completeness | Description |
|--------|-------------|-------------|
| `wp` | 99% | Win probability for `posteam` **before** the play (0–1) |
| `home_wp` | 100% | Win probability for home team (consistent throughout game) |
| `wpa` | 98% | Win probability **added** by the play (can be negative) |
| `vegas_wp` | 99% | Win probability adjusted for Vegas spread |
| `vegas_wpa` | 98% | WPA based on Vegas-adjusted model |
| `def_wp` | 99% | Win probability for the defensive team |

**Note:** Use `home_wp` (not `wp`) for cross-play comparisons within a game — `wp` flips perspective when possession changes.

### 2.5 ⭐ Completion Probability & Passing Efficiency
| Column | Completeness | Description |
|--------|-------------|-------------|
| `cp` | 36% | Model-estimated completion probability for this throw |
| `cpoe` | 36% | Completion probability over expected (`complete_pass` minus `cp`) |
| `air_yards` | 38% | Yards the ball traveled in the air past the LOS (can be negative for screens) |
| `yards_after_catch` | 24% | YAC (only populated on completions) |
| `pass_length` | 37% | `short` (≤15 air yards) or `deep` (>15 air yards) |
| `pass_location` | 37% | `left`, `middle`, or `right` |
| `time_to_throw` | 40% | QB's time from snap to release (seconds) |
| `was_pressure` | 93% | Whether QB faced pressure on the play |

### 2.6 ⭐ Expected YAC (xYAC)
| Column | Completeness | Description |
|--------|-------------|-------------|
| `xyac_epa` | 33% | Expected EPA from YAC given catch location |
| `xyac_mean_yardage` | 33% | Expected YAC yards |
| `xyac_median_yardage` | 33% | Median expected YAC |
| `xyac_success` | 33% | Probability the YAC results in a "success" play |
| `xyac_fd` | 33% | Probability YAC results in a first down |

### 2.7 Play Result Flags (97% complete)
Binary (0/1) columns for: `complete_pass`, `incomplete_pass`, `interception`, `sack`, `touchdown`, `pass_touchdown`, `rush_touchdown`, `fumble`, `fumble_lost`, `qb_hit`, `tackled_for_loss`, `penalty`, `first_down_pass`, `first_down_rush`, `third_down_converted`, `third_down_failed`, `fourth_down_converted`, `fourth_down_failed`, `success`

**`success`** = 1 if the play gained ≥45% of yards needed on 1st down, ≥60% on 2nd, or converted on 3rd/4th.

### 2.8 Player Names (variable completeness)
| Column | Completeness | Notes |
|--------|-------------|-------|
| `passer_player_name` | 41% | Only on pass plays |
| `rusher_player_name` | 31% | Only on run plays |
| `receiver_player_name` | 36% | Only on completions and some incompletions |
| `fantasy_player_name` | 68% | Primary ball-carrier or passer for fantasy |

Player names are formatted as `F.Lastname` (e.g., `P.Mahomes`, `J.Love`).

### 2.9 Formation & Personnel (93% complete)
| Column | Description |
|--------|-------------|
| `offense_formation` | Backfield formation: `SHOTGUN`, `UNDER CENTER`, `PISTOL`, etc. |
| `offense_personnel` | Personnel grouping: e.g., `1 RB, 1 TE, 3 WR` |
| `defenders_in_box` | Number of defenders in the box pre-snap |
| `defense_personnel` | Defensive personnel: e.g., `4 DL, 2 LB, 5 DB` |
| `number_of_pass_rushers` | Pass rushers sent on the play |
| `defense_man_zone_type` | `MAN_COVERAGE`, `ZONE_COVERAGE`, or `UNKNOWN` |
| `defense_coverage_type` | Specific coverage shell (Cover 1, Cover 2, Cover 3, Cover 4, etc.) |
| `route` | Route run by the primary receiver |

### 2.10 Drive-Level Aggregates (99% complete)
`drive_play_count`, `drive_time_of_possession`, `drive_first_downs`, `drive_inside20`, `drive_ended_with_score`, `drive_start_transition`, `drive_end_transition`

### 2.11 Low-Value / Sparse Columns to Avoid
These are rarely populated and add noise to most analyses:
- All `lateral_*` columns (0% populated)
- `ngs_air_yards` (0%)
- `two_point_conv_result`, `two_point_attempt` (<1%)
- `safety_player_name/id`, `blocked_player_*`, `own_kickoff_recovery_*` (<1%)
- `st_play_type` (0%)

---

## 3. Analysis Patterns

Run these patterns in order of expected surprisingness. For each, compute the metric, compare to a baseline, and flag anything more than 1.5 standard deviations from the mean.

### Pattern 1: EPA Outliers by Team × Situation
Find teams whose EPA changes dramatically across situations the league treats as equivalent.

```python
# Key splits to check:
# - Early downs (1st & 2nd) vs late downs (3rd & 4th)
# - Score differential buckets: blowout (|diff| > 14), close (|diff| <= 7)
# - Field position: own half (yardline_100 > 50) vs opponent half
# - Shotgun vs under center
# - First half vs second half
```

Flag when a team's EPA in a split is >1.5 SD from the league mean for that split.

### Pattern 2: Pass Rate Over Expected (PROE) Extremes
`pass_oe = pass - xpass` (74% populated). Teams and QBs who deviate most from situationally expected pass rates, and whether deviation correlates with EPA.

- Positive `pass_oe` means passing more than expected
- Check if high-PROE teams have higher or lower EPA (reveals scheme efficiency vs. stubbornness)
- Flag teams where PROE > +0.15 or < -0.15 consistently

### Pattern 3: Third & Fourth Down Efficiency by Distance Bucket
```
3rd & short (1–3): conversion rate + EPA vs league
3rd & medium (4–7): conversion rate + EPA vs league
3rd & long (8+): conversion rate + EPA vs league
4th down go-for-it decisions: EPA vs expected punt/FG value
```
Teams that over-index on runs on 3rd & long or under-convert 3rd & short relative to league are flagging real tendencies.

### Pattern 4: Pressure Impact on Passing EPA
Using `was_pressure` (93% populated):
- EPA on pressured vs clean dropbacks by QB
- Which QBs maintain EPA under pressure (resilient) vs collapse?
- Which defenses generate the most pressure and translate it to negative EPA?

Baseline: League EPA on clean dropbacks vs pressured dropbacks is typically a ~0.4–0.6 EPA swing.

### Pattern 5: cpoe vs. Outcome — Identifying Luck vs. Skill
`cpoe` (completion % over expected) separates QB accuracy from receiver/situation effects.

- High cpoe + high EPA = genuinely elite QB
- High cpoe + low EPA = accurate but in a bad scheme or bad supporting cast
- Low cpoe + high EPA = likely benefiting from YAC, play design, or hot streak
- Flag QBs whose EPA and cpoe rank differ by more than 5 positions

### Pattern 6: Air Yards vs. YAC Decomposition
`air_epa` vs. `yac_epa` splits tell you whether a passing game is scheme/QB-driven (air) or skill-position driven (YAC).

- Receivers with high `yards_after_catch` but low `air_yards` are RAC (run after catch) dependent
- Offensive systems with high `yac_epa` and low `air_epa` are short/dink-and-dunk schemes living on YAC
- Flag any team where >60% of passing EPA comes from YAC

### Pattern 7: Coverage Type Exploitation
Using `defense_coverage_type` and `defense_man_zone_type`:
- Which offenses have the largest EPA swing between man and zone coverage?
- Which QBs have the highest EPA against Cover 2 specifically? Against Cover 3?
- Which defenses allow the worst EPA per coverage shell (signals scheme weakness)

### Pattern 8: Clutch vs. Non-Clutch Performance Split
Define clutch as: `qtr == 4` (or OT), `game_seconds_remaining <= 300`, `abs(score_differential) <= 8`.

- Compare every QB's EPA/play in clutch vs. non-clutch situations
- Same for team-level run/pass success rate
- Flag splits >0.15 EPA difference between clutch and non-clutch (positive or negative)

### Pattern 9: Play-Calling Tendency Deviations
Identify coordinators whose play calling in specific situations deviates from league norms:

```
- Run rate on 1st & 10 by field position (league ~55% run)
- Pass rate on 2nd & short (league ~40% pass)
- 4th down go-for-it rate vs. league by distance
- Play-action usage rate (filter desc for "play action" or use qb_dropback + rush context)
- Personnel groupings on passing downs (12 personnel on 3rd & long = tendency to flag)
```

### Pattern 10: Sample Size Sensitivity Flags
Before reporting any split analysis, always check n. Apply these thresholds:

| Split Type | Minimum n for Reporting | Confidence Note |
|------------|------------------------|-----------------|
| QB EPA/play | 150 plays | Reliable above 200 |
| Team situation EPA | 30 plays | Flag if <50 |
| Individual coverage-type splits | 20 plays | Treat as directional only |
| Deep pass zone splits | 15 plays | Very noisy, note explicitly |
| 4th down decisions | 10 plays | Directional only |

**When n < minimum:** Report the number, label the finding as "small sample (n=X), directional only," and do not rank it above findings with adequate sample sizes.

Rule of thumb: EPA standard deviation is approximately 1.5 per play, so a meaningful signal (0.1 EPA/play difference) requires roughly 450 plays for statistical significance at p < 0.05.

### Pattern 11: Win Probability Leverage & Clutch Plays
Highest-impact plays by WPA reveal which players/teams delivered value when it mattered most.

- Top 10 plays by WPA for any team or player subset
- QB WPA in high-leverage situations (wp between 0.2 and 0.8, late game)
- Which games had the most WPA swing on a single play (game entropy via binary entropy of `home_wp`)

### Pattern 12: Defensive Coverage Pressure & EPA Allowed
`number_of_pass_rushers` and `defenders_in_box` vs. EPA allowed:
- Teams that blitz frequently — do they generate enough pressure to justify it?
- Blitz rate above 35% pass rushers with negative EPA allowed = effective; above 35% with positive EPA allowed = exploitable
- Identify QBs who generate the most EPA against heavy boxes (good vs. stacked boxes)

---

## 4. Presenting Findings

### 4.1 Ranking by Surprisingness
Always present findings in this order:
1. **Most surprising** first — define surprising as: deviation from league mean × statistical confidence. A 30% deviation with n=500 is more surprising than a 50% deviation with n=20.
2. **Lead with the number**, then the context. Never bury the key stat.
3. **Include the baseline** for every finding. A number without context is meaningless.

**Example of good presentation:**
> "Sam Darnold completed 61.9% of deep right throws (n=42) vs. 38.5% league average — the largest gap of any QB with 20+ attempts. He also had zero interceptions on those attempts."

**Example of bad presentation:**
> "Sam Darnold was very good at throwing deep right."

### 4.2 Required Elements per Finding
Every reported finding must include:
- The metric and its value
- The comparison baseline (league average, position average, or prior split)
- Sample size (n=X)
- Whether the sample is adequate or directional only
- One concrete play example if WPA or EPA is extreme

### 4.3 Output Format
Structure all analysis outputs as:

```
## [Finding Title] — Surprisingness: [High / Medium / Low]

**Metric:** [value] vs. [baseline] ([n=X] plays)
**Direction:** [Better / Worse / Mixed]

[2–3 sentences of interpretation]

| Supporting breakdown table |
|----------------------------|

**Best example play:** [desc snippet with WPA/EPA]

**Caveat:** [any sample size or context warnings]
```

### 4.4 Common Pitfalls to Flag
- **Garbage time inflation:** Plays with `score_differential` > 17 or < -17 in the 4th quarter skew EPA. Consider filtering for competitive situations.
- **Opponent quality:** EPA doesn't adjust for defensive strength. A team with +0.25 EPA/play against weak opponents may not replicate that vs. elite defenses.
- **Positional context for rushing:** QBs who scramble have artificially high rush EPA; separate designed runs from scrambles using `qb_scramble == 1`.
- **YAC conflation:** High `yards_after_catch` may reflect receiver ability, not QB accuracy. Don't credit QB for YAC without checking `cpoe`.
- **Small sample deep pass splits:** Deep passes are only ~19% of pass attempts. Any zone (e.g., deep middle) has very few observations — always report n.

### 4.5 Conciseness Rules
- **No finding longer than 150 words** of prose
- **Tables preferred** over lists of numbers in prose
- **One finding per section** — don't combine "Team A is good at X and also Y and also Z" into one paragraph
- **Rank no more than 10 items** in any leaderboard; cut at 5 if the bottom entries are unremarkable
- **Bold the single most important number** in every finding

---

## 5. Quick Reference: Most Analytically Valuable Columns

| Priority | Column(s) | Use Case |
|----------|-----------|----------|
| ⭐⭐⭐ | `epa`, `qb_epa` | Primary performance metric for any player/team analysis |
| ⭐⭐⭐ | `wpa`, `home_wp` | Game importance weighting and clutch analysis |
| ⭐⭐⭐ | `cpoe`, `cp` | QB accuracy adjusted for throw difficulty |
| ⭐⭐ | `air_yards`, `pass_length`, `pass_location` | Targeting scheme analysis |
| ⭐⭐ | `xpass`, `pass_oe` | Play-calling tendency vs. expectation |
| ⭐⭐ | `was_pressure`, `time_to_throw` | Protection and QB pocket performance |
| ⭐⭐ | `defense_coverage_type`, `defense_man_zone_type` | Scheme exploitation analysis |
| ⭐⭐ | `yards_after_catch`, `xyac_epa` | Skill position vs. scheme separation |
| ⭐ | `shotgun`, `no_huddle`, `offense_formation` | Situational tendency flags |
| ⭐ | `defenders_in_box`, `number_of_pass_rushers` | Defensive alignment analysis |
| ⭐ | `success` | Binary efficiency — useful for sample-stable trend analysis |
| ⭐ | `spread_line`, `total_line` | Vegas-adjusted context for game script analysis |
