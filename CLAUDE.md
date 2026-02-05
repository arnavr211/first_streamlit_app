# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
streamlit run streamlit_app.py
```

## Dependencies

Python packages (no requirements.txt yet — install manually):
- `streamlit`
- `nfl_data_py` (Python interface to nflfastR/nflverse data)
- `pandas`

## Architecture

Single-file Streamlit app (`streamlit_app.py`) that loads 2025 NFL play-by-play data from nflverse via `nfl_data_py.import_pbp_data([2025])`. The dataset contains ~48k plays with 372 columns.

Key details:
- Data is cached with `@st.cache_data` to avoid re-downloading on each rerun
- Interactive filters: team on offense, play type, and week
- Displays a subset of columns (game_id, week, teams, play_type, desc, yards_gained, epa, down, ydstogo, player names)
- The nfl_data_py package pulls parquet files from the nflverse-data GitHub releases
