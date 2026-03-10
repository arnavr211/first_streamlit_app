import streamlit as st
import nfl_data_py as nfl
import pandas as pd

st.title('2025 NFL Play-by-Play Data')
st.write('Data sourced from nflfastR via the nfl_data_py package.')

@st.cache_data
def load_pbp_data():
    return nfl.import_pbp_data([2025])

pbp = load_pbp_data()

st.header('Dataset Overview')
st.write(f'**{len(pbp):,}** plays across **{pbp["game_id"].nunique()}** games with **{len(pbp.columns)}** columns')

# Filters
st.header('Filter Plays')
teams = sorted(pbp['posteam'].dropna().unique())
selected_teams = st.multiselect('Select team(s) on offense:', teams)

play_types = sorted(pbp['play_type'].dropna().unique())
selected_play_types = st.multiselect('Select play type(s):', play_types, default=['pass', 'run'])

weeks = sorted(pbp['week'].dropna().unique())
selected_weeks = st.multiselect('Select week(s):', weeks)

filtered = pbp.copy()
if selected_teams:
    filtered = filtered[filtered['posteam'].isin(selected_teams)]
if selected_play_types:
    filtered = filtered[filtered['play_type'].isin(selected_play_types)]
if selected_weeks:
    filtered = filtered[filtered['week'].isin(selected_weeks)]

st.write(f'Showing **{len(filtered):,}** plays after filters.')

display_cols = ['game_id', 'week', 'posteam', 'defteam', 'play_type', 'desc',
                'yards_gained', 'epa', 'down', 'ydstogo', 'passer_player_name',
                'rusher_player_name', 'receiver_player_name']
st.dataframe(filtered[display_cols].reset_index(drop=True))


