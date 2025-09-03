import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import streamlit as st

# ----------------------------
# STREAMLIT INTERFACE
# ----------------------------
st.set_page_config(page_title="Previsione Tiri", layout="wide")

st.title("⚽ Previsione Tiri Squadre")
st.markdown("Seleziona le squadre e visualizza le stime dei tiri previste.")

# ----------------------------
# CARICAMENTO CSV
# ----------------------------
csv_folder = "./"
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

if not csv_files:
    st.error("⚠️ Nessun file CSV trovato nella cartella.")
    st.stop()

df_list = [pd.read_csv(file) for file in csv_files]
df_raw = pd.concat(df_list, ignore_index=True)

# ----------------------------
# CREAZIONE DATASET SQUADRA VS AVVERSARIO
# ----------------------------
rows = []
for idx, row in df_raw.iterrows():
    if pd.isna(row.get("HomeTeam")) or pd.isna(row.get("AwayTeam")):
        continue
    rows.append({'team': row['HomeTeam'].title(),
                 'opponent': row['AwayTeam'].title(),
                 'shots_for': row['HS'],
                 'shots_against': row['AS'],
                 'date': row.get('Date'),
                 'home': 1})
    rows.append({'team': row['AwayTeam'].title(),
                 'opponent': row['HomeTeam'].title(),
                 'shots_for': row['AS'],
                 'shots_against': row['HS'],
                 'date': row.get('Date'),
                 'home': 0})

df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# ----------------------------
# CALCOLO FORZA OFF/DIF
# ----------------------------
teams = df['team'].unique()
team_stats = []

for team in teams:
    df_team = df[df['team']==team].sort_values('date', ascending=False)
    df_last5 = df_team.head(5)
    pesi = np.linspace(1, 0.2, len(df_last5)) if not df_last5.empty else [1]

    # Ultime 5 totali
    off_last5_total = np.average(df_last5['shots_for'], weights=pesi) if not df_last5.empty else 0
    def_last5_total = np.average(df_last5['shots_against'], weights=pesi) if not df_last5.empty else 0

    # Casa/trasferta ultime 5
    df_last5_home = df_last5[df_last5['home']==1]
    df_last5_away = df_last5[df_last5['home']==0]
    off_last5_home = np.average(df_last5_home['shots_for'], weights=np.linspace(1,0.2,len(df_last5_home))) if not df_last5_home.empty else 0
    def_last5_home = np.average(df_last5_home['shots_against'], weights=np.linspace(1,0.2,len(df_last5_home))) if not df_last5_home.empty else 0
    off_last5_away = np.average(df_last5_away['shots_for'], weights=np.linspace(1,0.2,len(df_last5_away))) if not df_last5_away.empty else 0
    def_last5_away = np.average(df_last5_away['shots_against'], weights=np.linspace(1,0.2,len(df_last5_away))) if not df_last5_away.empty else 0

    # Media totale
    off_total = df_team['shots_for'].mean() if not df_team.empty else 0
    def_total = df_team['shots_against'].mean() if not df_team.empty else 0
    df_home = df_team[df_team['home']==1]
    df_away = df_team[df_team['home']==0]
    off_total_home = df_home['shots_for'].mean() if not df_home.empty else 0
    def_total_home = df_home['shots_against'].mean() if not df_home.empty else 0
    off_total_away = df_away['shots_for'].mean() if not df_away.empty else 0
    def_total_away = df_away['shots_against'].mean() if not df_away.empty else 0

    team_stats.append({
        'team': team,
        'off_last5_home': round(off_last5_home,2),
        'def_last5_home': round(def_last5_home,2),
        'off_last5_away': round(off_last5_away,2),
        'def_last5_away': round(def_last5_away,2),
        'off_last5_total': round(off_last5_total,2),
        'def_last5_total': round(def_last5_total,2),
        'off_total_home': round(off_total_home,2),
        'def_total_home': round(def_total_home,2),
        'off_total_away': round(off_total_away,2),
        'def_total_away': round(def_total_away,2),
        'off_total': round(off_total,2),
        'def_total': round(def_total,2)
    })

df_teams = pd.DataFrame(team_stats)

# ----------------------------
# SELEZIONE SQUADRE STREAMLIT
# ----------------------------
team_home = st.selectbox("Seleziona squadra di casa", df_teams['team'].values)
team_away = st.selectbox("Seleziona squadra in trasferta", df_teams['team'].values)

# ----------------------------
# FUNZIONI PREVISIONE
# ----------------------------
def predict_shots(team_home, team_away, mode='last5', home_away=True):
    t_home = df_teams[df_teams['team']==team_home].iloc[0]
    t_away = df_teams[df_teams['team']==team_away].iloc[0]

    if mode=='last5':
        if home_away:
            t1_off = t_home['off_last5_home']
            t1_def = t_home['def_last5_home']
            t2_off = t_away['off_last5_away']
            t2_def = t_away['def_last5_away']
        else:
            t1_off = t_home['off_last5_total']
            t1_def = t_home['def_last5_total']
            t2_off = t_away['off_last5_total']
            t2_def = t_away['def_last5_total']
    else:
        if home_away:
            t1_off = t_home['off_total_home']
            t1_def = t_home['def_total_home']
            t2_off = t_away['off_total_away']
            t2_def = t_away['def_total_away']
        else:
            t1_off = t_home['off_total']
            t1_def = t_home['def_total']
            t2_off = t_away['off_total']
            t2_def = t_away['def_total']

    shots_home = (t1_off + t2_def)/2
    shots_away = (t2_off + t1_def)/2
    return shots_home, shots_away

def predict_shots_combined(team1, team2):
    t1_off5 = df_teams[df_teams['team']==team1]['off_last5_total'].values[0]
    t1_def5 = df_teams[df_teams['team']==team1]['def_last5_total'].values[0]
    t2_off5 = df_teams[df_teams['team']==team2]['off_last5_total'].values[0]
    t2_def5 = df_teams[df_teams['team']==team2]['def_last5_total'].values[0]

    t1_off_home = df_teams[df_teams['team']==team1]['off_total_home'].values[0]
    t1_def_home = df_teams[df_teams['team']==team1]['def_total_home'].values[0]
    t2_off_away = df_teams[df_teams['team']==team2]['off_total_away'].values[0]
    t2_def_away = df_teams[df_teams['team']==team2]['def_total_away'].values[0]

    last5_home = (t1_off5 + t2_def5)/2
    last5_away = (t2_off5 + t1_def5)/2
    total_home = (t1_off_home + t2_def_away)/2
    total_away = (t2_off_away + t1_def_home)/2

    pred_home = 0.6*last5_home + 0.4*total_home
    pred_away = 0.6*last5_away + 0.4*total_away
    return pred_home, pred_away

# ----------------------------
# CALCOLO PREVISIONI
# ----------------------------
s_last5_ht, s_last5_at = predict_shots(team_home, team_away, mode='last5', home_away=True)
s_total_total, s_total_total_2 = predict_shots(team_home, team_away, mode='total', home_away=False)
s_comb_home, s_comb_away = predict_shots_combined(team_home, team_away)

# ----------------------------
# STAMPA RISULTATI
# ----------------------------
st.subheader("📊 Risultati previsione tiri")
st.write(f"**Ultime 5 partite (casa/trasferta):** {team_home}: {s_last5_ht:.2f}, {team_away}: {s_last5_at:.2f}, Totale: {s_last5_ht+s_last5_at:.2f}")
st.write(f"**Media storica (totale):** {team_home}: {s_total_total:.2f}, {team_away}: {s_total_total_2:.2f}, Totale: {s_total_total+s_total_total_2:.2f}")
st.write(f"**Combinata 60/40:** {team_home}: {s_comb_home:.2f}, {team_away}: {s_comb_away:.2f}, Totale: {s_comb_home+s_comb_away:.2f}")

# ----------------------------
# GRAFICO STREAMLIT
# ----------------------------
import matplotlib
matplotlib.use("Agg")

fig, ax = plt.subplots(figsize=(10,6))
width = 0.25
x = np.arange(2)

bars1 = ax.bar(x - width, [s_last5_ht, s_last5_at], width, label='Ultime 5 (casa/trasferta)', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, [s_total_total, s_total_total_2], width, label='Media storica', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, [s_comb_home, s_comb_away], width, label='Combinata 60/40', color='#2ca02c', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels([team_home, team_away], fontsize=12, fontweight='bold')
ax.set_ylabel("Tiri stimati", fontsize=12)
ax.set_title(f"Confronto previsioni tiri: {team_home} vs {team_away}", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0,5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

st.pyplot(fig)
