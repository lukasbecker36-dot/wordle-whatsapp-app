import re
from datetime import datetime
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Streamlit page ----------
st.set_page_config(page_title="Wordle WhatsApp Analyzer", page_icon="🟩", layout="wide")
st.title("Wordle WhatsApp Analyzer 🟩🟨⬛")

# ---------- Parsing ----------
def parse_chat(text: str, date_locale: str = "dd/mm/yyyy") -> pd.DataFrame:
    """
    Parse WhatsApp export text and return a tidy DataFrame: Date, Name, Score.
    'X' is mapped to 7. Missing players later become 8 in the wide table.
    """
    # Clean WhatsApp special markers
    text = (text.replace("\u202f", " ")
                .replace("\u200e", "")
                .replace("\u00a0", " "))

    fmt = "%d/%m/%Y" if date_locale == "dd/mm/yyyy" else "%m/%d/%Y"

    # iOS style: [dd/mm/yyyy, 12:34:56] Name: Wordle #### 3/6
    p1 = re.compile(
        r"\[(\d{1,2}/\d{1,2}/\d{4}),\s*\d{1,2}:\d{2}(?::\d{2})?\]\s+([^:]+?):\s+Wordle\s+[\d,]+\s+([1-6xX])/6"
    )
    # Android style: dd/mm/yyyy, 12:34 - Name: Wordle #### 4/6
    p2 = re.compile(
        r"(\d{1,2}/\d{1,2}/\d{4}),\s*\d{1,2}:\d{2}(?:\s*[APMapm]{2})?\s*-\s*([^:]+?):\s+Wordle\s+[\d,]+\s+([1-6xX])/6"
    )

    rows = []
    for (date_str, name, score) in p1.findall(text) + p2.findall(text):
        # parse date
        try:
            date = datetime.strptime(date_str.strip(), fmt).date().isoformat()
        except ValueError:
            continue
        # normalise name: strip leading non-alpha, trim spaces
        name = re.sub(r"^[^A-Za-z]*", "", name).strip()
        numeric = 7 if score.upper() == "X" else int(score)
        rows.append({"Date": date, "Name": name, "Score": numeric})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Date", "Name"]).reset_index(drop=True)
    return df

def build_daily_frame(tidy_df: pd.DataFrame, team_a: List[str], team_b: List[str],
                      label_a: str, label_b: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a daily wide dataframe with one column per player, plus team totals and cumulative wins.
    Missing players are filled with 8.
    """
    if tidy_df.empty:
        return pd.DataFrame(), []

    # pivot: Date rows, Name columns, Score values
    pivot = tidy_df.pivot_table(index="Date", columns="Name", values="Score", aggfunc="first")
    # ensure dates are chronological
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()

    # Fill missing entries with 8 (treated as 'didn't play')
    pivot = pivot.fillna(8)

    player_cols = list(pivot.columns)

    # Some selected team members may not exist in the data; intersect to be safe
    team_a_cols = [c for c in team_a if c in player_cols]
    team_b_cols = [c for c in team_b if c in player_cols]

    # Compute team totals (if teams empty, totals will be 0)
    pivot[f"{label_a} Total"] = pivot[team_a_cols].sum(axis=1) if team_a_cols else 0
    pivot[f"{label_b} Total"] = pivot[team_b_cols].sum(axis=1) if team_b_cols else 0

    # Daily winners (True/False) -> cumulative
    a_better = (pivot[f"{label_a} Total"] < pivot[f"{label_b} Total"]).astype(int)
    b_better = (pivot[f"{label_b} Total"] < pivot[f"{label_a} Total"]).astype(int)
    pivot[f"{label_a} Wins"] = a_better.cumsum()
    pivot[f"{label_b} Wins"] = b_better.cumsum()

    return pivot, player_cols

# ---------- Plot helpers (return matplotlib Figure objects) ----------
def fig_team_totals(df: pd.DataFrame, label_a: str, label_b: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df[[f"{label_a} Total", f"{label_b} Total"]], ax=ax)
    ax.set_title("Daily Team Totals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Score")
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig

def fig_cumulative_wins(df: pd.DataFrame, label_a: str, label_b: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=df[[f"{label_a} Wins", f"{label_b} Wins"]],
        drawstyle="steps-post",
        ax=ax
    )
    ax.set_title("Cumulative Team Wins")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wins")
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig

def fig_player_averages(df: pd.DataFrame, player_cols: List[str]):
    # average including 8s (same convention as your script)
    avgs = df[player_cols].mean().sort_values()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=avgs.index, y=avgs.values, ax=ax)
    ax.set_title("Overall Average Score per Player (lower is better)")
    ax.set_xlabel("Player")
    ax.set_ylabel("Average Score")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    return fig

def fig_player_misses(df: pd.DataFrame, player_cols: List[str]):
    # count of 8s per player
    misses = (df[player_cols] == 8).sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=misses.index, y=misses.values, ax=ax)
    ax.set_title("Number of Misses (score of 8) per Player")
    ax.set_xlabel("Player")
    ax.set_ylabel("Misses")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    return fig

# ---------- UI ----------
uploaded = st.file_uploader("Upload your WhatsApp export (_chat.txt)", type=["txt"])
date_format = st.radio("Date format in your chat", ["dd/mm/yyyy", "mm/dd/yyyy"], horizontal=True)

if uploaded:
    text = uploaded.read().decode("utf-8", errors="ignore")
    tidy = parse_chat(text, date_locale=date_format)

    if tidy.empty:
        st.warning("No Wordle lines found. Make sure your export contains lines like 'Wordle #### 3/6'.")
        st.stop()

    st.success(f"Parsed {tidy['Date'].nunique()} days, {tidy['Name'].nunique()} players, {len(tidy)} entries.")
    with st.expander("Preview parsed entries"):
        st.dataframe(tidy.head(50), use_container_width=True)

    # Team selection
    players = sorted(tidy["Name"].unique().tolist())
    st.subheader("Teams")
    col1, col2 = st.columns(2)
    with col1:
        team_a_label = st.text_input("Team A name", value="Boys")
        team_a = st.multiselect(f"{team_a_label} members", players)
    with col2:
        team_b_label = st.text_input("Team B name", value="Girls")
        team_b = st.multiselect(f"{team_b_label} members", players)

    # Build frame + charts
    df_daily, player_cols = build_daily_frame(tidy, team_a, team_b, team_a_label, team_b_label)
    if df_daily.empty:
        st.stop()

    st.subheader("Charts")
    tabs = st.tabs(["Team totals", "Cumulative wins", "Player averages", "Misses (8s)"])

    with tabs[0]:
        st.pyplot(fig_team_totals(df_daily, team_a_label, team_b_label), clear_figure=True)
    with tabs[1]:
        st.pyplot(fig_cumulative_wins(df_daily, team_a_label, team_b_label), clear_figure=True)
    with tabs[2]:
        st.pyplot(fig_player_averages(df_daily, player_cols), clear_figure=True)
    with tabs[3]:
        st.pyplot(fig_player_misses(df_daily, player_cols), clear_figure=True)

    # Download CSV of the wide daily frame (includes team totals + wins)
    st.download_button(
        "Download daily CSV",
        df_daily.reset_index().rename(columns={"index": "Date"}).to_csv(index=False).encode("utf-8"),
        "wordle_scores_daily.csv",
        "text/csv"
    )
else:
    st.info("Upload your `_chat.txt` export to begin.")
