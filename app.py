import re
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Streamlit page config ----------------
st.set_page_config(page_title="Wordle WhatsApp Analyzer", page_icon="🟩", layout="wide")
st.title("Wordle WhatsApp Analyzer 🟩🟨⬛")

# ---------------- Parsing ----------------
def parse_chat(text: str, date_locale: str = "dd/mm/yyyy") -> pd.DataFrame:
    """
    Parse WhatsApp export text and return a tidy DataFrame: Date, Name, Score.
    'X' -> 7. (We'll fill true missing entries with 8 later.)
    """
    text = (text.replace("\u202f", " ")
                .replace("\u200e", "")
                .replace("\u00a0", " "))

    fmt = "%d/%m/%Y" if date_locale == "dd/mm/yyyy" else "%m/%d/%Y"

    # iOS
    p1 = re.compile(
        r"\[(\d{1,2}/\d{1,2}/\d{4}),\s*\d{1,2}:\d{2}(?::\d{2})?\]\s+([^:]+?):\s+Wordle\s+[\d,]+\s+([1-6xX])/6"
    )
    # Android
    p2 = re.compile(
        r"(\d{1,2}/\d{1,2}/\d{4}),\s*\d{1,2}:\d{2}(?:\s*[APMapm]{2})?\s*-\s*([^:]+?):\s+Wordle\s+[\d,]+\s+([1-6xX])/6"
    )

    rows = []
    for (date_str, name, score) in p1.findall(text) + p2.findall(text):
        try:
            date = datetime.strptime(date_str.strip(), fmt).date().isoformat()
        except ValueError:
            continue
        name = re.sub(r"^[^A-Za-z]*", "", name).strip()
        numeric = 7 if score.upper() == "X" else int(score)
        rows.append({"Date": date, "Name": name, "Score": numeric})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Date", "Name"]).reset_index(drop=True)
    return df

def build_individuals_wide(tidy_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Pivot to Date x Player wide frame, fill missing with 8.
    """
    pivot = tidy_df.pivot_table(index="Date", columns="Name", values="Score", aggfunc="first")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    pivot = pivot.fillna(8)  # 8 = didn't play / missing
    return pivot, list(pivot.columns)

# ---------------- Plot helpers (matplotlib figures) ----------------
def fig_all_time_player_average(df_wide: pd.DataFrame, player_cols: List[str]):
    avgs = df_wide[player_cols].mean().sort_values()  # includes 8s
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=avgs.index, y=avgs.values, ax=ax)
    ax.set_title("All-time Player Average (lower is better)")
    ax.set_xlabel("Player")
    ax.set_ylabel("Average Score")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    return fig, avgs

def fig_rolling_28_day_average(df_wide: pd.DataFrame, player_cols: List[str]):
    # Rolling mean over the last 28 rows (days), includes 8s by design
    rolling = df_wide[player_cols].rolling(window=28, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=rolling[player_cols], ax=ax, linewidth=1.6)
    ax.set_title("Rolling 28-Day Player Average")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Avg (28d)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Player", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    fig.tight_layout()
    return fig

def fig_overall_leader(df_wide: pd.DataFrame, player_cols: List[str]):
    """
    Show cumulative average per player + highlight the leader over time.
    We'll plot only the "current best" (lowest cumulative avg) line to keep it readable,
    and display the all-time leader as a metric.
    """
    cumavg = df_wide[player_cols].expanding(min_periods=1).mean()
    # Per date, find the player with the lowest cum avg
    best_vals = cumavg.min(axis=1)                    # numeric best value per date
    best_player = cumavg.idxmin(axis=1)               # which player leads per date

    # All-time leader (lowest average over the full period)
    all_time_avgs = df_wide[player_cols].mean()
    all_time_leader = all_time_avgs.idxmin()
    all_time_value = all_time_avgs.min()

    # Plot the best cumulative average over time
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(best_vals.index, best_vals.values, linewidth=2)
    ax.set_title("Overall Leader Over Time (lowest cumulative average)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Best Cumulative Avg")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig, all_time_leader, float(all_time_value), best_player

def fig_score_distributions(df_wide: pd.DataFrame, player_cols: List[str]):
    """
    Histograms of scores per player, excluding 8s.
    """
    data = df_wide[player_cols].replace(8, pd.NA)

    # Compute grid size
    n = len(player_cols)
    cols = min(4, n if n > 0 else 1)
    rows = (n + cols - 1) // cols if n else 1

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False, sharex=True, sharey=True)
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]  # 1..7

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < n:
                player = player_cols[idx]
                series = data[player].dropna()
                if len(series):
                    ax.hist(series, bins=bins, edgecolor="black")
                ax.set_title(player, fontsize=10)
                ax.set_xticks(range(1, 8))
                ax.set_xlabel("Score", fontsize=9)
                ax.set_ylabel("Freq", fontsize=9)
            else:
                ax.axis("off")
            idx += 1

    fig.suptitle("Player Score Distributions (excluding 8s)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ---------------- UI: Upload + Individuals first ----------------
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

    # Build Individuals wide frame
    df_wide, player_cols = build_individuals_wide(tidy)

    # -------- Individuals Section --------
    st.header("Individuals")

    tabs = st.tabs([
        "All-time averages",
        "Rolling 28-day averages",
        "Overall leader",
        "Score distributions",
    ])

    with tabs[0]:
        fig1, avgs = fig_all_time_player_average(df_wide, player_cols)
        st.pyplot(fig1, clear_figure=True)

    with tabs[1]:
        st.pyplot(fig_rolling_28_day_average(df_wide, player_cols), clear_figure=True)

    with tabs[2]:
        fig3, leader_name, leader_value, leaders_over_time = fig_overall_leader(df_wide, player_cols)
        cols = st.columns(3)
        with cols[0]:
            st.metric("All-time Leader", leader_name)
        with cols[1]:
            st.metric("All-time Best Average", f"{leader_value:.2f}")
        with cols[2]:
            st.write("")  # spacer
        st.pyplot(fig3, clear_figure=True)

        # Optional: show a small table indicating who led most days
        leader_counts = leaders_over_time.value_counts().rename_axis("Player").to_frame("Days Leading")
        with st.expander("Who led most often over time?"):
            st.dataframe(leader_counts, use_container_width=True)

    with tabs[3]:
        st.pyplot(fig_score_distributions(df_wide, player_cols), clear_figure=True)

    # Download parsed data (tidy + wide)
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download tidy CSV (Date, Name, Score)",
            tidy.to_csv(index=False).encode("utf-8"),
            "wordle_tidy.csv",
            "text/csv"
        )
    with dl_col2:
        st.download_button(
            "Download daily wide CSV (players as columns)",
            df_wide.reset_index().rename(columns={"index": "Date"}).to_csv(index=False).encode("utf-8"),
            "wordle_daily_wide.csv",
            "text/csv"
        )

    # -------- Teams (optional dropdown) --------
    with st.expander("Team comparison (optional)"):
        players = sorted(tidy["Name"].unique().tolist())
        colA, colB = st.columns(2)
        with colA:
            team_a_label = st.text_input("Team A name", value="Boys")
            team_a = st.multiselect(f"{team_a_label} members", players, key="teamA")
        with colB:
            team_b_label = st.text_input("Team B name", value="Girls")
            team_b = st.multiselect(f"{team_b_label} members", players, key="teamB")

        # Only compute if user selects members
        if team_a or team_b:
            # Limit to selected members that actually exist
            a_cols = [c for c in team_a if c in df_wide.columns]
            b_cols = [c for c in team_b if c in df_wide.columns]

            df_teams = df_wide.copy()
            df_teams[f"{team_a_label} Total"] = df_teams[a_cols].sum(axis=1) if a_cols else 0
            df_teams[f"{team_b_label} Total"] = df_teams[b_cols].sum(axis=1) if b_cols else 0
            df_teams[f"{team_a_label} Wins"] = (df_teams[f"{team_a_label} Total"] < df_teams[f"{team_b_label} Total"]).astype(int).cumsum()
            df_teams[f"{team_b_label} Wins"] = (df_teams[f"{team_b_label} Total"] < df_teams[f"{team_a_label} Total"]).astype(int).cumsum()

            t_tabs = st.tabs(["Team totals over time", "Cumulative wins"])
            with t_tabs[0]:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=df_teams[[f"{team_a_label} Total", f"{team_b_label} Total"]], ax=ax)
                ax.set_title("Daily Team Totals")
                ax.set_xlabel("Date")
                ax.set_ylabel("Total Score")
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig, clear_figure=True)

            with t_tabs[1]:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(
                    data=df_teams[[f"{team_a_label} Wins", f"{team_b_label} Wins"]],
                    drawstyle="steps-post",
                    ax=ax
                )
                ax.set_title("Cumulative Team Wins")
                ax.set_xlabel("Date")
                ax.set_ylabel("Wins")
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig, clear_figure=True)

else:
    st.info("Upload your `_chat.txt` export to begin.")
