import re
from datetime import datetime, date, timedelta, timezone
from typing import List, Tuple

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# =======================================================
# Page config
# =======================================================
st.set_page_config(page_title="Wordle Stats", page_icon="🟩", layout="wide")

# --- Global CSS (dark Wordle look) ---
CUSTOM_CSS = """
<style>
/* Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

:root {
  --bg: #0E1117;
  --bg-2: #1B1F2A;
  --text: #E3E6EC;
  --muted: rgba(227,230,236,0.75);
  --border: rgba(255,255,255,0.06);
  --accent: #6AAA64; /* Wordle green */
  --accent-2: #B59F3B; /* Wordle yellow */
  --accent-3: #3A3A3C; /* Wordle gray */
}

body { color: var(--text); }
.block-container { max-width: 1200px; padding-top: 1rem; }

/* Headings */
h1, h2, h3 { letter-spacing: 0.2px; }

/* Hero row */
.hero {
  display: flex; gap: 16px; align-items: center;
  background: var(--bg-2); border: 1px solid var(--border);
  border-radius: 16px; padding: 14px 16px; margin-bottom: 12px;
}
.hero .title { font-size: 1.25rem; font-weight: 600; }
.hero .subtitle { color: var(--muted); }

/* Success note */
.stAlert { border-radius: 12px; }

/* Buttons */
.stButton>button {
  border-radius: 10px;
  padding: 0.6rem 1rem;
  border: 1px solid var(--border);
  box-shadow: 0 2px 12px rgba(0,0,0,0.25);
}
.stButton>button:hover { transform: translateY(-1px); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-bottom: none;
  border-radius: 10px 10px 0 0;
  padding: 8px 14px;
}

/* Metric cards */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.04);
  border-radius: 14px;
  padding: 14px;
  border: 1px solid var(--border);
}

/* File uploader tweak */
[data-testid="stFileUploader"] {
  background: var(--bg-2);
  border: 1px dashed var(--border);
  border-radius: 12px; padding: 10px 12px;
}

/* Date row alignment */
.date-row { margin-top: 6px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Seaborn theme & palette ---
sns.set_theme(
    style="whitegrid",
    rc={"axes.edgecolor": "0.15", "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35},
)
WORDLE_PALETTE = ["#6AAA64", "#B59F3B", "#3A3A3C", "#538D4E", "#C9B458", "#787C7E"]
sns.set_palette(WORDLE_PALETTE)

# =======================================================
# Hero + build stamp + privacy note
# =======================================================
st.markdown(
    """
<div class="hero">
  <div>
    <div class="title">Wordle Stats</div>
    <div class="subtitle">Upload your WhatsApp export and explore individual & team performance.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption(f"Build: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

st.success(
    "Privacy: Your file is processed in memory for this session only. "
    "We don’t persist or store your data—nothing is kept between sessions."
)

# =======================================================
# Parsing / data helpers
# =======================================================
def parse_chat(text: str, date_locale: str = "dd/mm/yyyy") -> pd.DataFrame:
    text = (text.replace("\u202f", " ")
                .replace("\u200e", "")
                .replace("\u00a0", " "))

    fmt = "%d/%m/%Y" if date_locale == "dd/mm/yyyy" else "%m/%d/%Y"

    p1 = re.compile(
        r"\[(\d{1,2}/\d{1,2}/\d{4}),\s*\d{1,2}:\d{2}(?::\d{2})?\]\s+([^:]+?):\s+Wordle\s+[\d,]+\s+([1-6xX])/6"
    )
    p2 = re.compile(
        r"(\d{1,2}/\d{1,2}/\d{4}),\s*\d{1,2}:\d{2}(?:\s*[APMapm]{2})?\s*-\s*([^:]+?):\s+Wordle\s+[\d,]+\s+([1-6xX])/6"
    )

    rows = []
    for (date_str, name, score) in p1.findall(text) + p2.findall(text):
        try:
            d = datetime.strptime(date_str.strip(), fmt).date().isoformat()
        except ValueError:
            continue
        name = re.sub(r"^[^A-Za-z]*", "", name).strip()
        numeric = 7 if score.upper() == "X" else int(score)
        rows.append({"Date": d, "Name": name, "Score": numeric})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Date", "Name"]).reset_index(drop=True)
    return df


def build_individuals_wide(tidy_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    pivot = tidy_df.pivot_table(index="Date", columns="Name", values="Score", aggfunc="first")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    pivot = pivot.fillna(8)  # 8 = missed day
    return pivot, list(pivot.columns)


def filter_by_date_range(df_wide: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if df_wide.empty:
        return df_wide
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    return df_wide[(df_wide.index >= start_ts) & (df_wide.index <= end_ts)]

# =======================================================
# Plot helpers
# =======================================================
def compute_cumsum(df_wide: pd.DataFrame, player_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, str, float]:
    if df_wide.empty:
        return pd.DataFrame(), pd.Series(dtype=float), "—", float("nan")
    cumsum = df_wide[player_cols].cumsum()
    latest = cumsum.iloc[-1]
    leader_name = latest.idxmin()
    leader_value = float(latest.min())
    return cumsum, latest.sort_values(), leader_name, leader_value


def plot_cumsum(cumsum: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=cumsum, ax=ax, linewidth=1.8)
    ax.set_title("Overall Leader — Cumulative Score per Player Over Time (lower is better)")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Score")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Player", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    fig.tight_layout()
    return fig


def fig_all_time_player_average(df_wide: pd.DataFrame, player_cols: List[str]):
    if df_wide.empty:
        return None
    avgs = df_wide[player_cols].mean().sort_values()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=avgs.index, y=avgs.values, ax=ax)
    ax.set_title("All-time Player Average (includes 8s; lower is better)")
    ax.set_xlabel("Player"); ax.set_ylabel("Average Score")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def compute_rolling28(df_wide: pd.DataFrame, player_cols: List[str]) -> pd.DataFrame:
    if df_wide.empty:
        return pd.DataFrame()
    return df_wide[player_cols].rolling(window=28, min_periods=1).mean()


def plot_rolling28(rolling: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=rolling, ax=ax, linewidth=1.8)
    ax.set_title("Rolling 28-Day Player Average")
    ax.set_xlabel("Date"); ax.set_ylabel("Rolling Avg (28d)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Player", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    fig.tight_layout()
    return fig


def fig_score_distributions(df_wide: pd.DataFrame, player_cols: List[str]):
    if df_wide.empty:
        return None
    data = df_wide[player_cols].replace(8, pd.NA)
    n = len(player_cols)
    cols = min(4, n if n > 0 else 1)
    rows = (n + cols - 1) // cols if n else 1

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False, sharex=True, sharey=True)
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]

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


def fig_day_of_week_averages(df_wide: pd.DataFrame, player_cols: List[str]):
    if df_wide.empty:
        return None
    temp = df_wide.copy()
    temp["DayOfWeek"] = temp.index.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    temp["DayOfWeek"] = pd.Categorical(temp["DayOfWeek"], categories=order, ordered=True)
    long = temp[player_cols + ["DayOfWeek"]].melt(id_vars=["DayOfWeek"], var_name="Player", value_name="Score")
    grp = long.groupby(["DayOfWeek", "Player"], observed=True)["Score"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=grp, x="DayOfWeek", y="Score", hue="Player", ax=ax, marker="o")
    ax.set_title("Average Score by Day of Week (includes 8s; lower is better)")
    ax.set_xlabel("Day of Week"); ax.set_ylabel("Average Score")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Player", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    fig.tight_layout()
    return fig


def compute_weekly_winners(df_wide: pd.DataFrame, player_cols: List[str]):
    if df_wide.empty:
        return pd.Series(dtype=int), []
    weekly = df_wide[player_cols].resample("W-SUN").sum(min_count=1)
    if weekly.empty:
        return pd.Series(dtype=int), []
    max_date = df_wide.index.max().normalize()
    if weekly.index[-1] > max_date:
        weekly = weekly.iloc[:-1]
    if weekly.empty:
        return pd.Series(dtype=int), []
    weekly_min = weekly.min(axis=1)
    winners_per_week = []
    for idx, _ in weekly_min.items():
        winners = weekly.columns[(weekly.loc[idx] == weekly_min.loc[idx])].tolist()
        winners_per_week.append(winners)
    all_winners_flat = [p for winners in winners_per_week for p in winners]
    winners_count = pd.Series(all_winners_flat).value_counts().sort_values(ascending=False)
    last_full_week_winners = winners_per_week[-1] if winners_per_week else []
    return winners_count, last_full_week_winners

# =======================================================
# UI
# =======================================================
uploaded = st.file_uploader("Upload your WhatsApp export (_chat.txt)", type=["txt"])
date_format = st.radio("Date format in your chat", ["dd/mm/yyyy", "mm/dd/yyyy"], horizontal=True)

if uploaded:
    text = uploaded.read().decode("utf-8", errors="ignore")
    tidy = parse_chat(text, date_locale=date_format)

    if tidy.empty:
        st.warning("No Wordle lines found. Make sure your export contains lines like 'Wordle #### 3/6'.")
        st.stop()

    # Top metrics
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Days parsed", tidy['Date'].nunique())
    with m2: st.metric("Players", tidy['Name'].nunique())
    with m3: st.metric("Total entries", len(tidy))

    with st.expander("Preview parsed entries"):
        st.dataframe(tidy.head(50), use_container_width=True)

    # Build wide
    df_wide, player_cols = build_individuals_wide(tidy)

    # --- Date range controls (UK format + reset, aligned) ---
    min_d = df_wide.index.min().date()
    max_d = df_wide.index.max().date()

    if "date_range" not in st.session_state:
        st.session_state["date_range"] = (min_d, max_d)

    st.markdown("**Select date range for charts — calculate scores based on this period only**")
    col_date, col_reset = st.columns([4, 1], vertical_alignment="center")

    with col_date:
        dr_val = st.date_input(
            label="Date range",
            value=st.session_state["date_range"],
            min_value=min_d,
            max_value=max_d,
            key="date_input_widget",
            format="DD/MM/YYYY",
            label_visibility="collapsed",
        )
    with col_reset:
        st.write("")  # spacer for alignment
        if st.button("Reset", use_container_width=True, help="Reset to full available date range"):
            st.session_state["date_range"] = (min_d, max_d)
            st.experimental_rerun()

    def _normalize_date_input(val, fallback_start, fallback_end):
        if isinstance(val, (list, tuple)):
            if len(val) == 2:
                s, e = val
                s = s or fallback_start
                e = e or fallback_end
                return s, e
            if len(val) == 1:
                d = val[0] or fallback_start
                return d, d
            return fallback_start, fallback_end
        if val is None:
            return fallback_start, fallback_end
        return val, val

    start_d, end_d = _normalize_date_input(dr_val, min_d, max_d)
    if start_d > end_d:
        start_d, end_d = end_d, start_d

    st.session_state["date_range"] = (start_d, end_d)
    df_range = filter_by_date_range(df_wide, start_d, end_d)

    # -------- Individuals --------
    st.header("Individuals")

    tabs = st.tabs([
        "Overall leader 🏆",
        "All-time averages 📊",
        "Rolling 28-day 📈",
        "Score distributions 📦",
        "By weekday 📅",
        "Weekly winners 🥇",
    ])

    # Overall leader
    with tabs[0]:
        cumsum_full, latest_sorted, leader_name, leader_value = compute_cumsum(df_range, player_cols)
        if cumsum_full.empty:
            st.warning("No data in selected date range.")
        else:
            c1, c2 = st.columns(2)
            with c1: st.metric("Current leader (lowest total)", leader_name)
            with c2: st.metric("Best cumulative score", f"{leader_value:.0f}")

            others = latest_sorted.drop(labels=[leader_name]) if leader_name in latest_sorted.index else latest_sorted
            if not others.empty:
                lines = [f"- **{name}** — {int(val)}" for name, val in others.items()]
                st.markdown("<div style='font-size:0.9rem'>Other cumulative totals:</div>", unsafe_allow_html=True)
                st.markdown("\n".join(lines), unsafe_allow_html=True)

            b1, b2 = st.columns([1, 1])
            with b1: last30 = st.button("Last 30 days view")
            with b2: fullview = st.button("Full view")

            cumsum_plot = cumsum_full
            if last30 and not cumsum_full.empty:
                cutoff = cumsum_full.index.max() - timedelta(days=29)
                cumsum_plot = cumsum_full[cumsum_full.index >= cutoff]
            elif fullview:
                cumsum_plot = cumsum_full

            fig = plot_cumsum(cumsum_plot)
            st.pyplot(fig, clear_figure=True)

    # All-time averages
    with tabs[1]:
        fig = fig_all_time_player_average(df_range, player_cols)
        st.pyplot(fig, clear_figure=True)

    # Rolling 28-day
    with tabs[2]:
        rolling_full = compute_rolling28(df_range, player_cols)
        if rolling_full.empty:
            st.warning("No data in selected date range.")
        else:
            b1, b2 = st.columns([1, 1])
            with b1: last30 = st.button("Last 30 days view", key="roll30")
            with b2: fullview = st.button("Full view", key="rollfull")

            rolling_plot = rolling_full
            if last30 and not rolling_full.empty:
                cutoff = rolling_full.index.max() - timedelta(days=29)
                rolling_plot = rolling_full[rolling_full.index >= cutoff]
            elif fullview:
                rolling_plot = rolling_full

            fig = plot_rolling28(rolling_plot)
            st.pyplot(fig, clear_figure=True)

    # Distributions
    with tabs[3]:
        fig = fig_score_distributions(df_range, player_cols)
        st.pyplot(fig, clear_figure=True)

    # By weekday
    with tabs[4]:
        fig = fig_day_of_week_averages(df_range, player_cols)
        st.pyplot(fig, clear_figure=True)

    # Weekly winners
    with tabs[5]:
        winners_count, last_full_week_winners = compute_weekly_winners(df_range, player_cols)
        if winners_count.empty:
            st.warning("Not enough data to compute weekly winners in the selected date range.")
        else:
            st.metric("Most recent full-week winner(s)", ", ".join(last_full_week_winners))
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=winners_count.index, y=winners_count.values, ax=ax)
            ax.set_title("Weekly Winners (count of Monday→Sunday wins)")
            ax.set_xlabel("Player"); ax.set_ylabel("Weeks Won")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            st.pyplot(fig, clear_figure=True)

    # -------- Teams --------
    st.header("Teams")
    players = sorted(tidy["Name"].unique().tolist())
    cA, cB = st.columns(2)
    with cA:
        team_a_label = st.text_input("Team A name", value="Team A", key="teamA_label")
        team_a = st.multiselect(f"{team_a_label} members", players, key="teamA_members")
    with cB:
        team_b_label = st.text_input("Team B name", value="Team B", key="teamB_label")
        team_b = st.multiselect(f"{team_b_label} members", players, key="teamB_members")

    if team_a or team_b:
        a_cols = [c for c in team_a if c in df_range.columns]
        b_cols = [c for c in team_b if c in df_range.columns]

        df_teams = df_range.copy()
        df_teams[f"{team_a_label} Total"] = df_teams[a_cols].sum(axis=1) if a_cols else 0
        df_teams[f"{team_b_label} Total"] = df_teams[b_cols].sum(axis=1) if b_cols else 0
        df_teams[f"{team_a_label} Wins"] = (df_teams[f"{team_a_label} Total"] < df_teams[f"{team_b_label} Total"]).astype(int).cumsum()
        df_teams[f"{team_b_label} Wins"] = (df_teams[f"{team_b_label} Total"] < df_teams[f"{team_a_label} Total"]).astype(int).cumsum()

        t_tabs = st.tabs(["Team totals over time", "Cumulative wins"])
        with t_tabs[0]:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=df_teams[[f"{team_a_label} Total", f"{team_b_label} Total"]], ax=ax, linewidth=1.8)
            ax.set_title("Daily Team Totals")
            ax.set_xlabel("Date"); ax.set_ylabel("Total Score")
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig, clear_figure=True)

        with t_tabs[1]:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(
                data=df_teams[[f"{team_a_label} Wins", f"{team_b_label} Wins"]],
                drawstyle="steps-post",
                ax=ax,
                linewidth=1.8
            )
            ax.set_title("Cumulative Team Wins")
            ax.set_xlabel("Date"); ax.set_ylabel("Wins")
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig, clear_figure=True)
    else:
        st.info("Select members for Team A and/or Team B to see team charts.")

    # Scoring details
    st.markdown("### Scoring details")
    st.markdown(
        "- **X/6** counts as **7**.\n"
        "- If a player doesn’t post on a day, that day is recorded as **8**.\n"
        "- **Lower is better** across all charts.\n"
        "- **Overall leader** uses the **cumulative sum** over time.\n"
        "- **Rolling 28-day average** and other averages include 8s."
    )

else:
    st.info("Upload your `_chat.txt` export to begin.")
