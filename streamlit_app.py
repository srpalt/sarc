import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import data_processing as dp

st.set_page_config(layout="wide", page_title="Student Alumni Relations Cell Dashboard", page_icon="📊")

# =========================================================================
# GLOBAL STYLE — validated dark palette (status + categorical), from the
# dataviz skill's reference palette. Status colors carry severity (good/
# warning/critical); the violet is reserved for "external stakeholder"
# identity so it's never confused with a severity signal.
# =========================================================================

GOOD_COLOR = "#1fbf3f"       # above-average / high standing
WARN_COLOR = "#fab219"       # medium standing
CRITICAL_COLOR = "#e0524f"   # below-average / low standing
STAKE_COLOR = "#9085e9"      # external stakeholders — categorical, not severity
NEUTRAL_COLOR = "#8b93a7"
ABOVE_COLOR = GOOD_COLOR
BELOW_COLOR = CRITICAL_COLOR

INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
SURFACE_1 = "#171a20"
BORDER = "rgba(255,255,255,0.09)"

st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-variant-numeric: tabular-nums; }}

h1, h2, h3 {{ letter-spacing: -0.01em; }}

hr {{ margin: 1.6rem 0 !important; opacity: 0.15; }}

div[data-testid="stMetric"] {{
    background: {SURFACE_1};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 12px 14px 8px 14px;
}}
div[data-testid="stMetricLabel"] {{ opacity: 0.7; font-size: 0.8rem; }}

div[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}

div[data-testid="stExpander"] {{
    border-radius: 12px !important;
    border: 1px solid {BORDER} !important;
}}

/* ---- KPI stat tiles ---- */
.kpi-strip {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 6px; }}
.kpi-tile {{
    flex: 1 1 200px;
    background: {SURFACE_1};
    border: 1px solid {BORDER};
    border-left: 3px solid var(--accent, {NEUTRAL_COLOR});
    border-radius: 12px;
    padding: 14px 18px;
}}
.kpi-label {{ font-size: 0.78rem; color: {INK_MUTED}; margin-bottom: 6px; }}
.kpi-value {{ font-size: 1.7rem; font-weight: 700; color: {INK_PRIMARY}; line-height: 1.15; }}
.kpi-sub {{ font-size: 0.8rem; color: {INK_SECONDARY}; margin-top: 5px; }}

/* ---- Insight cards, in a responsive grid ---- */
.insight-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; margin-bottom: 12px; }}
.insight-card {{
    background: {SURFACE_1};
    border: 1px solid {BORDER};
    border-left: 4px solid var(--accent, {NEUTRAL_COLOR});
    border-radius: 12px;
    padding: 14px 16px;
    line-height: 1.55;
    font-size: 0.94rem;
    color: {INK_SECONDARY};
}}
.insight-card.full {{ grid-column: 1 / -1; }}
.insight-card b {{ color: {INK_PRIMARY}; }}
.insight-card .head {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
.insight-card .icon {{ font-size: 1.05rem; }}
.insight-card .tag {{
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--accent, {NEUTRAL_COLOR});
}}

/* ---- Name pills (status/identity, always paired with the name as text) ---- */
.name-pill {{
    display: inline-block;
    padding: 1px 10px 2px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.93em;
    color: #0b0d13;
    white-space: nowrap;
}}

/* ---- Custom leaderboard (meter rows) ---- */
.lb-row {{
    display: flex; align-items: center; gap: 14px;
    background: {SURFACE_1}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 9px 16px; margin-bottom: 7px;
}}
.lb-rank {{ width: 28px; color: {INK_MUTED}; font-weight: 700; font-size: 0.82rem; text-align: center; flex-shrink: 0; white-space: nowrap; }}
.lb-name {{ width: 128px; flex-shrink: 0; }}
.lb-meter {{ flex: 1 1 160px; min-width: 100px; }}
.lb-track {{ height: 9px; border-radius: 6px; position: relative; overflow: hidden; }}
.lb-fill {{ height: 100%; border-radius: 6px; }}
.lb-score {{ width: 42px; text-align: right; font-weight: 700; color: {INK_PRIMARY}; font-variant-numeric: tabular-nums; flex-shrink: 0; }}
.lb-chips {{ display: flex; gap: 5px; flex-wrap: wrap; width: 300px; flex-shrink: 0; }}
.lb-chip {{ font-size: 0.7rem; padding: 2px 7px; border-radius: 999px; background: rgba(255,255,255,0.055); color: {INK_SECONDARY}; white-space: nowrap; }}
.lb-badge {{ width: 108px; text-align: right; font-size: 0.75rem; font-weight: 700; flex-shrink: 0; }}
.lb-header {{ display: flex; gap: 14px; padding: 0 16px; margin-bottom: 6px; font-size: 0.72rem; color: {INK_MUTED}; text-transform: uppercase; letter-spacing: 0.04em; }}

.section-eyebrow {{
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
    font-weight: 700;
    color: {INK_MUTED};
    margin-bottom: 2px;
}}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# LOAD + CACHE
# =========================================================================

@st.cache_data(show_spinner="Parsing WhatsApp exports...")
def get_data():
    df = dp.load_messages(".")
    tag_resp = dp.compute_tag_responses(df)
    bc_resp = dp.compute_broadcast_responses(df)
    sh_resp = dp.compute_stakeholder_responses(df)
    ext_eng = dp.compute_external_engagement(df)
    return df, tag_resp, bc_resp, sh_resp, ext_eng

df, tag_resp, bc_resp, sh_resp, ext_eng = get_data()
team_df = df[df["person_type"] == "team"].copy()

TEAM = dp.TEAM_ORDER
STAKEHOLDERS = dp.STAKEHOLDER_ORDER
TODAY = df["datetime"].max()
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIME_BUCKET_ORDER = ["Morning (5-12)", "Afternoon (12-17)", "Evening (17-22)", "Late Night (22-5)"]

# =========================================================================
# HELPERS
# =========================================================================

def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    rng = s.max() - s.min()
    return s * 0 if rng == 0 else (s - s.min()) / rng

def rate_by(resp_df, key_col):
    """member -> response rate (0-1), 0 if no data."""
    if resp_df.empty:
        return pd.Series(0.0, index=TEAM)
    g = resp_df.groupby(key_col)["responded"].mean()
    return g.reindex(TEAM).fillna(0.0)

def sorted_order(s: pd.Series, ascending=False):
    return s.sort_values(ascending=ascending).index.tolist()

def category(score):
    if score >= 65:
        return "🟢 High"
    elif score >= 40:
        return "🟡 Medium"
    return "🔴 Low"

def category_color(score):
    if score >= 65:
        return GOOD_COLOR
    elif score >= 40:
        return WARN_COLOR
    return CRITICAL_COLOR

CHAT_OPTIONS = ["internal", "chair", "manager", "office"]

# =========================================================================
# MASTER METRICS TABLE
# =========================================================================

internal_scoped = team_df[team_df.chat == "internal"]
external_scoped = team_df[team_df.chat != "internal"]

msg_vol = internal_scoped.groupby("person").size().reindex(TEAM, fill_value=0)
word_vol = internal_scoped.groupby("person")["word_count"].sum().reindex(TEAM, fill_value=0)
external_msgs = external_scoped.groupby("person").size().reindex(TEAM, fill_value=0)
active_days = internal_scoped.groupby("person")["date"].nunique().reindex(TEAM, fill_value=0)
total_days = (internal_scoped["date"].max() - internal_scoped["date"].min()).days + 1
active_day_pct = (active_days / total_days * 100).round(1)

daily_counts = internal_scoped.groupby(["person", "date"]).size().reset_index(name="count")
cons_stats = daily_counts.groupby("person")["count"].agg(["mean", "std"]).reindex(TEAM).fillna(0)
consistency_raw = cons_stats["mean"] / (1 + cons_stats["std"])
consistency_score = (consistency_raw / consistency_raw.max() * 100).round(1) if consistency_raw.max() > 0 else consistency_raw

# Ranking is INTERNAL-GROUP-ONLY by design: the internal chat is where the actual
# cell work happens (tasks, coordination, day-to-day). External chats (chair/
# manager/office) are stakeholder-facing and easy to be visible in without doing
# the underlying work, so external activity is deliberately excluded from the
# score and instead surfaced separately as a credit-chasing check.
bc_resp_internal = bc_resp[bc_resp.chat == "internal"] if not bc_resp.empty else bc_resp
tag_resp_internal = tag_resp[tag_resp.chat == "internal"] if not tag_resp.empty else tag_resp

bc_rate = rate_by(bc_resp_internal, "member")
tag_rate = rate_by(tag_resp_internal[tag_resp_internal.mentioned.isin(TEAM)] if not tag_resp_internal.empty else tag_resp_internal, "mentioned")
stake_rate = rate_by(sh_resp, "member")  # external — context only, not scored

last_active = team_df.groupby("person")["datetime"].max().reindex(TEAM)
days_since = (TODAY - last_active).dt.days

components = pd.DataFrame({
    "Msg Volume": minmax(msg_vol),
    "Word Depth": minmax(word_vol),
    "Active Days %": minmax(active_day_pct),
    "Consistency": minmax(consistency_score),
    "@all Responsiveness": minmax(bc_rate),
    "Tag Responsiveness": minmax(tag_rate),
}, index=TEAM)

WEIGHTS = {
    "Msg Volume": 0.30, "Word Depth": 0.15, "Active Days %": 0.15,
    "Consistency": 0.15, "@all Responsiveness": 0.15, "Tag Responsiveness": 0.10,
}
score = (sum(components[c] * w for c, w in WEIGHTS.items()) * 100)

master = pd.DataFrame({
    "Member": TEAM,
    "Score": score.round(1).values,
    "Internal Messages": msg_vol.values,
    "Total Words": word_vol.values,
    "Active Days %": active_day_pct.values,
    "Consistency": consistency_score.round(1).values,
    "@all Response %": (bc_rate * 100).round(1).values,
    "Tag Response %": (tag_rate * 100).round(1).values,
    "Days Since Last Active": days_since.reindex(TEAM).values,
}).sort_values("Score", ascending=False).reset_index(drop=True)
master.insert(0, "Rank", range(1, len(master) + 1))
master["Category"] = master["Score"].apply(category)
MEAN_SCORE = round(master["Score"].mean(), 1)
master["Vs. Average"] = (master["Score"] - MEAN_SCORE).round(1)
master["Standing"] = np.where(master["Score"] >= MEAN_SCORE, "Above Average", "Below Average")

# External activity — context only, NOT part of the ranking. Used purely to
# check for "high external visibility, low internal contribution" patterns.
external_rank = external_msgs.rank(ascending=False, method="min").reindex(TEAM)
internal_rank = master.set_index("Member")["Rank"].reindex(TEAM)
external_view = pd.DataFrame({
    "Member": TEAM,
    "Internal Rank": internal_rank.values.astype(int),
    "External Messages": external_msgs.values,
    "External Rank": external_rank.values.astype(int),
    "Stakeholder Response %": (stake_rate * 100).round(1).values,
})
external_view["Clout Gap"] = external_view["Internal Rank"] - external_view["External Rank"]
external_view = external_view.sort_values("Clout Gap", ascending=False).reset_index(drop=True)

# =========================================================================
# NAME HIGHLIGHTING — consistent color per person, used everywhere text
# mentions a name: blue = above team average, red = below, purple = stakeholder.
# =========================================================================

score_by_member = master.set_index("Member")["Score"]
NAME_COLORS = {m: category_color(score_by_member[m]) for m in TEAM}
NAME_COLORS.update({s: STAKE_COLOR for s in STAKEHOLDERS})

_NAME_PATTERN = re.compile(
    "|".join(re.escape(n) for n in sorted(NAME_COLORS, key=len, reverse=True))
)

def hl(text: str) -> str:
    """Wrap every known member/stakeholder name in a colored pill span."""
    def _sub(m):
        name = m.group(0)
        color = NAME_COLORS[name]
        return f'<span class="name-pill" style="background:{color}">{name}</span>'
    return _NAME_PATTERN.sub(_sub, text)

def card_html(text: str, tag: str = "FLAG", accent: str = BELOW_COLOR, icon: str = "⚠️", full: bool = False) -> str:
    return (
        f'<div class="insight-card{" full" if full else ""}" style="--accent:{accent}">'
        f'<div class="head"><span class="icon">{icon}</span><span class="tag">{tag}</span></div>'
        f'{hl(text)}</div>'
    )

def kpi_html(label: str, value: str, sub: str = "", accent: str = NEUTRAL_COLOR) -> str:
    return (
        f'<div class="kpi-tile" style="--accent:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{hl(value)}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )

def lb_row_html(rank: int, row: pd.Series) -> str:
    color = category_color(row["Score"])
    track = f"{color}26"
    width = max(row["Score"], 2)
    delta = row["Vs. Average"]
    delta_color = GOOD_COLOR if delta >= 0 else CRITICAL_COLOR
    chips = "".join(
        f'<span class="lb-chip">{label} {val}</span>'
        for label, val in [
            ("Active", f'{row["Active Days %"]:.0f}%'),
            ("Consistency", f'{row["Consistency"]:.0f}'),
            ("@all", f'{row["@all Response %"]:.0f}%'),
            ("Tags", f'{row["Tag Response %"]:.0f}%'),
        ]
    )
    name_pill = f'<span class="name-pill" style="background:{color}">{row["Member"]}</span>'
    return (
        '<div class="lb-row">'
        f'<div class="lb-rank">#{rank}</div>'
        f'<div class="lb-name">{name_pill}</div>'
        '<div class="lb-meter"><div class="lb-track" style="background:'
        f'{track}"><div class="lb-fill" style="width:{width}%;background:{color}"></div></div></div>'
        f'<div class="lb-score">{row["Score"]:.1f}</div>'
        f'<div class="lb-chips">{chips}</div>'
        f'<div class="lb-badge" style="color:{delta_color}">{delta:+.1f} vs avg</div>'
        '</div>'
    )

# =========================================================================
# AUTO-GENERATED INSIGHTS
# =========================================================================

def build_insights():
    items = []  # each: (tag, accent, icon, text, full)

    below_avg = master[master["Standing"] == "Below Average"].sort_values("Score")
    if len(below_avg):
        parts = [f"{row['Member']} ({row['Score']}, {row['Vs. Average']:+.1f} vs. team avg of {MEAN_SCORE})"
                 for _, row in below_avg.iterrows()]
        items.append((
            "Focus area", BELOW_COLOR, "🎯",
            f"{len(below_avg)} of {len(TEAM)} are below the team's average internal contribution (avg score {MEAN_SCORE}/100), "
            "worst first: " + "; ".join(parts) + ". This is the group to focus the conversation on — not just the "
            "extreme cases, everyone who's carrying less than their share.",
            True,
        ))

    low_scorers = master[master["Category"] == "🔴 Low"].sort_values("Score")
    if len(low_scorers):
        names = ", ".join(low_scorers["Member"].tolist())
        items.append((
            "Lowest tier", BELOW_COLOR, "🔻",
            f"Score under 40: {names} — these need a direct check-in, contribution is minimal on actual internal "
            "group work.", False,
        ))

    silent = master[master["Days Since Last Active"] >= 7].sort_values("Days Since Last Active", ascending=False)
    if len(silent):
        parts = [f"{row['Member']} ({int(row['Days Since Last Active'])}d)" for _, row in silent.iterrows()]
        items.append((
            "Gone quiet", BELOW_COLOR, "😴",
            ", ".join(parts) + f" haven't posted anywhere in 7+ days as of {TODAY.strftime('%d %b')}.", False,
        ))

    bc_low = master.sort_values("@all Response %").head(3)
    items.append((
        "Ignores @all", BELOW_COLOR, "🔕",
        ", ".join(f"{row['Member']} ({row['@all Response %']}%)" for _, row in bc_low.iterrows()) +
        " — least likely to act on group-wide asks within 30 min.", False,
    ))

    tag_low = master.sort_values("Tag Response %").head(3)
    items.append((
        "Ignores @-tags", BELOW_COLOR, "🙈",
        ", ".join(f"{row['Member']} ({row['Tag Response %']}%)" for _, row in tag_low.iterrows()) +
        " — slowest to respond even when personally called out.", False,
    ))

    flagged = external_view[(external_view["Clout Gap"] >= 4) & (external_view["External Messages"] >= 20)]
    if len(flagged):
        parts = [
            f"{row['Member']} (internal rank #{int(row['Internal Rank'])}, external rank #{int(row['External Rank'])})"
            for _, row in flagged.iterrows()
        ]
        items.append((
            "Credit-chasing check", STAKE_COLOR, "🎭",
            ", ".join(parts) + " show up much more in the Alumni Chair/Manager/Office chats than their internal "
            "contribution would suggest — visible externally, quieter on the actual internal work. See the "
            "External Visibility table below.", False,
        ))
    else:
        items.append((
            "Credit-chasing check", NEUTRAL_COLOR, "🎭",
            "No one shows a large mismatch between internal contribution and external visibility right now — "
            "external activity roughly tracks internal effort.", False,
        ))

    cons_low = master.sort_values("Consistency").head(3)
    items.append((
        "Most sporadic", BELOW_COLOR, "📉",
        ", ".join(f"{r.Member} ({r.Consistency}/100)" for r in cons_low.itertuples()) + " — bursty, not steady.", False,
    ))

    counts = master["Category"].value_counts()
    items.append((
        "Team balance", NEUTRAL_COLOR, "⚖️",
        f"{counts.get('🟢 High', 0)} High / {counts.get('🟡 Medium', 0)} Medium / "
        f"{counts.get('🔴 Low', 0)} Low engagement, out of {len(TEAM)} members — "
        f"{len(master[master['Standing']=='Below Average'])} sitting below the team average.", False,
    ))
    return items

INSIGHTS = build_insights()

# =========================================================================
# HEADER
# =========================================================================

st.title("📊 Student Alumni Relations Cell — Contribution & Engagement Dashboard")
st.caption(
    f"Internal group analyzed from **1 Mar 2026** onward · Alumni Chair / Manager / Office chats analyzed for full history "
    f"· data through **{TODAY.strftime('%d %b %Y')}** · {len(TEAM)} core members tracked "
    f"(Khushi Jain & Parth excluded — left the cell; Yash excluded from analytics) "
    f"· a 'response' = a message sent in the same chat within 30 minutes"
)
st.info(
    "**Ranking scope:** the Score/leaderboard below is based on internal-group activity only — that's where the "
    "actual cell work happens. Alumni Chair/Manager/Office activity is shown separately as context and to check for "
    "'visible externally, quiet internally' patterns — it does not count toward anyone's rank.",
    icon="ℹ️",
)

tab_summary, tab_deep, tab_p2p, tab_profile = st.tabs(
    ["📋 Summary & Decisions", "📈 Deep Dive", "⚔️ P2P Comparison", "👤 Person Profile"]
)

# -------------------------------------------------------------------------
# SUMMARY & DECISIONS
# -------------------------------------------------------------------------
with tab_summary:
    st.subheader("Where the Gaps Are")
    st.caption("Focused on who's contributing less than their share — not just the extreme cases, everyone below "
               f"the team average (avg score: **{MEAN_SCORE}/100**).")

    below_avg_all = master[master["Standing"] == "Below Average"]
    most_at_risk = master.iloc[-1]
    most_sporadic = master.sort_values("Consistency").iloc[0]
    kpis = [
        kpi_html("Team avg score", f"{MEAN_SCORE}", "out of 100 · internal only", NEUTRAL_COLOR),
        kpi_html("Below team average", f"{len(below_avg_all)} / {len(TEAM)}",
                 "members carrying less than their share", CRITICAL_COLOR),
        kpi_html("Most at risk", most_at_risk["Member"],
                 f"score {most_at_risk['Score']} · {int(most_at_risk['Days Since Last Active'])}d since last msg", CRITICAL_COLOR),
        kpi_html("Most sporadic", most_sporadic["Member"],
                 f"consistency {most_sporadic['Consistency']}/100 — bursty, not steady", CRITICAL_COLOR),
    ]
    st.markdown(f'<div class="kpi-strip">{"".join(kpis)}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    cards_html = "".join(card_html(text, tag, accent, icon, full) for tag, accent, icon, text, full in INSIGHTS)
    st.markdown(f'<div class="insight-grid">{cards_html}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Contribution Gaps — Internal Group Only")
    st.caption("Sorted weakest first. Bar = Score (0-100), colored by tier: green ≥65, amber ≥40, red below. "
               "'vs avg' on the right = how far above/below the team's average score.")
    lb_header = (
        '<div class="lb-header">'
        '<div style="width:28px"></div><div style="width:128px">Member</div>'
        '<div style="flex:1">Score</div><div style="width:42px"></div>'
        '<div style="width:300px">Signals</div><div style="width:108px;text-align:right">vs average</div>'
        '</div>'
    )
    rows_html = "".join(
        lb_row_html(i + 1, row) for i, (_, row) in enumerate(master.sort_values("Score").iterrows())
    )
    st.markdown(lb_header + rows_html, unsafe_allow_html=True)
    with st.expander("View as a sortable table"):
        display_cols = ["Member", "Score", "Vs. Average", "Standing", "Internal Messages", "Total Words",
                         "Active Days %", "Consistency", "@all Response %", "Tag Response %", "Days Since Last Active"]
        st.dataframe(
            master[display_cols].sort_values("Score").reset_index(drop=True),
            width="stretch",
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
                "Vs. Average": st.column_config.NumberColumn(format="%+.1f"),
                "Active Days %": st.column_config.NumberColumn(format="%.1f%%"),
                "@all Response %": st.column_config.NumberColumn(format="%.1f%%"),
                "Tag Response %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    st.divider()
    st.subheader("External Visibility Check — Not Part of Ranking")
    st.caption("Compares each person's internal rank to how much they show up in the Alumni Chair/Manager/Office "
               "chats. A large positive **Clout Gap** = ranks much better externally than internally — visible to "
               "alumni stakeholders without matching internal contribution. Sorted worst-gap first.")
    fig = px.scatter(master.merge(external_view[["Member", "External Messages", "Clout Gap"]], on="Member"),
                      x="Score", y="External Messages", text="Member", color="Clout Gap",
                      size="External Messages", size_max=30,
                      color_continuous_scale=[[0, "#3a3f4d"], [1, CRITICAL_COLOR]])
    fig.update_traces(textposition="top center")
    fig.update_layout(xaxis_title="Internal Score (the ranking)", yaxis_title="External messages (chair+manager+office)",
                       plot_bgcolor=SURFACE_1, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Top-left = high external visibility with a low internal score — the pattern to watch for.")
    with st.expander("View as a sortable table"):
        st.dataframe(
            external_view.set_index("Member"),
            width="stretch",
            column_config={"Stakeholder Response %": st.column_config.NumberColumn(format="%.1f%%")},
        )

# -------------------------------------------------------------------------
# DEEP DIVE
# -------------------------------------------------------------------------
with tab_deep:
    with st.expander("📨 Messages & words — by chat", expanded=False):
        chat_choice = st.selectbox("Chat", CHAT_OPTIONS, format_func=lambda c: dp.CHAT_LABELS[c], key="msg_chat")
        scoped = team_df[team_df.chat == chat_choice]

        m_counts = scoped.groupby("person").size().reindex(TEAM, fill_value=0)
        w_counts = scoped.groupby("person")["word_count"].sum().reindex(TEAM, fill_value=0)
        avg_words = (w_counts / m_counts.replace(0, np.nan)).fillna(0)
        media_counts = scoped[scoped.is_media].groupby("person").size().reindex(TEAM, fill_value=0)

        rank_df = pd.DataFrame({
            "Member": TEAM, "Messages": m_counts.values, "Words": w_counts.values,
            "Avg Words/Msg": avg_words.round(1).values, "Media Shared": media_counts.values,
        }).sort_values("Messages", ascending=False).reset_index(drop=True)
        rank_df.index += 1

        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.dataframe(rank_df, width="stretch")
        with c2:
            fig = px.bar(rank_df.sort_values("Messages"), x="Messages", y="Member", orientation="h", text="Messages")
            fig.update_layout(yaxis_title="")
            st.plotly_chart(fig, width="stretch")

        st.markdown("**Daily volume over time**")
        daily = scoped.groupby("date").size().reset_index(name="Messages")
        daily["date"] = pd.to_datetime(daily["date"])
        fig = px.line(daily, x="date", y="Messages")
        if chat_choice == "internal":
            fig.add_vline(x=dp.INTERNAL_CUTOFF, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Internal vs. external split, per person** (external = chair + manager + office combined)")
        split_df = pd.DataFrame({"Member": TEAM, "Internal": msg_vol.values, "External": external_msgs.values})
        c3, c4 = st.columns(2)
        with c3:
            order = sorted_order(pd.Series(msg_vol.values, index=TEAM))
            fig = px.bar(split_df.sort_values("Internal", ascending=False), x="Member", y="Internal",
                         category_orders={"Member": order}, title="Internal messages")
            st.plotly_chart(fig, width="stretch")
        with c4:
            order = sorted_order(pd.Series(external_msgs.values, index=TEAM))
            fig = px.bar(split_df.sort_values("External", ascending=False), x="Member", y="External",
                         category_orders={"Member": order}, title="External messages (chair+manager+office)")
            st.plotly_chart(fig, width="stretch")

    with st.expander("⏰ Activity timing", expanded=False):
        chat_choice_t = st.selectbox("Chat", ["all"] + CHAT_OPTIONS,
                                      format_func=lambda c: "All Chats" if c == "all" else dp.CHAT_LABELS[c],
                                      key="time_chat")
        scoped_t = team_df if chat_choice_t == "all" else team_df[team_df.chat == chat_choice_t]
        member_order = sorted_order(scoped_t.groupby("person").size().reindex(TEAM, fill_value=0))

        st.markdown("**Hour-of-day heatmap** (rows ordered by total volume, high to low)")
        heat = scoped_t.pivot_table(index="person", columns="hour", aggfunc="size", fill_value=0).reindex(member_order, fill_value=0)
        heat.index.name = "Member"
        heat.columns.name = "Hour"
        fig = px.imshow(heat, aspect="auto", labels=dict(x="Hour of Day", y="", color="Messages"), color_continuous_scale="Viridis")
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Time-of-day mix per person**")
            tb = scoped_t.groupby(["person", "time_bucket"]).size().reset_index(name="Messages")
            fig = px.bar(tb, x="person", y="Messages", color="time_bucket",
                         category_orders={"person": member_order, "time_bucket": TIME_BUCKET_ORDER})
            fig.update_layout(xaxis_title="", legend_title="")
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown("**Weekday mix per person**")
            wd = scoped_t.groupby(["person", "weekday"]).size().reset_index(name="Messages")
            fig = px.bar(wd, x="weekday", y="Messages", color="person",
                         category_orders={"weekday": WEEKDAY_ORDER, "person": member_order})
            fig.update_layout(xaxis_title="", legend_title="")
            st.plotly_chart(fig, width="stretch")

        st.markdown("**Peak hour per person**")
        peak_hour = scoped_t.groupby(["person", "hour"]).size().reset_index(name="c")
        peak_hour = peak_hour.loc[peak_hour.groupby("person")["c"].idxmax()].set_index("person").reindex(member_order)
        peak_tbl = pd.DataFrame({
            "Member": member_order,
            "Peak Hour": [f"{int(h):02d}:00" if pd.notna(h) else "-" for h in peak_hour["hour"]],
            "Messages at Peak": peak_hour["c"].fillna(0).astype(int).values,
        })
        st.dataframe(peak_tbl, width="stretch", hide_index=True)

    with st.expander("🏷️ Tag & @all response detail", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Personal @-tag response rate**")
            tr = master[["Member", "Tag Response %"]].sort_values("Tag Response %", ascending=False)
            fig = px.bar(tr, x="Member", y="Tag Response %", category_orders={"Member": tr["Member"].tolist()})
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown("**@all / @everyone response rate**")
            br = master[["Member", "@all Response %"]].sort_values("@all Response %", ascending=False)
            fig = px.bar(br, x="Member", y="@all Response %", category_orders={"Member": br["Member"].tolist()})
            st.plotly_chart(fig, width="stretch")

        st.markdown("**Who initiates tags / @all messages?**")
        c3, c4 = st.columns(2)
        with c3:
            tagger_counts = (tag_resp.groupby("tagged_by").size().reindex(TEAM, fill_value=0)
                              if not tag_resp.empty else pd.Series(0, index=TEAM)).sort_values(ascending=False)
            st.dataframe(tagger_counts.rename("Tags Sent").rename_axis("Member").reset_index(), width="stretch", hide_index=True)
        with c4:
            bc_sender = (df[df.is_broadcast_tag & df.person.isin(TEAM)].groupby("person").size()
                         .reindex(TEAM, fill_value=0)).sort_values(ascending=False)
            st.dataframe(bc_sender.rename("Broadcasts Sent").rename_axis("Member").reset_index(), width="stretch", hide_index=True)

    with st.expander("🎓 Stakeholder response detail", expanded=False):
        st.markdown("**Response rate to each stakeholder, by member** (rows sorted by average response rate)")
        if not sh_resp.empty:
            pivot = sh_resp.groupby(["member", "stakeholder"])["responded"].mean().unstack()
            for s in STAKEHOLDERS:
                if s not in pivot.columns:
                    pivot[s] = 0
            pivot = (pivot[STAKEHOLDERS] * 100).round(1)
            pivot = pivot.reindex(TEAM).fillna(0)
            pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
            pivot.index.name = "Member"
            fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                             labels=dict(color="Response %"))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(pivot, width="stretch")

            st.markdown("**Raw reply counts** (how many times each member actually responded)")
            counts = sh_resp[sh_resp.responded].groupby(["member", "stakeholder"]).size().unstack(fill_value=0)
            for s in STAKEHOLDERS:
                if s not in counts.columns:
                    counts[s] = 0
            counts = counts[STAKEHOLDERS].reindex(TEAM, fill_value=0)
            counts.index.name = "Member"
            st.dataframe(counts, width="stretch")

        st.markdown("**Stakeholder message volume** — who talks the most in each external chat")
        stake_vol = df[df.person_type == "stakeholder"].groupby(["chat_label", "person"]).size().reset_index(name="Messages")
        stake_order = sorted_order(df[df.person_type == "stakeholder"].groupby("person").size())
        fig = px.bar(stake_vol, x="person", y="Messages", color="chat_label", category_orders={"person": stake_order})
        fig.update_layout(xaxis_title="", legend_title="")
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Stakeholder- vs. teammate-triggered response rate** (external chats only)")
        st.caption("Positive gap = more likely to respond to alumni stakeholders than to fellow teammates in these chats.")
        if not ext_eng.empty:
            comp = ext_eng.groupby(["member", "trigger_type"])["responded"].mean().unstack().reindex(TEAM)
            comp = (comp * 100).round(1)
            comp.columns = [f"{c.title()}-triggered %" for c in comp.columns]
            comp["Gap (Stakeholder − Team)"] = (comp.get("Stakeholder-triggered %", 0) - comp.get("Team-triggered %", 0)).round(1)
            comp = comp.sort_values("Gap (Stakeholder − Team)", ascending=False)
            comp.index.name = "Member"
            st.dataframe(comp, width="stretch")

# -------------------------------------------------------------------------
# P2P COMPARISON
# -------------------------------------------------------------------------
with tab_p2p:
    st.subheader("⚔️ Person-to-Person Comparison")
    st.caption("Pick 2-3 members to put head-to-head across internal contribution, responsiveness, and external "
               "visibility. Ranking metrics are internal-only; external numbers are shown for context.")

    selected = st.multiselect("Select members to compare", TEAM, default=TEAM[:2], max_selections=3)

    if len(selected) < 2:
        st.info("Pick at least 2 members to start a comparison.")
    else:
        pdf = master[master.Member.isin(selected)].set_index("Member").reindex(selected)
        edf = external_view[external_view.Member.isin(selected)].set_index("Member").reindex(selected)

        st.markdown("### Scorecards")
        cols = st.columns(len(selected))
        for col, member in zip(cols, selected):
            row, erow = pdf.loc[member], edf.loc[member]
            with col:
                st.markdown(f"#### {hl(member)}", unsafe_allow_html=True)
                st.metric("Internal Rank", f"#{int(row['Rank'])} / {len(TEAM)}", row["Category"])
                st.metric("Score", f"{row['Score']}", f"{row['Vs. Average']:+.1f} vs avg")
                st.metric("Internal Messages", int(row["Internal Messages"]))
                st.metric("Total Words", int(row["Total Words"]))
                st.metric("Active Days %", f"{row['Active Days %']}%")
                st.metric("Consistency", row["Consistency"])
                st.metric("@all Response %", f"{row['@all Response %']}%")
                st.metric("Tag Response %", f"{row['Tag Response %']}%")
                st.metric("Days Since Last Active", int(row["Days Since Last Active"]))
                st.metric("External Messages", int(erow["External Messages"]), f"rank #{int(erow['External Rank'])}")
                st.metric("Stakeholder Response %", f"{erow['Stakeholder Response %']}%")

        st.divider()
        st.markdown("### Head-to-Head — who leads on each metric")
        h2h_metrics = {
            "Score": True, "Internal Messages": True, "Total Words": True, "Active Days %": True,
            "Consistency": True, "@all Response %": True, "Tag Response %": True,
            "Days Since Last Active": False,  # lower is better here
        }
        rows = []
        for metric, higher_is_better in h2h_metrics.items():
            vals = pdf[metric]
            leader = vals.idxmin() if not higher_is_better else vals.idxmax()
            row = {"Metric": metric, **{m: vals[m] for m in selected}, "Leads": leader}
            rows.append(row)
        h2h_df = pd.DataFrame(rows).set_index("Metric")
        st.dataframe(h2h_df, width="stretch")
        wins = pd.Series([r["Leads"] for r in rows]).value_counts().reindex(selected, fill_value=0)
        st.markdown(
            "<div style='opacity:0.85; font-size:0.9rem;'><b>Metrics led:</b> " +
            " &nbsp;·&nbsp; ".join(f"{hl(m)} {wins[m]}/{len(h2h_metrics)}" for m in selected) + "</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(f"### Radar — normalized profile (0-100, relative to all {len(TEAM)} members)")
        radar_labels = ["Msg Volume", "Word Depth", "Active Days %", "Consistency",
                         "@all Responsiveness", "Tag Responsiveness"]
        fig = go.Figure()
        for member in selected:
            vals = (components.loc[member, radar_labels] * 100).round(1).tolist()
            fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=radar_labels + [radar_labels[0]],
                                           fill="toself", name=member))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
        st.plotly_chart(fig, width="stretch")

        st.divider()
        st.markdown("### Side-by-side bars")
        c1, c2 = st.columns(2)
        with c1:
            vol_df = pdf.reset_index()[["Member", "Internal Messages", "Total Words"]].melt(
                id_vars="Member", var_name="Metric", value_name="Value")
            fig = px.bar(vol_df, x="Metric", y="Value", color="Member", barmode="group")
            st.plotly_chart(fig, width="stretch")
        with c2:
            rate_df = pdf.reset_index()[["Member", "Active Days %", "@all Response %", "Tag Response %"]].melt(
                id_vars="Member", var_name="Metric", value_name="Value")
            fig = px.bar(rate_df, x="Metric", y="Value", color="Member", barmode="group")
            fig.update_layout(yaxis_title="%")
            st.plotly_chart(fig, width="stretch")

# -------------------------------------------------------------------------
# PERSON PROFILE
# -------------------------------------------------------------------------
with tab_profile:
    st.subheader("👤 Individual Profile")
    person = st.selectbox("Select member", master["Member"].tolist())

    p_row = master[master.Member == person].iloc[0]
    p_ext = external_view[external_view.Member == person].iloc[0]
    p_all = team_df[team_df.person == person]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Internal Rank", f"#{int(p_row.Rank)} / {len(TEAM)}", p_row["Category"])
    c2.metric("Internal Messages", int(p_row["Internal Messages"]))
    c3.metric("External Messages", int(p_ext["External Messages"]), f"rank #{int(p_ext['External Rank'])}")
    last_msg = p_all["datetime"].max()
    c4.metric("Last Active", last_msg.strftime("%d %b %Y") if pd.notna(last_msg) else "—",
              f"{int(p_row['Days Since Last Active'])}d ago")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("@all Response Rate", f"{p_row['@all Response %']}%")
    r2.metric("Tag Response Rate", f"{p_row['Tag Response %']}%")
    r3.metric("Stakeholder Response Rate", f"{p_ext['Stakeholder Response %']}%", help="Context only — not part of the internal ranking")
    r4.metric("Consistency Score", f"{p_row['Consistency']}/100")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Messages by chat**")
        by_chat = p_all.groupby("chat_label").size().sort_values(ascending=False)
        fig = px.pie(values=by_chat.values, names=by_chat.index, hole=0.4)
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.markdown("**Active hours**")
        by_hour = p_all.groupby("hour").size().reindex(range(24), fill_value=0)
        fig = px.bar(x=by_hour.index, y=by_hour.values, labels={"x": "Hour", "y": "Messages"})
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Daily activity (internal group)**")
    daily = p_all[p_all.chat == "internal"].groupby("date").size().reset_index(name="Messages")
    daily["date"] = pd.to_datetime(daily["date"])
    fig = px.line(daily, x="date", y="Messages")
    st.plotly_chart(fig, width="stretch")

    if not sh_resp.empty:
        st.markdown("**Response rate by stakeholder**")
        per_stake = sh_resp[sh_resp.member == person].groupby("stakeholder")["responded"].mean().reindex(STAKEHOLDERS).fillna(0) * 100
        per_stake = per_stake.sort_values(ascending=False)
        st.bar_chart(per_stake.round(1))
