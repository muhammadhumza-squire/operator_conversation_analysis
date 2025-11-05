# app.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Phone Agent Conversation Analysis", layout="wide")
st.title("📞 Operator Conversation Analysis – Full Feature Report")

# -----------------------------
# Data
# -----------------------------
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    # best-effort datetime parse
    if "booked_time" in df.columns:
        df["booked_time"] = pd.to_datetime(df["booked_time"], errors="coerce", utc=True)
    if "analysis_timestamp" in df.columns:
        df["analysis_timestamp"] = pd.to_datetime(df["analysis_timestamp"], errors="coerce", utc=True)
    # normalize bool-like strings if any
    for c in df.columns:
        if df[c].dtype == object and set(df[c].dropna().unique()).issubset({"True","False","true","false"}):
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
    return df

# 👇 change this filename to your CSV if different
df = load_data("operator_conversation_analysis.csv")

# Helpers
def has(col): return col in df.columns

def pct(x): 
    try:
        return 100.0 * float(x)
    except Exception:
        return x

def safe_mean(col):
    return df[col].mean() if has(col) else np.nan

def bool_rate(col):
    return (df[col].mean() * 100) if has(col) else np.nan

def ntrue(col):
    return int(df[col].sum()) if has(col) else 0

def count_bar(series, title, xlab="Category", ylab="Count", sort_desc=True):
    c = series.value_counts(dropna=False).reset_index()
    c.columns = [xlab, ylab]
    if sort_desc:
        c = c.sort_values(ylab, ascending=False)
    fig = px.bar(c, x=xlab, y=ylab, text_auto=True, title=title)
    fig.update_layout(yaxis_title=ylab, xaxis_title=xlab)
    return fig

def hist(col, title, nbins=20):
    fig = px.histogram(df, x=col, nbins=nbins, title=title)
    fig.update_layout(xaxis_title=col, yaxis_title="Count")
    return fig

def box(col, by=None, title=""):
    fig = px.box(df, x=by, y=col, title=title, points="outliers")
    return fig

def line(col, x="analysis_timestamp", title=""):
    if not has(x): 
        tmp = df.copy()
        tmp["row_index"] = np.arange(len(tmp))
        x = "row_index"
    else:
        tmp = df.sort_values(x)
    fig = px.line(tmp, x=x, y=col, title=title, markers=True)
    return fig

def parse_tags(col):
    """Split comma-separated tags into tidy frame"""
    if not has(col):
        return pd.DataFrame(columns=["tag"])
    s = df[col].fillna("").astype(str)
    tags = []
    for row in s:
        for t in [x.strip() for x in row.split(",") if x.strip()]:
            tags.append(t)
    return pd.DataFrame({"tag": tags})

def cooccurrence(col):
    """Co-occurrence matrix for comma-separated tags"""
    if not has(col):
        return pd.DataFrame()
    # build binary matrix
    all_tags = sorted(parse_tags(col)["tag"].unique().tolist())
    if not all_tags:
        return pd.DataFrame()
    binM = np.zeros((len(df), len(all_tags)), dtype=int)
    for i, row in enumerate(df[col].fillna("").astype(str)):
        row_tags = {t.strip() for t in row.split(",") if t.strip()}
        for j, t in enumerate(all_tags):
            if t in row_tags:
                binM[i, j] = 1
    M = pd.DataFrame(binM, columns=all_tags)
    co = M.T @ M
    np.fill_diagonal(co.values, 0)
    return co

def heatmap_from_matrix(mat_df, title):
    if mat_df.empty:
        return go.Figure()
    fig = px.imshow(
        mat_df,
        x=mat_df.columns,
        y=mat_df.index,
        color_continuous_scale="Blues",
        title=title,
        aspect="auto",
        labels=dict(color="Co-occurrence"),
    )
    return fig

# -----------------------------
# Global Filters (optional)
# -----------------------------
with st.expander("🔍 Filters", expanded=False):
    sentiment_sel = st.multiselect("Final Sentiment", sorted(df["final_sentiment"].dropna().unique()) if has("final_sentiment") else [], default=None)
    resolution_sel = st.multiselect("Resolution Status", sorted(df["resolution_status"].dropna().unique()) if has("resolution_status") else [], default=None)
    service_sel = st.multiselect("Booked Service", sorted(df["booked_service"].dropna().unique()) if has("booked_service") else [], default=None)
    barber_sel = st.multiselect("Booked Barber", sorted(df["booked_barber"].dropna().unique()) if has("booked_barber") else [], default=None)

mask = pd.Series(True, index=df.index)
if sentiment_sel and has("final_sentiment"):
    mask &= df["final_sentiment"].isin(sentiment_sel)
if resolution_sel and has("resolution_status"):
    mask &= df["resolution_status"].isin(resolution_sel)
if service_sel and has("booked_service"):
    mask &= df["booked_service"].isin(service_sel)
if barber_sel and has("booked_barber"):
    mask &= df["booked_barber"].isin(barber_sel)

df = df[mask].copy()

# -----------------------------
# KPI Row
# -----------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Calls", len(df))
k2.metric("Booking Success %", f"{bool_rate('booking_success'):.1f}%" if has("booking_success") else "–")
k3.metric("Avg Length (sec)", f"{safe_mean('conversation_length_seconds'):.1f}" if has("conversation_length_seconds") else "–")
k4.metric("Avg Empathy", f"{safe_mean('empathy_score'):.2f}" if has("empathy_score") else "–")
k5.metric("Avg Adherence", f"{safe_mean('assistant_adherence_score'):.2f}" if has("assistant_adherence_score") else "–")

st.markdown("---")

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "Overview",
    "Assistant KPIs",
    "Client Struggles (Legacy)",
    "Conversational Quality",
    "Flow & Recovery",
    "Handoff / Reschedule / Cancel",
    "Struggling Areas (New)",
    "Signals & Metadata",
    "Correlations",
    "Raw Data",
])

# -----------------------------
# 1) Overview
# -----------------------------
with tabs[0]:
    c1, c2 = st.columns(2)
    if has("final_sentiment"):
        fig = px.pie(df, names="final_sentiment", title="Final Sentiment Distribution")
        c1.plotly_chart(fig, use_container_width=True)
    if has("resolution_status"):
        fig = px.pie(df, names="resolution_status", title="Resolution Status Distribution")
        c2.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    # Booking by service/barber if present
    if has("booked_service") and has("booking_success"):
        svc = df.groupby("booked_service")["booking_success"].mean().mul(100).reset_index()
        fig = px.bar(svc.sort_values("booking_success", ascending=False),
                     x="booked_service", y="booking_success", text_auto=True,
                     title="Booking Success by Service (%)", color="booking_success")
        c3.plotly_chart(fig, use_container_width=True)
    if has("booked_barber") and has("booking_success"):
        barb = df.groupby("booked_barber")["booking_success"].mean().mul(100).reset_index()
        fig = px.bar(barb.sort_values("booking_success", ascending=False),
                     x="booked_barber", y="booking_success", text_auto=True,
                     title="Booking Success by Barber (%)", color="booking_success")
        c4.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 2) Assistant KPIs
# -----------------------------
with tabs[1]:
    st.subheader("Protocol & Policy")
    proto_cols = [
        "protocol_asked_name",
        "protocol_performed_lookup",
        "payment_policy_discussed",
        "alternatives_offered",
        "existing_appointment_detected",
        "double_booking_attempted_by_user",
        "double_booking_prevented_by_assistant",
    ]
    data = []
    for c in proto_cols:
        if has(c):
            data.append({"KPI": c, "Rate %": df[c].mean()*100})
    if data:
        kpi_df = pd.DataFrame(data).sort_values("Rate %", ascending=False)
        fig = px.bar(kpi_df, x="KPI", y="Rate %", text_auto=".1f", title="Assistant KPI Rates (%)", color="Rate %")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Booked Details")
    c1, c2 = st.columns(2)
    if has("booked_time"):
        fig = hist("booked_time", "Distribution of Booked Time (if present)", nbins=30)
        c1.plotly_chart(fig, use_container_width=True)
    if has("assistant_notes"):
        # Top tokens from assistant_notes (simple, no extra deps)
        words = (
            df["assistant_notes"]
            .dropna().astype(str).str.lower()
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.split()
        )
        stop = set("the a and to of in for is it this that with on by at from as be or not you i we they he she are was were will would could should".split())
        tokens = [w for lst in words for w in lst if w not in stop and len(w) > 2]
        wc = pd.Series(tokens).value_counts().head(25).reset_index()
        wc.columns = ["word", "count"]
        if not wc.empty:
            fig = px.bar(wc, x="word", y="count", title="Assistant Notes – Top Keywords", text_auto=True)
            c2.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 3) Client Struggles (Legacy)
# -----------------------------
with tabs[2]:
    c1, c2 = st.columns(2)
    if has("time_confusion"):
        fig = count_bar(df["time_confusion"].map({True:"Yes", False:"No"}), "Time Confusion (Legacy)")
        c1.plotly_chart(fig, use_container_width=True)
    if has("payment_confusion"):
        fig = count_bar(df["payment_confusion"].map({True:"Yes", False:"No"}), "Payment Confusion (Legacy)")
        c2.plotly_chart(fig, use_container_width=True)

    if has("proactive_pause_gap"):
        fig = count_bar(df["proactive_pause_gap"].map({True:"Yes", False:"No"}), "Proactive Pause / Long Gap")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 4) Conversational Quality
# -----------------------------
with tabs[3]:
    c1, c2 = st.columns(2)
    if has("assistant_tone_consistency"):
        c1.plotly_chart(hist("assistant_tone_consistency", "Tone Consistency – Distribution"), use_container_width=True)
    if has("empathy_score"):
        c2.plotly_chart(hist("empathy_score", "Empathy Score – Distribution"), use_container_width=True)

    c3, c4 = st.columns(2)
    if has("clarity_score") and has("empathy_score"):
        fig = px.scatter(df, x="clarity_score", y="empathy_score",
                         color=df["final_sentiment"] if has("final_sentiment") else None,
                         title="Clarity vs Empathy")
        c3.plotly_chart(fig, use_container_width=True)
    if has("assistant_adherence_score") and has("analysis_timestamp"):
        c4.plotly_chart(line("assistant_adherence_score", x="analysis_timestamp", title="Adherence Over Time"),
                        use_container_width=True)

# -----------------------------
# 5) Flow & Recovery
# -----------------------------
with tabs[4]:
    c1, c2 = st.columns(2)
    if has("conversation_length_seconds"):
        c1.plotly_chart(hist("conversation_length_seconds", "Conversation Length (sec) – Distribution"), use_container_width=True)
    if has("assistant_recovery_actions"):
        # simple frequency of phrases
        words = (
            df["assistant_recovery_actions"]
            .dropna().astype(str).str.lower()
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.split()
        )
        stop = set("the a and to of in for is it this that with on by at from as be or not you i we they he she are was were will would could should".split())
        tokens = [w for lst in words for w in lst if w not in stop and len(w) > 2]
        wc = pd.Series(tokens).value_counts().head(25).reset_index()
        wc.columns = ["word", "count"]
        if not wc.empty:
            fig = px.bar(wc, x="word", y="count", title="Assistant Recovery – Top Keywords", text_auto=True)
            c2.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    for col, label, container in [
        ("assistant_repetition_count","Assistant Repetition Count",c3),
        ("assistant_clarification_count","Assistant Clarification Count",c4)
    ]:
        if has(col):
            container.plotly_chart(hist(col, f"{label} – Distribution"), use_container_width=True)

    c5, c6 = st.columns(2)
    if has("user_clarification_count"):
        c5.plotly_chart(hist("user_clarification_count", "User Clarification Count – Distribution"), use_container_width=True)
    if has("repeated_user_input"):
        fig = count_bar(df["repeated_user_input"].map({True:"Yes", False:"No"}), "Repeated User Input")
        c6.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 6) Handoff / Reschedule / Cancel
# -----------------------------
with tabs[5]:
    c1, c2, c3 = st.columns(3)
    if has("user_asks_human"):
        c1.plotly_chart(count_bar(df["user_asks_human"].map({True:"Yes", False:"No"}), "User Asked for Human"), use_container_width=True)
    if has("user_requests_reschedule"):
        c2.plotly_chart(count_bar(df["user_requests_reschedule"].map({True:"Yes", False:"No"}), "Reschedule Requested"), use_container_width=True)
    if has("user_requests_cancel"):
        c3.plotly_chart(count_bar(df["user_requests_cancel"].map({True:"Yes", False:"No"}), "Cancellation Requested"), use_container_width=True)

    # Impact on booking success
    impact_cols = [("user_asks_human","Human Handoff"),
                   ("user_requests_reschedule","Reschedule"),
                   ("user_requests_cancel","Cancel")]
    for c, label in impact_cols:
        if has(c) and has("booking_success"):
            tmp = df.groupby(c)["booking_success"].mean().mul(100).reset_index()
            tmp[c] = tmp[c].map({True:"Yes", False:"No"})
            fig = px.bar(tmp, x=c, y="booking_success", text_auto=True, title=f"Booking Success vs {label}")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 7) Struggling Areas (New)
# -----------------------------
with tabs[6]:
    c1, c2 = st.columns(2)
    if has("user_struggle_present"):
        c1.plotly_chart(count_bar(df["user_struggle_present"].map({True:"Yes", False:"No"}), "User Struggle Present"), use_container_width=True)
    if has("assistant_struggle_present"):
        c2.plotly_chart(count_bar(df["assistant_struggle_present"].map({True:"Yes", False:"No"}), "Assistant Struggle Present"), use_container_width=True)

    st.subheader("User Struggle Tags")
    utags = parse_tags("user_struggle_tags")
    if not utags.empty:
        fig = count_bar(utags["tag"], "User Struggles by Tag", xlab="Tag")
        st.plotly_chart(fig, use_container_width=True)

        # Co-occurrence
        uco = cooccurrence("user_struggle_tags")
        if not uco.empty:
            st.plotly_chart(heatmap_from_matrix(uco, "User Struggle Tag Co-occurrence"), use_container_width=True)

    st.subheader("Assistant Struggle Tags")
    atags = parse_tags("assistant_struggle_tags")
    if not atags.empty:
        fig = count_bar(atags["tag"], "Assistant Struggles by Tag", xlab="Tag")
        st.plotly_chart(fig, use_container_width=True)

        aco = cooccurrence("assistant_struggle_tags")
        if not aco.empty:
            st.plotly_chart(heatmap_from_matrix(aco, "Assistant Struggle Tag Co-occurrence"), use_container_width=True)

    # Confusion type x Resolution
    if has("user_confusion_type") and has("resolution_status"):
        pivot = pd.crosstab(df["user_confusion_type"].fillna("none"),
                            df["resolution_status"].fillna("unresolved"))
        fig = px.imshow(pivot, text_auto=True, color_continuous_scale="Blues",
                        title="User Confusion Type × Resolution Status")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 8) Signals & Metadata
# -----------------------------
with tabs[7]:
    c1, c2, c3 = st.columns(3)
    if has("language_switch_detected"):
        c1.plotly_chart(count_bar(df["language_switch_detected"].map({True:"Yes", False:"No"}), "Language Switch Detected"), use_container_width=True)
    if has("function_error_present"):
        c2.plotly_chart(count_bar(df["function_error_present"].map({True:"Yes", False:"No"}), "Function Error Present"), use_container_width=True)
    if has("service_not_offered_by_selected_barber"):
        c3.plotly_chart(count_bar(df["service_not_offered_by_selected_barber"].map({True:"Yes", False:"No"}), "Service Not Offered by Selected Barber"), use_container_width=True)

    if has("analysis_timestamp"):
        tmp = df.dropna(subset=["analysis_timestamp"]).copy()
        if not tmp.empty:
            tmp["date"] = tmp["analysis_timestamp"].dt.date
            daily = tmp.groupby("date").size().reset_index(name="count")
            fig = px.line(daily, x="date", y="count", title="Analyses per Day", markers=True)
            st.plotly_chart(fig, use_container_width=True)

    if has("model_source"):
        st.plotly_chart(count_bar(df["model_source"], "Model Source Share"), use_container_width=True)

# -----------------------------
# 9) Correlations
# -----------------------------
with tabs[8]:
    st.subheader("Correlation Matrix – Core Numeric Metrics")
    num_cols = [c for c in [
        "assistant_adherence_score","assistant_tone_consistency","empathy_score","clarity_score",
        "assistant_repetition_count","assistant_clarification_count","user_clarification_count",
        "conversation_length_seconds"
    ] if has(c)]
    if num_cols:
        corr = df[num_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                        title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    # Booking success by categorical factors
    st.subheader("Booking Success by Factors")
    if has("booking_success"):
        cat_factors = [c for c in ["final_sentiment","user_confusion_type","resolution_status","booked_service","booked_barber"] if has(c)]
        for c in cat_factors:
            tmp = df.groupby(c)["booking_success"].mean().mul(100).reset_index().dropna()
            if not tmp.empty:
                fig = px.bar(tmp.sort_values("booking_success", ascending=False), x=c, y="booking_success",
                             text_auto=True, title=f"Booking Success by {c} (%)", color="booking_success")
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 10) Raw Data
# -----------------------------
with tabs[9]:
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Download Filtered Data (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="conversation_analysis_filtered.csv", mime="text/csv")

st.caption("Tip: Use the Filters expander above to slice sentiments, resolution, services, or barbers. All charts update live.")
