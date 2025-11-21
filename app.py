# app.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================
# Streamlit Config
# =============================
st.set_page_config(page_title="Phone Agent Conversation Analysis", layout="wide")
st.title("📞 Operator Conversation Analysis – Full Feature Report")

# =============================
# Required Defaults (your schema)
# =============================
REQUIRED_DEFAULTS = {
    # --- Assistant KPIs ---
    "protocol_asked_name": False,
    "protocol_performed_lookup": False,
    "nonexistent_barber_detected": False,
    "alternatives_offered": False,
    "existing_appointment_detected": False,
    "double_booking_attempted_by_user": False,
    "double_booking_prevented_by_assistant": False,
    "user_confusion_present": False,
    "user_confusion_type": "none",
    "user_confusion_resolved_by_assistant": False,
    "confusion_resolution_details": None,
    "payment_policy_discussed": False,

    # --- Tri-state outcomes ---
    "booking_success": None,      # null | true | false
    "reschedule_success": None,   # null | true | false
    "cancel_success": None,       # null | true | false
    "user_asks_walkin": None,     # ✅ fixed name

    # Booking details
    "booked_service": None,
    "booked_barber": None,
    "booked_time": None,

    # Schedule change notes
    "reschedule_details": None,
    "cancel_details": None,

    # Low booking analytics
    "low_booking_reason": "",
    "low_booking_comment": "",

    "assistant_adherence_score": 0.0,
    "assistant_notes": None,

    # --- Client Struggles (legacy) ---
    "time_confusion": False,
    "time_confusion_details": None,
    "payment_confusion": False,
    "payment_confusion_details": None,
    "proactive_pause_gap": False,
    "proactive_pause_gap_details": None,
    "summary": None,

    # --- Conversational Quality ---
    "assistant_tone_consistency": 0.0,
    "empathy_score": 0.0,
    "clarity_score": 0.0,
    "final_sentiment": "neutral",
    "resolution_status": "unresolved",

    # --- Flow & Recovery ---
    "assistant_recovery_actions": None,
    "repeated_user_input": False,
    "assistant_repetition_count": 0,
    "assistant_clarification_count": 0,
    "user_clarification_count": 0,
    "conversation_length_seconds": None,
    "user_long_pause_count": 0,
    "user_pause_pattern": "",
    "assistant_long_pause_count": 0,
    "assistant_pause_pattern": "",

    # --- Human handoff / schedule changes ---
    "user_asks_human": False,
    "user_asks_human_details": None,
    "user_requests_reschedule": False,
    "user_requests_cancel": False,

    # --- Struggles (per new definition) ---
    "user_struggle_present": False,
    "user_struggle_tags": "",
    "user_struggle_details": None,

    "assistant_struggle_present": False,
    "assistant_struggle_tags": "",
    "assistant_struggle_details": None,

    # --- Optional helpful signals ---
    "languages_detected": "",          # ✅ will never be NaN if missing
    "language_switch_detected": False,
    "function_error_present": False,
    "service_not_offered_by_selected_barber": False,
    "unavailable_requested_items": "",

    # --- Metadata ---
    "model_source": "openai:gpt-4o-mini",
    "analysis_timestamp": None,
}

# =============================
# Data Loading + Normalization
# =============================
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", engine="c")
    return df


def normalize_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Ensure all expected columns exist
    for col, default in REQUIRED_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    # 2) Normalize common null-like strings
    df.replace({"null": np.nan, "None": np.nan, "none": np.nan, "": np.nan}, inplace=True)

    # 3) Coerce tri-state to True/False/NaN (if strings slipped in)
    for tri in ["booking_success", "reschedule_success", "cancel_success"]:
        if tri in df.columns:
            s = df[tri].astype(str).str.strip().str.lower()
            df[tri] = s.replace({"true": True, "false": False, "nan": np.nan})
            df[tri] = df[tri].where(df[tri].isin([True, False]), np.nan)

    # 4) Parse datetimes
    if "booked_time" in df.columns:
        df["booked_time"] = pd.to_datetime(df["booked_time"], utc=True, errors="coerce")
    if "analysis_timestamp" in df.columns:
        df["analysis_timestamp"] = pd.to_datetime(df["analysis_timestamp"], utc=True, errors="coerce")

    # Normalize explicit `date` column (for filters & all time-series)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 5) Cast numeric quality/flow fields
    numeric_cols = [
        "assistant_adherence_score",
        "assistant_tone_consistency",
        "empathy_score",
        "clarity_score",
        "assistant_repetition_count",
        "assistant_clarification_count",
        "user_clarification_count",
        "conversation_length_seconds",
        "user_long_pause_count",
        "assistant_long_pause_count",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 6) Cast booleans consistently where intended (non-tristate)
    bool_cols = [
        "protocol_asked_name", "protocol_performed_lookup", "nonexistent_barber_detected",
        "alternatives_offered", "existing_appointment_detected", "double_booking_attempted_by_user",
        "double_booking_prevented_by_assistant", "user_confusion_present", "payment_policy_discussed",
        "time_confusion", "payment_confusion", "proactive_pause_gap", "repeated_user_input",
        "user_asks_human", "user_struggle_present", "assistant_struggle_present",
        "language_switch_detected", "function_error_present", "service_not_offered_by_selected_barber",
        "user_asks_walkin",
    ]
    for b in bool_cols:
        if b in df.columns:
            if df[b].dtype == object:
                df[b] = df[b].astype(str).str.lower().map({"true": True, "false": False})
            df[b] = df[b].astype("boolean").astype(object).map({True: True, False: False, None: np.nan})
    return df


# 👇 change this filename to your CSV if different
df = load_csv("../operator_conversation_analysis.tsv")
df = normalize_and_validate(df)

# =============================
# Helpers
# =============================
def has(col): 
    return col in df.columns

def safe_mean(col):
    return df[col].mean() if has(col) else np.nan

def bool_rate(col):
    # For tri-state (True/False/NaN), mean() calculates success rate among requested (nulls excluded)
    return (df[col].mean() * 100) if has(col) else np.nan

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

def heatmap_from_matrix(mat_df, title):
    """
    Safely render a heatmap for a co-occurrence matrix.
    Returns an empty figure if the matrix is empty.
    """
    if mat_df is None or mat_df.empty:
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

def line(col, x="date", title=""):
    # prefer `date` column; if missing, try analysis_timestamp
    if x == "date" and not has("date"):
        x = "analysis_timestamp"
    if not has(x):
        tmp = df.copy()
        tmp["row_index"] = np.arange(len(tmp))
        x = "row_index"
    else:
        tmp = df.dropna(subset=[x, col]).sort_values(x)
    fig = px.line(tmp, x=x, y=col, title=title, markers=True)
    return fig

def parse_tags(col):
    if not has(col):
        return pd.DataFrame(columns=["tag"])
    s = df[col].fillna("").astype(str)
    tags = []
    for row in s:
        for t in [x.strip() for x in row.split(",") if x.strip()]:
            tags.append(t)
    return pd.DataFrame({"tag": tags})

def cooccurrence(col):
    if not has(col):
        return pd.DataFrame()
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

def parse_pause_patterns(col: str, side_label: str) -> pd.DataFrame:
    """
    Parse pattern strings like '2-one_moment;1-long_pause' into rows:
    side | pause_type | count
    """
    if not has(col):
        return pd.DataFrame(columns=["side", "pause_type", "count"])
    rows = []
    for raw in df[col].fillna("").astype(str):
        raw = raw.strip()
        if not raw:
            continue
        parts = [p for p in raw.split(";") if p.strip()]
        for p in parts:
            try:
                cnt_str, p_type = p.split("-", 1)
                cnt = int(cnt_str)
                p_type = p_type.strip()
                if p_type:
                    rows.append({"side": side_label, "pause_type": p_type, "count": cnt})
            except Exception:
                continue
    return pd.DataFrame(rows)

# =============================
# Filters (Date + Customer only)
# =============================
with st.expander("🔍 Filters", expanded=True):
    # Date filter – prefer `date` column
    if has("date") and df["date"].notna().any():
        min_dt = df["date"].min().date()
        max_dt = df["date"].max().date()
        date_mode = st.radio(
            "Date filter (based on `date` column)",
            options=["All dates", "Select range"],
            horizontal=True,
        )
        selected_start, selected_end = min_dt, max_dt
        if date_mode == "Select range":
            selected_start, selected_end = st.date_input(
                "Select date range (inclusive)",
                value=(min_dt, max_dt),
                min_value=min_dt,
                max_value=max_dt,
            )
    else:
        date_mode = "All dates"
        selected_start, selected_end = None, None
        st.info("`date` column not available or all null – date filter disabled.")

    # Customer filter – try customer_id, else conversation_id
    customer_id_col = "customer_id" if has("customer_id") else "conversation_id"
    customer_filter_label = "Customer ID" if customer_id_col == "customer_id" else "Conversation ID"

    if has(customer_id_col):
        cust_values = sorted(df[customer_id_col].dropna().astype(str).unique())
        cust_values_display = ["All"] + cust_values
        selected_customer = st.selectbox(
            f"{customer_filter_label} filter",
            options=cust_values_display,
            index=0,
            help="Select a specific customer/conversation or keep 'All'.",
        )
    else:
        selected_customer = "All"
        st.info("`customer_id` / `conversation_id` not found – customer filter disabled.")

# Apply filters
mask = pd.Series(True, index=df.index)

# Date range filter using `date`
if date_mode == "Select range" and selected_start is not None and selected_end is not None and has("date"):
    dt_series = df["date"].dt.date
    mask &= dt_series.between(selected_start, selected_end)

# Customer filter
if selected_customer != "All" and has(customer_id_col):
    mask &= df[customer_id_col].astype(str) == selected_customer

df = df[mask].copy()

# =============================
# KPI Row (Overview metrics)
# =============================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Calls", len(df))
k2.metric(
    "Booking Success %",
    f"{bool_rate('booking_success'):.1f}%" if has("booking_success") else "–",
    help="Success rate among requests (booking_success True / (True+False)). Nulls mean 'not requested'.",
)
k3.metric(
    "Avg Length (sec)",
    f"{safe_mean('conversation_length_seconds'):.1f}" if has("conversation_length_seconds") else "–",
)
k4.metric("Avg Empathy", f"{safe_mean('empathy_score'):.2f}" if has("empathy_score") else "–")
k5.metric("Avg Adherence", f"{safe_mean('assistant_adherence_score'):.2f}" if has("assistant_adherence_score") else "–")

if has("user_asks_human"):
    total_human = int((df["user_asks_human"] == True).sum())
else:
    total_human = 0

if has("user_asks_walkin"):
    total_walkin = int((df["user_asks_walkin"] == True).sum())
else:
    total_walkin = 0

m1, m2 = st.columns(2)

m1.metric(
    "User Asked for Human",
    value=str(total_human),
    help="Number of calls where the customer asked to talk to a human.",
)

m2.metric(
    "User Asked About Walk-ins",
    value=str(total_walkin),
    help="Number of calls where the customer asked about walk-ins.",
)

st.caption("Use the Filters above to slice by date or customer. All charts update live.")
st.markdown("---")

# =============================
# Tabs
# =============================
tabs = st.tabs([
    "Overview",
    "Assistant KPIs",
    "Client Struggles (Legacy)",
    "Conversational Quality",
    "Outcomes (Booking/Reschedule/Cancel)",  # tri-state tab
    "Flow & Recovery",
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

    # Final Sentiment as bar chart
    if has("final_sentiment"):
        fs_counts = df["final_sentiment"].value_counts().reset_index()
        fs_counts.columns = ["final_sentiment", "count"]
        c1.plotly_chart(
            px.bar(
                fs_counts,
                x="final_sentiment",
                y="count",
                text_auto=True,
                title="Final Sentiment Distribution",
            ),
            use_container_width=True,
        )
    else:
        c1.info("final_sentiment column not found.")

    # Resolution Status as bar chart
    if has("resolution_status"):
        rs_counts = df["resolution_status"].value_counts().reset_index()
        rs_counts.columns = ["resolution_status", "count"]
        c2.plotly_chart(
            px.bar(
                rs_counts,
                x="resolution_status",
                y="count",
                text_auto=True,
                title="Resolution Status Distribution",
            ),
            use_container_width=True,
        )
    else:
        c2.info("resolution_status column not found.")

    c3, c4 = st.columns(2)
    if has("booked_service") and has("booking_success"):
        svc = df.groupby("booked_service")["booking_success"].mean().mul(100).reset_index()
        c3.plotly_chart(
            px.bar(
                svc.sort_values("booking_success", ascending=False),
                x="booked_service",
                y="booking_success",
                text_auto=True,
                title="Booking Success by Service (%)",
                color="booking_success",
            ),
            use_container_width=True,
        )
    else:
        c3.info("booked_service or booking_success column not found.")

    if has("booked_barber") and has("booking_success"):
        barb = df.groupby("booked_barber")["booking_success"].mean().mul(100).reset_index()
        c4.plotly_chart(
            px.bar(
                barb.sort_values("booking_success", ascending=False),
                x="booked_barber",
                y="booking_success",
                text_auto=True,
                title="Booking Success by Barber (%)",
                color="booking_success",
            ),
            use_container_width=True,
        )
    else:
        c4.info("booked_barber or booking_success column not found.")

    # --- Summary table for requested outcomes (from tri-state) ---
    st.markdown("---")
    st.subheader("Summary (Requested Only) – Booking / Reschedule / Cancel")

    def tri_stats(dataframe, col, label):
        if col not in dataframe.columns:
            return {"label": label, "exists": False}
        series = dataframe[col]
        req_mask = series.notna()
        total_req = int(req_mask.sum())
        succ = int((series == True).sum())
        fail = int((series == False).sum())
        succ_rate = (succ / total_req * 100.0) if total_req > 0 else np.nan
        return {
            "label": label,
            "exists": True,
            "total_req": total_req,
            "success": succ,
            "fail": fail,
            "success_rate": succ_rate,
        }

    stats = [
        tri_stats(df, "booking_success", "Booking"),
        tri_stats(df, "reschedule_success", "Reschedule"),
        tri_stats(df, "cancel_success", "Cancel"),
    ]

    stats_df = pd.DataFrame(
        [
            {
                "Outcome": s["label"],
                "Requested": s.get("total_req", 0),
                "Success": s.get("success", 0),
                "Fail": s.get("fail", 0),
                "Success % (of requests)": s.get("success_rate", np.nan),
            }
            for s in stats
            if s["exists"]
        ]
    )

    if not stats_df.empty:
        stats_df["Success % (of requests)"] = stats_df["Success % (of requests)"].round(1)
        st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("No tri-state outcome columns found to build the summary table.")

    st.markdown("---")
    st.subheader("Time Series – Calls & Booking Success")

    if has("date") and df["date"].notna().any():
        tmp = df.copy()
        tmp["day"] = tmp["date"].dt.date

        # 1) Total calls per day
        daily_calls = tmp.groupby("day").size().reset_index(name="calls")
        fig_calls = px.line(
            daily_calls,
            x="day",
            y="calls",
            markers=True,
            title="Total Calls per Day",
        )
        fig_calls.update_layout(xaxis_title="Day", yaxis_title="Number of calls")
        st.plotly_chart(fig_calls, use_container_width=True)

        # 2) Booking success % per day (among requested)
        if has("booking_success"):
            tmp_req = tmp[tmp["booking_success"].notna()].copy()
            if not tmp_req.empty:
                daily_rate = (
                    tmp_req.groupby("day")["booking_success"]
                    .mean()
                    .mul(100)
                    .reset_index(name="booking_success_pct")
                )
                fig_rate = px.line(
                    daily_rate,
                    x="day",
                    y="booking_success_pct",
                    markers=True,
                    title="Booking Success % per Day (Requested Only)",
                )
                fig_rate.update_yaxes(range=[0, 100], title="Success rate (%)")
                st.plotly_chart(fig_rate, use_container_width=True)
            else:
                st.info(
                    "`booking_success` present but all values are null – cannot compute daily success rate."
                )
        else:
            st.info("`booking_success` column not found – cannot compute daily success rate.")
    else:
        st.info("No `date` column with non-null values – time series charts are disabled.")

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
            data.append({"KPI": c, "Rate %": df[c].mean() * 100})
    if data:
        kpi_df = pd.DataFrame(data).sort_values("Rate %", ascending=False)
        st.plotly_chart(
            px.bar(
                kpi_df,
                x="KPI",
                y="Rate %",
                text_auto=".1f",
                title="Assistant KPI Rates (%)",
                color="Rate %",
            ),
            use_container_width=True,
        )
    else:
        st.info("No protocol KPI columns found.")

    st.subheader("Booked Details")
    c1, c2 = st.columns(2)
    if has("booked_time") and df["booked_time"].notna().any():
        c1.plotly_chart(
            hist("booked_time", "Distribution of Booked Time (if present)", nbins=30),
            use_container_width=True,
        )
    else:
        c1.info("booked_time column not found or empty.")

    if has("assistant_notes"):
        words = (
            df["assistant_notes"]
            .dropna()
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.split()
        )
        stop = set(
            "the a and to of in for is it this that with on by at from as be or not you i we they he she are was were will would could should".split()
        )
        tokens = [w for lst in words for w in lst if w not in stop and len(w) > 2]
        wc = pd.Series(tokens).value_counts().head(25).reset_index()
        wc.columns = ["word", "count"]
        if not wc.empty:
            c2.plotly_chart(
                px.bar(
                    wc,
                    x="word",
                    y="count",
                    title="Assistant Notes – Top Keywords",
                    text_auto=True,
                ),
                use_container_width=True,
            )
        else:
            c2.info("No keywords found in assistant_notes.")
    else:
        c2.info("assistant_notes column not found.")

# -----------------------------
# 3) Client Struggles (Legacy)
# -----------------------------
with tabs[2]:
    c1, c2 = st.columns(2)
    if has("time_confusion"):
        c1.plotly_chart(
            count_bar(
                df["time_confusion"].map({True: "Yes", False: "No"}),
                "Time Confusion (Legacy)",
            ),
            use_container_width=True,
        )
    else:
        c1.info("time_confusion column not found.")

    if has("payment_confusion"):
        c2.plotly_chart(
            count_bar(
                df["payment_confusion"].map({True: "Yes", False: "No"}),
                "Payment Confusion (Legacy)",
            ),
            use_container_width=True,
        )
    else:
        c2.info("payment_confusion column not found.")

    if has("proactive_pause_gap"):
        st.plotly_chart(
            count_bar(
                df["proactive_pause_gap"].map({True: "Yes", False: "No"}),
                "Proactive Pause / Long Gap",
            ),
            use_container_width=True,
        )
    else:
        st.info("proactive_pause_gap column not found.")

# -----------------------------
# 4) Conversational Quality
# -----------------------------
with tabs[3]:
    c1, c2 = st.columns(2)
    if has("assistant_tone_consistency"):
        if df["assistant_tone_consistency"].notna().any():
            c1.plotly_chart(
                hist("assistant_tone_consistency", "Tone Consistency – Distribution"),
                use_container_width=True,
            )
        else:
            c1.info("`assistant_tone_consistency` is present but all values are null.")
    else:
        c1.info("`assistant_tone_consistency` column not found.")

    if has("empathy_score"):
        if df["empathy_score"].notna().any():
            c2.plotly_chart(
                hist("empathy_score", "Empathy Score – Distribution"),
                use_container_width=True,
            )
        else:
            c2.info("`empathy_score` is present but all values are null.")
    else:
        c2.info("`empathy_score` column not found.")

    c3, c4 = st.columns(2)
    if (
        has("clarity_score")
        and has("empathy_score")
        and df[["clarity_score", "empathy_score"]].notna().any().any()
    ):
        fig = px.scatter(
            df,
            x="clarity_score",
            y="empathy_score",
            color=df["final_sentiment"] if has("final_sentiment") else None,
            title="Clarity vs Empathy",
        )
        c3.plotly_chart(fig, use_container_width=True)
    else:
        c3.info("`clarity_score` and/or `empathy_score` missing or all null.")

    # Adherence over time – use `date` for x-axis
    if has("assistant_adherence_score") and has("date"):
        tmp = df.dropna(subset=["assistant_adherence_score", "date"])
        if not tmp.empty:
            c4.plotly_chart(
                line(
                    "assistant_adherence_score",
                    x="date",
                    title="Adherence Over Time (date)",
                ),
                use_container_width=True,
            )
        else:
            c4.info("`assistant_adherence_score`/`date` present but all rows are null.")
    else:
        c4.info("`assistant_adherence_score` or `date` column not found.")

# -----------------------------
# 5) Outcomes (Booking/Reschedule/Cancel) – Tri-state analytics + Low booking
# -----------------------------
with tabs[4]:
    st.subheader("📦 Outcomes from Tri-State Fields (null | true | false)")
    st.caption(
        "Denominator = all requests where the action was asked (values true/false). Rows with null mean the action was not requested."
    )

    def tri_stats_local(dataframe, col, label):
        if col not in dataframe.columns:
            return {"label": label, "exists": False}
        series = dataframe[col]
        req_mask = series.notna()
        total_req = int(req_mask.sum())
        succ = int((series == True).sum())
        fail = int((series == False).sum())
        succ_rate = (succ / total_req * 100.0) if total_req > 0 else np.nan
        return {
            "label": label,
            "exists": True,
            "total_req": total_req,
            "success": succ,
            "fail": fail,
            "success_rate": succ_rate,
        }

    stats = [
        tri_stats_local(df, "booking_success", "Booking"),
        tri_stats_local(df, "reschedule_success", "Reschedule"),
        tri_stats_local(df, "cancel_success", "Cancel"),
    ]

    for s in stats:
        if not s["exists"]:
            st.info(
                f"Column for **{s['label']}** not found. Add `{s['label'].lower()}_success` tri-state field to view these charts."
            )
        elif s["total_req"] == 0:
            st.info(
                f"No **{s['label']}** requests detected (all values are null). Users did not ask for this action."
            )
        else:
            if s["success"] == 0:
                st.warning(f"**{s['label']}**: 0 successful outcomes. All requests failed.")
            if s["fail"] == 0:
                st.info(f"**{s['label']}**: 0 failures. All requested actions succeeded.")

    stats_df = pd.DataFrame(
        [
            {
                "Outcome": s["label"],
                "Requested": s.get("total_req", 0),
                "Success": s.get("success", 0),
                "Fail": s.get("fail", 0),
                "Success % (of requests)": s.get("success_rate", np.nan),
            }
            for s in stats
            if s["exists"]
        ]
    )

    if not stats_df.empty and stats_df["Requested"].sum() > 0:
        st.markdown("#### ✅ Success Rate (of requested only)")
        fig_rate = px.bar(
            stats_df,
            x="Outcome",
            y="Success % (of requests)",
            text_auto=".1f",
            color="Outcome",
            title="Success % by Outcome (Requested = true|false; null excluded)",
        )
        fig_rate.update_yaxes(range=[0, 100], title="Success rate (%)")
        st.plotly_chart(fig_rate, use_container_width=True)

        st.markdown("#### 🧱 Success vs Fail (counts, requested only)")
        stacked_df = stats_df.melt(
            id_vars=["Outcome", "Requested"],
            value_vars=["Success", "Fail"],
            var_name="Result",
            value_name="Count",
        )
        fig_stack = px.bar(
            stacked_df,
            x="Outcome",
            y="Count",
            color="Result",
            barmode="stack",
            text_auto=True,
            title="Requested Outcomes: Success vs Fail",
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        st.markdown("#### 📋 Summary (Requested Only)")
        show_df = stats_df.copy()
        show_df["Success % (of requests)"] = show_df["Success % (of requests)"].round(1)
        st.dataframe(show_df, use_container_width=True)
    else:
        st.info(
            "No requested outcomes to display yet (all three tri-state fields may be missing or null)."
        )

    st.markdown("---")
    st.subheader("📉 Low Booking Analytics (Reasons & Comments)")

    if has("low_booking_reason"):
        df_low = df.copy()
        # Only where both reason and comment are non-empty
        df_low["low_booking_reason"] = df_low["low_booking_reason"].fillna("").astype(str).str.strip()
        df_low["low_booking_comment"] = df_low["low_booking_comment"].fillna("").astype(str).str.strip()
        df_low = df_low[
            (df_low["low_booking_reason"] != "") & (df_low["low_booking_comment"] != "")
        ]

        if df_low.empty:
            st.info("No rows with low_booking_reason and low_booking_comment filled.")
        else:
            cols = ["conversation_id", "low_booking_reason", "low_booking_comment"]
            if has("date"):
                cols.append("date")
            cols = [c for c in cols if c in df_low.columns]

            st.markdown("##### 📋 Low Booking Reason Table")
            st.dataframe(df_low[cols], use_container_width=True)

            # Over time – use `date` on x-axis
            if has("date"):
                df_low_ts = df_low.dropna(subset=["date"]).copy()
                df_low_ts = df_low_ts.sort_values("date")
                df_low_ts["index_order"] = np.arange(len(df_low_ts))
                hover_cols = [c for c in cols if c != "index_order"]
                fig_low = px.scatter(
                    df_low_ts,
                    x="date",
                    y="index_order",
                    color="low_booking_reason",
                    hover_data=hover_cols,
                    title="Low Booking Reasons Over Time (hover to see reason & comment)",
                )
                fig_low.update_yaxes(visible=False, showticklabels=False, title="")
                st.plotly_chart(fig_low, use_container_width=True)
    else:
        st.info("low_booking_reason column not found.")

# -----------------------------
# 6) Flow & Recovery
# -----------------------------
with tabs[5]:
    c1, c2 = st.columns(2)
    if has("conversation_length_seconds"):
        c1.plotly_chart(
            hist("conversation_length_seconds", "Conversation Length (sec) – Distribution"),
            use_container_width=True,
        )
    else:
        c1.info("conversation_length_seconds column not found.")

    if has("assistant_recovery_actions"):
        words = (
            df["assistant_recovery_actions"]
            .dropna()
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.split()
        )
        stop = set(
            "the a and to of in for is it this that with on by at from as be or not you i we they he she are was were will would could should".split()
        )
        tokens = [w for lst in words for w in lst if w not in stop and len(w) > 2]
        wc = pd.Series(tokens).value_counts().head(25).reset_index()
        wc.columns = ["word", "count"]
        if not wc.empty:
            c2.plotly_chart(
                px.bar(
                    wc,
                    x="word",
                    y="count",
                    title="Assistant Recovery – Top Keywords",
                    text_auto=True,
                ),
                use_container_width=True,
            )
        else:
            c2.info("No frequent tokens found in assistant_recovery_actions.")
    else:
        c2.info("assistant_recovery_actions column not found.")

    c3, c4 = st.columns(2)
    for col_name, label, container in [
        ("assistant_repetition_count", "Assistant Repetition Count", c3),
        ("assistant_clarification_count", "Assistant Clarification Count", c4),
    ]:
        if has(col_name):
            container.plotly_chart(
                hist(col_name, f"{label} – Distribution"),
                use_container_width=True,
            )
        else:
            container.info(f"{col_name} column not found.")

    c5, c6 = st.columns(2)
    if has("user_clarification_count"):
        c5.plotly_chart(
            hist("user_clarification_count", "User Clarification Count – Distribution"),
            use_container_width=True,
        )
    else:
        c5.info("user_clarification_count column not found.")

    if has("repeated_user_input"):
        c6.plotly_chart(
            count_bar(
                df["repeated_user_input"].map({True: "Yes", False: "No"}),
                "Repeated User Input",
            ),
            use_container_width=True,
        )
    else:
        c6.info("repeated_user_input column not found.")

    # Long pause distributions
    c7, c8 = st.columns(2)
    if has("user_long_pause_count"):
        if df["user_long_pause_count"].notna().any():
            c7.plotly_chart(
                hist("user_long_pause_count", "User Long Pause Count – Distribution"),
                use_container_width=True,
            )
        else:
            c7.info("user_long_pause_count present but all values are null/zero.")
    else:
        c7.info("user_long_pause_count column not found.")

    if has("assistant_long_pause_count"):
        if df["assistant_long_pause_count"].notna().any():
            c8.plotly_chart(
                hist("assistant_long_pause_count", "Assistant Long Pause Count – Distribution"),
                use_container_width=True,
            )
        else:
            c8.info("assistant_long_pause_count present but all values are null/zero.")
    else:
        c8.info("assistant_long_pause_count column not found.")

    # Pause pattern types – TOP 5 (user + assistant)
    st.subheader("⏸ Pause Types – Top 5 (User vs Assistant)")
    up = parse_pause_patterns("user_pause_pattern", "User")
    ap = parse_pause_patterns("assistant_pause_pattern", "Assistant")
    pp_all = (
        pd.concat([up, ap], ignore_index=True)
        if (not up.empty or not ap.empty)
        else pd.DataFrame()
    )

    if not pp_all.empty:
        agg = pp_all.groupby(["side", "pause_type"])["count"].sum().reset_index()
        agg_top = agg.sort_values("count", ascending=False).groupby("side").head(5)
        fig_pause = px.bar(
            agg_top,
            x="pause_type",
            y="count",
            color="side",
            barmode="group",
            text_auto=True,
            title="Top 5 Pause Types by Side (Total Counts)",
        )
        fig_pause.update_xaxes(title="Pause type (parsed from pattern)")
        fig_pause.update_yaxes(title="Total count")
        st.plotly_chart(fig_pause, use_container_width=True)
        st.caption(
            "Hint: If counts are zero or no bars appear, it means pause_pattern fields are empty or all zero for this slice."
        )
    else:
        st.info(
            "No pause_pattern data found for user or assistant. (All patterns may be empty for this slice.)"
        )

# -----------------------------
# 7) Struggling Areas (New)
# -----------------------------
with tabs[6]:
    c1, c2 = st.columns(2)
    if has("user_struggle_present"):
        c1.plotly_chart(
            count_bar(
                df["user_struggle_present"].map({True: "Yes", False: "No"}),
                "User Struggle Present",
            ),
            use_container_width=True,
        )
    else:
        c1.info("user_struggle_present column not found.")

    if has("assistant_struggle_present"):
        c2.plotly_chart(
            count_bar(
                df["assistant_struggle_present"].map({True: "Yes", False: "No"}),
                "Assistant Struggle Present",
            ),
            use_container_width=True,
        )
    else:
        c2.info("assistant_struggle_present column not found.")

    st.subheader("User Struggle Tags")
    utags = parse_tags("user_struggle_tags")
    if not utags.empty:
        st.plotly_chart(
            count_bar(utags["tag"], "User Struggles by Tag", xlab="Tag"),
            use_container_width=True,
        )
        uco = cooccurrence("user_struggle_tags")
        if not uco.empty:
            st.plotly_chart(
                heatmap_from_matrix(uco, "User Struggle Tag Co-occurrence"),
                use_container_width=True,
            )
    else:
        st.info("user_struggle_tags column missing or empty.")

    st.subheader("Assistant Struggle Tags")
    atags = parse_tags("assistant_struggle_tags")
    if not atags.empty:
        st.plotly_chart(
            count_bar(atags["tag"], "Assistant Struggles by Tag", xlab="Tag"),
            use_container_width=True,
        )
        aco = cooccurrence("assistant_struggle_tags")
        if not aco.empty:
            st.plotly_chart(
                heatmap_from_matrix(aco, "Assistant Struggle Tag Co-occurrence"),
                use_container_width=True,
            )
    else:
        st.info("assistant_struggle_tags column missing or empty.")

    if has("user_confusion_type") and has("resolution_status"):
        pivot = pd.crosstab(
            df["user_confusion_type"].fillna("none"),
            df["resolution_status"].fillna("unresolved"),
        )
        st.plotly_chart(
            px.imshow(
                pivot,
                text_auto=True,
                color_continuous_scale="Blues",
                title="User Confusion Type × Resolution Status",
            ),
            use_container_width=True,
        )
    else:
        st.info("user_confusion_type or resolution_status column not found.")

    st.subheader("Unavailable Requested Items (Denied Services)")
    uitags = parse_tags("unavailable_requested_items")
    if not uitags.empty:
        st.plotly_chart(
            count_bar(
                uitags["tag"],
                "Unavailable Requested Items",
                xlab="Item",
            ),
            use_container_width=True,
        )
        uic = uitags["tag"].value_counts().reset_index()
        uic.columns = ["item", "count"]
        st.markdown("##### 📋 Unavailable Items Table")
        st.dataframe(uic, use_container_width=True)
    else:
        st.info("unavailable_requested_items column missing or empty.")

# -----------------------------
# 8) Signals & Metadata
# -----------------------------
with tabs[7]:
    c1, c2, c3 = st.columns(3)
    if has("language_switch_detected"):
        c1.plotly_chart(
            count_bar(
                df["language_switch_detected"].map({True: "Yes", False: "No"}),
                "Language Switch Detected",
            ),
            use_container_width=True,
        )
    else:
        c1.info("language_switch_detected column not found.")

    if has("function_error_present"):
        c2.plotly_chart(
            count_bar(
                df["function_error_present"].map({True: "Yes", False: "No"}),
                "Function Error Present",
            ),
            use_container_width=True,
        )
    else:
        c2.info("function_error_present column not found.")

    if has("service_not_offered_by_selected_barber"):
        c3.plotly_chart(
            count_bar(
                df["service_not_offered_by_selected_barber"].map({True: "Yes", False: "No"}),
                "Service Not Offered by Selected Barber",
            ),
            use_container_width=True,
        )
    else:
        c3.info("service_not_offered_by_selected_barber column not found.")

    # Analyses per day – use `date` if available
    if has("date") and df["date"].notna().any():
        tmp = df.dropna(subset=["date"]).copy()
        tmp["day"] = tmp["date"].dt.date
        daily = tmp.groupby("day").size().reset_index(name="count")
        st.plotly_chart(
            px.line(
                daily,
                x="day",
                y="count",
                title="Analyses per Day",
                markers=True,
            ),
            use_container_width=True,
        )
    else:
        st.info("No usable `date` column found for time series of analyses per day.")

    # Languages detected
    st.subheader("Languages Detected")
    lang_tags = parse_tags("languages_detected")
    if not lang_tags.empty:
        st.plotly_chart(
            count_bar(lang_tags["tag"], "Languages Detected", xlab="Language"),
            use_container_width=True,
        )
    else:
        st.info("languages_detected column missing or empty.")

    if has("model_source"):
        st.plotly_chart(
            count_bar(df["model_source"], "Model Source Share"),
            use_container_width=True,
        )
    else:
        st.info("model_source column not found.")

# -----------------------------
# 9) Correlations
# -----------------------------
with tabs[8]:
    st.subheader("Correlation Matrix – Core Numeric Metrics")
    num_cols = [
        c
        for c in [
            "assistant_adherence_score",
            "assistant_tone_consistency",
            "empathy_score",
            "clarity_score",
            "assistant_repetition_count",
            "assistant_clarification_count",
            "user_clarification_count",
            "conversation_length_seconds",
            "user_long_pause_count",
            "assistant_long_pause_count",
        ]
        if has(c)
    ]
    if num_cols:
        corr = df[num_cols].corr(numeric_only=True)
        st.plotly_chart(
            px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Correlation Heatmap",
            ),
            use_container_width=True,
        )
    else:
        st.info("No numeric columns available for correlation heatmap.")

    st.subheader("Booking Success by Factors")
    if has("booking_success"):
        cat_factors = [
            c
            for c in [
                "final_sentiment",
                "user_confusion_type",
                "resolution_status",
                "booked_service",
                "booked_barber",
            ]
            if has(c)
        ]
        for c in cat_factors:
            tmp = df.groupby(c)["booking_success"].mean().mul(100).reset_index().dropna()
            if not tmp.empty:
                st.plotly_chart(
                    px.bar(
                        tmp.sort_values("booking_success", ascending=False),
                        x=c,
                        y="booking_success",
                        text_auto=True,
                        title=f"Booking Success by {c} (%)",
                        color="booking_success",
                    ),
                    use_container_width=True,
                )
            else:
                st.info(f"No data to compute Booking Success by {c}.")
    else:
        st.info("booking_success column not found.")

# -----------------------------
# 10) Raw Data
# -----------------------------
with tabs[9]:
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download Filtered Data (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="conversation_analysis_filtered.csv",
        mime="text/csv",
    )
