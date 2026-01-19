# -*- coding: utf-8 -*-
import os
import io
from pathlib import Path
from datetime import date, timedelta
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import numpy as np
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest, Dimension, Metric, Filter, FilterExpression, FilterExpressionList, OrderBy
)

st.set_page_config(page_title="GA4 Professional Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        background-color: #0f172a;
        color: white;
        border: none;
        padding: 0.6rem;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #000000;
        color: white;
        transform: translateY(-1px);
    }
    .metric-container {
        background-color: white;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stExpander"] {
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

SCOPES = ["https://www.googleapis.com/auth/analytics.readonly"]

DASH_LOGO = st.secrets.get("DASH_LOGO", os.getenv("DASH_LOGO", ""))
SIDEBAR_LOGO = st.secrets.get("SIDEBAR_LOGO", os.getenv("SIDEBAR_LOGO", ""))

INVISIBLE = ("\ufeff", "\u200b", "\u2060", "\u00a0")
DROP_QUERY_KEYS = {
    "gclid", "fbclid", "yclid", "mc_cid", "mc_eid", "igshid", "ref", "ref_src",
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"
}

def fail_ui(msg: str):
    st.error(msg)
    st.stop()

def password_gate():
    app_pwd = str(st.secrets.get("APP_PASSWORD", "")).strip()
    if not app_pwd:
        return
    if st.session_state.get("authed"):
        return
    st.title("Вход")
    pwd = st.text_input("Пароль", type="password")
    if pwd and pwd == app_pwd:
        st.session_state["authed"] = True
        st.rerun()
    st.stop()

@st.cache_resource
def ga_client() -> BetaAnalyticsDataClient:
    sa = st.secrets.get("gcp_service_account")
    if not sa:
        fail_ui("Не найден секрет **gcp_service_account** в Streamlit Secrets.")
    sa_info = dict(sa)
    creds = service_account.Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    return BetaAnalyticsDataClient(credentials=creds)

def default_property_id() -> str:
    pid = str(st.secrets.get("GA4_PROPERTY_ID", "")).strip()
    if not pid:
        fail_ui("Не задан секрет **GA4_PROPERTY_ID**.")
    return pid

def clean_line(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    for ch in INVISIBLE:
        s = s.replace(ch, "")
    return s.strip()

def normalize_url_remove_tracking(raw_url: str) -> str:
    p = urlparse(raw_url)
    q = []
    for k, v in parse_qsl(p.query, keep_blank_values=True):
        kl = k.lower()
        if kl.startswith("utm_"):
            continue
        if kl in DROP_QUERY_KEYS:
            continue
        q.append((k, v))
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), ""))

def url_to_path_host(u: str) -> tuple[str, str | None]:
    s = clean_line(u)
    if not s:
        return "", None
    if s.lower().startswith(("http://", "https://")):
        s2 = normalize_url_remove_tracking(s)
        p = urlparse(s2)
        path = p.path or "/"
        host = p.hostname or None
        return path, host
    if not s.startswith("/"):
        s = "/" + s
    return s, None

def collect_paths_hosts(raw_list: list[str]) -> tuple[list[str], list[str], list[str]]:
    seen = set()
    unique_paths: list[str] = []
    hosts = set()
    order_list: list[str] = []
    for raw in raw_list:
        path, host = url_to_path_host(raw)
        if not path:
            continue
        order_list.append(path)
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
        if host:
            hosts.add(host)
    return unique_paths, sorted(hosts), order_list

def render_logo(path: str, width: int | None = None):
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    ext = p.suffix.lower()
    if ext == ".svg":
        try:
            from urllib.parse import quote
            svg_txt = p.read_text(encoding="utf-8")
            data_uri = f"data:image/svg+xml;utf8,{quote(svg_txt)}"
            w_attr = f' style="width:{width}px;"' if width else ""
            st.markdown(f'<img src="{data_uri}"{w_attr}>', unsafe_allow_html=True)
        except Exception:
            return
    else:
        st.image(str(p), use_column_width=width is None, width=width)

def make_path_filter(paths_batch: list[str], match_type: str) -> FilterExpression:
    mt = Filter.StringFilter.MatchType.EXACT
    if match_type == "begins_with":
        mt = Filter.StringFilter.MatchType.BEGINS_WITH
    elif match_type == "contains":
        mt = Filter.StringFilter.MatchType.CONTAINS

    exprs = [
        FilterExpression(
            filter=Filter(
                field_name="pagePath",
                string_filter=Filter.StringFilter(
                    value=pth,
                    match_type=mt,
                    case_sensitive=False,
                )
            )
        )
        for pth in paths_batch
    ]
    return FilterExpression(or_group=FilterExpressionList(expressions=exprs))

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ga4_by_identifiers_cached(
    property_id: str,
    identifiers: tuple[str, ...],
    hosts: tuple[str, ...],
    start_date: str,
    end_date: str,
    order_keys: tuple[str, ...],
    mode: str,
    path_match: str,
) -> pd.DataFrame:
    client = ga_client()
    rows, BATCH = [], 25

    metrics = [
        Metric(name="screenPageViews"),
        Metric(name="activeUsers"),
        Metric(name="averageEngagementTime"),
    ]

    id_list = list(identifiers)
    host_list = list(hosts)
    key_col = "pagePath" if mode == "path" else "pageLocation"

    for i in range(0, len(id_list), BATCH):
        batch = id_list[i:i + BATCH]

        if mode == "path":
            base = make_path_filter(batch, path_match)
            dim_filter = base

            dims = [Dimension(name="pagePath"), Dimension(name="pageTitle")]
            if host_list:
                host_expr = FilterExpression(
                    filter=Filter(
                        field_name="hostName",
                        in_list_filter=Filter.InListFilter(values=host_list[:50])
                    )
                )
                dim_filter = FilterExpression(
                    and_group=FilterExpressionList(expressions=[base, host_expr])
                )
                dims.append(Dimension(name="hostName"))

            req = RunReportRequest(
                property=f"properties/{property_id}",
                dimensions=dims,
                metrics=metrics,
                date_ranges=[{"start_date": start_date, "end_date": end_date}],
                dimension_filter=dim_filter,
                limit=100000,
            )

        else:
            dims = [Dimension(name="pageLocation"), Dimension(name="pageTitle")]
            req = RunReportRequest(
                property=f"properties/{property_id}",
                dimensions=dims,
                metrics=metrics,
                date_ranges=[{"start_date": start_date, "end_date": end_date}],
                dimension_filter=FilterExpression(
                    filter=Filter(
                        field_name="pageLocation",
                        in_list_filter=Filter.InListFilter(values=batch)
                    )
                ),
                limit=100000,
            )

        resp = client.run_report(req)
        for r in resp.rows:
            rec = {}
            idx = 0
            rec[key_col] = r.dimension_values[idx].value
            idx += 1
            rec["pageTitle"] = r.dimension_values[idx].value
            idx += 1
            if mode == "path" and host_list:
                rec["hostName"] = r.dimension_values[idx].value

            rec["screenPageViews"] = float(r.metric_values[0].value or 0)
            rec["activeUsers"] = float(r.metric_values[1].value or 0)
            rec["averageEngagementTime"] = float(r.metric_values[2].value or 0)
            rows.append(rec)

    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame(columns=[key_col, "pageTitle", "screenPageViews", "activeUsers", "averageEngagementTime"])
    else:
        for c in ["screenPageViews", "activeUsers", "averageEngagementTime"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        agg = {
            "screenPageViews": "sum",
            "activeUsers": "sum",
            "averageEngagementTime": "mean",
            "pageTitle": "first",
        }
        if "hostName" in df.columns:
            agg["hostName"] = "first"

        df = df.groupby([key_col], as_index=False).agg(agg)

    present = set(df[key_col].tolist()) if not df.empty else set()
    missing_unique = [p for p in id_list if p not in present]
    if missing_unique:
        base_zero = {
            key_col: None,
            "pageTitle": "",
            "screenPageViews": 0,
            "activeUsers": 0,
            "averageEngagementTime": 0,
        }
        zeros = pd.DataFrame([dict(base_zero, **{key_col: p}) for p in missing_unique])
        if "hostName" in df.columns:
            zeros["hostName"] = host_list[0] if host_list else ""
        df = pd.concat([df, zeros], ignore_index=True)

    df = df.set_index(key_col).reindex(list(order_keys)).reset_index().rename(columns={"index": key_col})
    df["pageTitle"] = df.get("pageTitle", "").fillna("")

    den = pd.to_numeric(df["activeUsers"], errors="coerce").replace(0, np.nan).astype(float)
    df["viewsPerActiveUser"] = (
        pd.to_numeric(df["screenPageViews"], errors="coerce").astype(float).div(den)
    ).fillna(0).round(2)

    df["averageEngagementTime"] = pd.to_numeric(df["averageEngagementTime"], errors="coerce").fillna(0).round(1)

    df["screenPageViews"] = pd.to_numeric(df["screenPageViews"], errors="coerce").fillna(0).astype(int)
    df["activeUsers"] = pd.to_numeric(df["activeUsers"], errors="coerce").fillna(0).astype(int)

    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_top_materials_cached(property_id: str, start_date: str, end_date: str, limit: int) -> pd.DataFrame:
    client = ga_client()
    req = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="pagePath"), Dimension(name="pageTitle")],
        metrics=[Metric(name="screenPageViews"), Metric(name="activeUsers"), Metric(name="averageEngagementTime")],
        date_ranges=[{"start_date": start_date, "end_date": end_date}],
        order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"), desc=True)],
        limit=int(limit),
    )
    resp = client.run_report(req)
    rows = []
    for r in resp.rows:
        rows.append({
            "Path": r.dimension_values[0].value,
            "Title": r.dimension_values[1].value,
            "Views": int(float(r.metric_values[0].value or 0)),
            "Active Users": int(float(r.metric_values[1].value or 0)),
            "Avg Engagement Time (s)": round(float(r.metric_values[2].value or 0), 1),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_site_totals_cached(property_id: str, start_date: str, end_date: str) -> tuple[int, int, int]:
    client = ga_client()
    req = RunReportRequest(
        property=f"properties/{property_id}",
        metrics=[Metric(name="sessions"), Metric(name="totalUsers"), Metric(name="screenPageViews")],
        date_ranges=[{"start_date": start_date, "end_date": end_date}],
        limit=1,
    )
    resp = client.run_report(req)
    if not resp.rows:
        return 0, 0, 0
    vals = resp.rows[0].metric_values
    return int(float(vals[0].value or 0)), int(float(vals[1].value or 0)), int(float(vals[2].value or 0))

def read_uploaded_lines(uploaded) -> list[str]:
    if uploaded is None:
        return []
    name = (uploaded.name or "").lower()
    if name.endswith(".txt"):
        raw = uploaded.getvalue()
        txt = raw.decode("utf-8", errors="ignore")
        return [clean_line(x) for x in txt.splitlines() if clean_line(x)]
    if name.endswith(".csv"):
        dfu = pd.read_csv(uploaded, header=None)
        col = dfu.iloc[:, 0].astype(str).tolist()
        return [clean_line(x) for x in col if clean_line(x)]
    return []

password_gate()

client = ga_client()
prop_id_default = default_property_id()

with st.sidebar:
    st.markdown("### Reporting Period")
    today = date.today()
    date_from = st.date_input("Date From", value=today - timedelta(days=30))
    date_to = st.date_input("Date To", value=today)

    st.divider()
    st.markdown("### Property")
    property_id = st.text_input("GA4 Property ID", value=prop_id_default)

    st.divider()
    st.markdown("### Developed by")
    st.markdown("**Alexey Terekhov**")
    st.markdown("[terekhov.digital@gmail.com](mailto:terekhov.digital@gmail.com)")

    if SIDEBAR_LOGO:
        st.markdown("<br>", unsafe_allow_html=True)
        render_logo(SIDEBAR_LOGO, width=160)

head_col1, head_col2 = st.columns([4, 1])
with head_col1:
    st.title("Analytics Console")
    st.markdown("Professional content performance and user engagement reporting.")
with head_col2:
    st.image("https://www.gstatic.com/analytics-suite/header/suite/v2/ic_analytics.svg", width=80)

st.divider()

tab1, tab2, tab3 = st.tabs(["Batch URL Analysis", "Top Materials", "Global Performance"])

with tab1:
    st.subheader("Batch Processing")

    smode = st.radio("Extraction Logic", ["By Path", "By URL"], horizontal=True)
    mode_key = "path" if smode == "By Path" else "url"

    path_match = "begins_with"
    if mode_key == "path":
        path_match_ui = st.radio(
            "Path matching",
            ["Begins with", "Exact", "Contains"],
            horizontal=True
        )
        path_match = {"Begins with": "begins_with", "Exact": "exact", "Contains": "contains"}[path_match_ui]

    cA, cB = st.columns([3, 2])
    with cA:
        uinput = st.text_area(
            "Identifiers (one per line)",
            height=200,
            placeholder="/content/news-item\nhttps://domain.com/page"
        )
    with cB:
        uploaded = st.file_uploader("Or upload .txt/.csv (1 per line)", type=["txt", "csv"])

    lines = []
    if uinput:
        lines.extend([clean_line(x) for x in uinput.splitlines() if clean_line(x)])
    lines.extend(read_uploaded_lines(uploaded))

    if mode_key == "path":
        page_paths, hostnames, order_keys = collect_paths_hosts(lines)
        st.caption(
            f"Lines: {len(lines)} | Unique paths: {len(page_paths)} | "
            f"Hosts: {', '.join(hostnames) if hostnames else '—'}"
        )
        identifiers = page_paths
        hosts = hostnames
    else:
        urls_clean = []
        seen = set()
        for u in lines:
            if not u.lower().startswith(("http://", "https://")):
                continue
            u2 = normalize_url_remove_tracking(u)
            if u2 not in seen:
                seen.add(u2)
                urls_clean.append(u2)
        st.caption(f"URLs: {len(urls_clean)}")
        identifiers = urls_clean
        hosts = []
        order_keys = urls_clean

    if st.button("Execute Analysis"):
        if date_from > date_to:
            fail_ui("Date From must be <= Date To.")
        if not property_id.strip():
            fail_ui("GA4 Property ID is empty.")
        if not identifiers:
            fail_ui("Input required: add at least one identifier.")

        with st.spinner("Fetching data..."):
            df_batch = fetch_ga4_by_identifiers_cached(
                property_id=property_id.strip(),
                identifiers=tuple(identifiers),
                hosts=tuple(hosts),
                start_date=str(date_from),
                end_date=str(date_to),
                order_keys=tuple(order_keys),
                mode=mode_key,
                path_match=path_match,
            )

        if not df_batch.empty:
            st.success(f"Processed {len(df_batch)} rows.")
            key_col = "pagePath" if mode_key == "path" else "pageLocation"

            show_cols = [key_col, "pageTitle", "screenPageViews", "activeUsers", "viewsPerActiveUser", "averageEngagementTime"]
            show = df_batch.reindex(columns=[c for c in show_cols if c in df_batch.columns]).rename(columns={
                key_col: "Identifier",
                "pageTitle": "Title",
                "screenPageViews": "Views",
                "activeUsers": "Active Users",
                "viewsPerActiveUser": "Views / Active User",
                "averageEngagementTime": "Avg Engagement Time (s)",
            })

            st.dataframe(show, use_container_width=True, hide_index=True)

            tot_views = int(pd.to_numeric(df_batch["screenPageViews"], errors="coerce").sum())
            tot_users = int(pd.to_numeric(df_batch["activeUsers"], errors="coerce").sum())
            ratio = (tot_views / max(tot_users, 1))
            avg_eng = float(pd.to_numeric(df_batch["averageEngagementTime"], errors="coerce").mean()) if len(df_batch) else 0.0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Views", f"{tot_views:,}")
            k2.metric("Active Users", f"{tot_users:,}")
            k3.metric("Views / Active User", f"{ratio:.2f}")
            k4.metric("Avg Engagement Time (s)", f"{avg_eng:.1f}")

            st.download_button(
                "Export Results (CSV)",
                show.to_csv(index=False).encode("utf-8"),
                "ga4_batch_report.csv",
                "text/csv"
            )
        else:
            st.info("No data returned for these identifiers.")

with tab2:
    st.subheader("High-Performance Content")
    c1, c2 = st.columns([1, 2])
    with c1:
        limit = st.number_input("Limit", min_value=1, max_value=500, value=10)

    if st.button("Extract Top Content"):
        if date_from > date_to:
            fail_ui("Date From must be <= Date To.")
        if not property_id.strip():
            fail_ui("GA4 Property ID is empty.")

        with st.spinner(f"Extracting top {int(limit)} materials..."):
            df_top = fetch_top_materials_cached(property_id.strip(), str(date_from), str(date_to), int(limit))

        if df_top.empty:
            st.info("No data returned for this period.")
        else:
            st.dataframe(df_top, use_container_width=True, hide_index=True)
            st.download_button(
                "Export Ranking (CSV)",
                df_top.to_csv(index=False).encode("utf-8"),
                "ga4_top.csv",
                "text/csv"
            )

with tab3:
    st.subheader("Global Site Summary")
    if st.button("Refresh Site Totals"):
        if date_from > date_to:
            fail_ui("Date From must be <= Date To.")
        if not property_id.strip():
            fail_ui("GA4 Property ID is empty.")

        with st.spinner("Aggregating..."):
            s, u, v = fetch_site_totals_cached(property_id.strip(), str(date_from), str(date_to))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Sessions", f"{s:,}")
        with c2:
            st.metric("Total Users", f"{u:,}")
        with c3:
            st.metric("Page Views", f"{v:,}")
