import streamlit as st
import pandas as pd
from io import BytesIO

# ============================================================
# IMPORT ENGINE FUNCTIONS
# ============================================================

from reconciliation_engine import (
    clean_csgmap,
    clean_partner_sheet,
    reconcile_sheet,
    match_ahan_transactions,
    consolidate_reconciliation_results,
    REQUIRED_COLUMNS,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MojoPay Reconciliation Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME-AWARE STYLES
# ============================================================

st.markdown("""
    <style>
        .stApp {
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3 {
            color: var(--text-color);
        }
        .stButton > button {
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
            border-radius: 6px;
        }
        .stDownloadButton > button {
            background-color: var(--primary-color);
            color: white;
            border-radius: 4px;
        }
        .status-card {
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 1rem;
            background-color: var(--secondary-background-color);
            box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

AMOUNT_COLUMN = REQUIRED_COLUMNS["amount"]
STATUS_COLUMN = REQUIRED_COLUMNS["status"]
REFERENCE_COLUMN = REQUIRED_COLUMNS["reference"]
DATE_COLUMN = REQUIRED_COLUMNS["date"]


def download_buttons(df, filename_prefix, key_prefix):
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇ Download CSV",
        data=csv_data,
        file_name=f"{filename_prefix}.csv",
        mime="text/csv",
        key=f"{key_prefix}-csv",
    )

    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    st.download_button(
        label="⬇ Download Excel",
        data=excel_buffer.getvalue(),
        file_name=f"{filename_prefix}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"{key_prefix}-xlsx",
    )


def render_dataframe_section(df, title, filename_prefix, key_prefix):
    with st.expander(f"{title} — {len(df)} rows"):
        st.dataframe(df, width="stretch")
        download_buttons(df, filename_prefix, key_prefix)

# ============================================================
# SIDEBAR — DYNAMIC STATUS PANEL
# ============================================================

def sidebar_status():
    with st.sidebar:
        st.image("https://mojo-pay.com/images/mojoLogo.svg", width=150)
        if st.session_state.get("results"):
            st.write("**Latest Run**")
            res = st.session_state["results"]
            st.write(f"Reconciled: {len(res.get('reconciled', []))}")
            st.write(f"Conflicts: {len(res.get('status_conflicts', []))}")
        st.caption("MojoPay • Automated Financial Data Integrity Engine")

# Call sidebar
sidebar_status()

# ============================================================
# UNIVERSAL FILE LOADER
# ============================================================

def load_file(file):
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file, encoding="utf-8-sig")
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        st.error(f"❌ Failed to load `{file.name}` — {e}")
        return None

def detect_partner_name(filename):
    name = filename.lower()
    for key in ["telecel", "ahan", "dd", "ep", "mtn", "ppt", "westom"]:
        if key in name:
            return key.upper()
    return "UNKNOWN"

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "results" not in st.session_state:
    st.session_state["results"] = {}

# ============================================================
# MAIN APP TABS
# ============================================================

tabs = st.tabs(["📁 Upload Files", "🧼 Cleaning", "🔗 Reconciliation", "📦 Results"])

# ============================================================
# TAB 1 — UPLOAD FILES
# ============================================================

with tabs[0]:
    st.header("📁 Upload Required Files")

    st.session_state["step_status"] = "Awaiting File Upload"

    st.subheader("🔹 CSGMAP File")
    csgmap_file = st.file_uploader("Main CSGMAP File (CSV/XLSX/XLS)", type=["csv", "xlsx", "xls"])
    if csgmap_file:
        st.success(f"✅ Uploaded: `{csgmap_file.name}`")
        st.session_state["csgmap_file"] = csgmap_file

    st.subheader("🔹 Partner Files")
    partner_files = st.file_uploader("Upload Partner Files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    if partner_files:
        st.success(f"✅ {len(partner_files)} Partner Files Uploaded")
        st.session_state["partner_files"] = partner_files

        with st.expander("🔍 File Summary"):
            for f in partner_files:
                st.markdown(f"• **{f.name}** → `{detect_partner_name(f.name)}`")
    st.session_state["step_status"] = "Files Uploaded"

# ============================================================
# TAB 2 — CLEANING
# ============================================================

with tabs[1]:
    st.header("🧼 Clean & Validate")

    if not st.session_state.get("csgmap_file") or not st.session_state.get("partner_files"):
        st.warning("⚠️ Please upload both CSGMAP and Partner files in Tab 1.")
        st.stop()

    st.session_state["step_status"] = "Cleaning Files"

    # Clean CSGMAP
    st.subheader("🔍 CSGMAP File")
    csgmap_df = load_file(st.session_state["csgmap_file"])
    csgmap_clean, csgmap_dupes, csgmap_summary, csgmap_diag = clean_csgmap(csgmap_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Total Rows", len(csgmap_df))
    col2.metric("✅ Cleaned Rows", csgmap_summary.get("valid", 0))
    col3.metric("⚠️ Duplicates", len(csgmap_dupes))

    with st.expander("📋 Duplicate Records"):
        st.dataframe(csgmap_dupes, width="stretch")

    # Store cleaned CSGMAP
    st.session_state["csgmap_clean"] = csgmap_clean

    st.divider()
    st.subheader("🔍 Partner Files")

    cleaned_partners = {}
    for f in st.session_state["partner_files"]:
        pname = detect_partner_name(f.name)
        df = load_file(f)
        cleaned, dupes, summary, diag = clean_partner_sheet(df, pname)
        cleaned_partners[pname] = cleaned

        st.markdown(f"### 🧩 Partner: `{pname}`")
        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Total Rows", len(df))
        col2.metric("✅ Cleaned Rows", summary.get("valid", 0))
        col3.metric("⚠️ Duplicates", len(dupes))

        with st.expander(f"🔍 Duplicates — {pname}"):
            st.dataframe(dupes, width="stretch")

    st.session_state["cleaned_partners"] = cleaned_partners
    st.session_state["step_status"] = "Cleaning Complete"

# ============================================================
# TAB 3 — RECONCILIATION
# ============================================================

with tabs[2]:
    st.header("🔗 Reconciliation Process")

    if "csgmap_clean" not in st.session_state or "cleaned_partners" not in st.session_state:
        st.warning("⚠️ Please complete cleaning in Tab 2.")
        st.stop()

    st.session_state["step_status"] = "Running Reconciliation"

    rec_list = []
    conflict_list = []
    partner_unmatch = []
    partner_reports = []
    csgmap_matched_views = []
    csgmap_non_success_views = []

    csgmap_remaining = st.session_state["csgmap_clean"].copy()

    for pname, df in st.session_state["cleaned_partners"].items():
        if pname == "AHAN":
            continue
        with st.spinner(f"Reconciling {pname}..."):
            (
                rec,
                conf,
                pun,
                cun,
                partner_report,
                csgmap_matched_view,
                csgmap_non_success_view,
            ) = reconcile_sheet(csgmap_remaining, df, pname)
        rec_list.append(rec)
        conflict_list.append(conf)
        partner_unmatch.append(pun)
        partner_reports.append(partner_report)
        csgmap_matched_views.append(csgmap_matched_view)
        csgmap_non_success_views.append(csgmap_non_success_view)
        csgmap_remaining = cun
        st.success(f"✅ {pname} Reconciled")

    # AHAN matching
    if "AHAN" in st.session_state["cleaned_partners"]:
        with st.spinner("Matching AHAN..."):
            ahan_match, ahan_amb, ahan_unmatch, csgmap_remaining = match_ahan_transactions(
                csgmap_remaining,
                st.session_state["cleaned_partners"]["AHAN"]
            )
        st.success("✅ AHAN Matched")
    else:
        ahan_match = ahan_amb = ahan_unmatch = pd.DataFrame()
        st.info("No AHAN file uploaded.")

    results = consolidate_reconciliation_results(
        rec_list,
        conflict_list,
        partner_unmatch,
        csgmap_remaining,
        ahan_match,
        ahan_amb,
        ahan_unmatch,
        partner_reports=partner_reports,
        csgmap_matched_views=csgmap_matched_views,
        csgmap_non_success_views=csgmap_non_success_views,
    )

    st.session_state["results"] = results
    st.success("🎉 Reconciliation Complete!")
    st.session_state["step_status"] = "Reconciliation Done"

# ============================================================
# TAB 4 — RESULTS
# ============================================================

with tabs[3]:
    st.header("📦 Final Reconciliation Results")

    if not st.session_state["results"]:
        st.info("⚠️ Run reconciliation first.")
        st.stop()

    results = st.session_state["results"]

    st.subheader("📊 Category Totals")
    category_rows = []
    for key, df in results.items():
        if key in {"partner_reports", "csgmap_summary"}:
            continue
        amount_total = df[AMOUNT_COLUMN].sum() if AMOUNT_COLUMN in df.columns else None
        category_rows.append({
            "Category": key.replace("_", " ").title(),
            "Rows": len(df),
            "Total Amount": amount_total,
        })
    if category_rows:
        st.dataframe(pd.DataFrame(category_rows), width="stretch")

    partner_reports = results.get("partner_reports", [])
    if partner_reports:
        st.subheader("🤝 Partner Reconciliation Summaries")
        for idx, report in enumerate(partner_reports, start=1):
            pname = report.get("partner", "UNKNOWN")
            st.markdown(f"### Partner: `{pname}`")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Matched Transactions", report.get("matched_total", 0))
            m2.metric("Unmatched Transactions", report.get("not_matched_total", 0))
            m3.metric(
                "Partner Status ≠ SUCCESS",
                report.get("matched_partner_non_success_status", 0),
            )
            m4.metric(
                "CSGMAP Status ≠ SUCCESS",
                report.get("matched_csgmap_non_success_status", 0),
            )
            m5.metric(
                "Amount Mismatches",
                report.get("amount_mismatch_total", 0),
            )

            render_dataframe_section(
                report.get("df_matched", pd.DataFrame()),
                "Matched Transactions",
                f"{pname}_matched",
                f"{pname}-matched-{idx}",
            )
            render_dataframe_section(
                report.get("df_not_matched", pd.DataFrame()),
                "Unmatched Partner Transactions",
                f"{pname}_unmatched",
                f"{pname}-unmatched-{idx}",
            )
            render_dataframe_section(
                report.get("df_partner_non_success_status", pd.DataFrame()),
                "Matched but Partner Status ≠ SUCCESS",
                f"{pname}_partner_non_success",
                f"{pname}-partner-non-success-{idx}",
            )
            render_dataframe_section(
                report.get("df_csgmap_non_success_status", pd.DataFrame()),
                "Matched but CSGMAP Status ≠ SUCCESS",
                f"{pname}_csgmap_non_success",
                f"{pname}-csgmap-non-success-{idx}",
            )
            render_dataframe_section(
                report.get("df_amount_mismatch", pd.DataFrame()),
                "Matched but Amounts Differ",
                f"{pname}_amount_mismatch",
                f"{pname}-amount-mismatch-{idx}",
            )

    csg_summary = results.get("csgmap_summary", {})
    if csg_summary:
        st.subheader("🗂 CSGMAP Summary")
        df_csg_all = csg_summary.get("df_csgmap_all", pd.DataFrame())
        df_csg_non_success = csg_summary.get("df_csgmap_non_success_status", pd.DataFrame())
        df_csg_unused = csg_summary.get("df_csgmap_unused_or_unmatched", pd.DataFrame())

        c1, c2, c3 = st.columns(3)
        c1.metric("Mapped Transactions", len(df_csg_all))
        c2.metric("Mapped with Status ≠ SUCCESS", len(df_csg_non_success))
        c3.metric("Unmatched / Unused", len(df_csg_unused))

        render_dataframe_section(
            df_csg_all,
            "CSGMAP Transactions Linked to Partners",
            "csgmap_mapped",
            "csgmap-mapped",
        )
        render_dataframe_section(
            df_csg_non_success,
            "CSGMAP Matched with Status ≠ SUCCESS",
            "csgmap_non_success",
            "csgmap-non-success",
        )
        render_dataframe_section(
            df_csg_unused,
            "CSGMAP Unmatched / Unused",
            "csgmap_unmatched",
            "csgmap-unmatched",
        )

        unused_by_status = csg_summary.get("df_csgmap_unused_by_status", {})
        for status_label, df_split in unused_by_status.items():
            render_dataframe_section(
                df_split,
                f"CSGMAP Unmatched — Status: {status_label}",
                f"csgmap_unmatched_status_{status_label}",
                f"csgmap-unmatched-status-{status_label}",
            )

    for key, df in results.items():
        if key in {"partner_reports", "csgmap_summary"}:
            continue
        with st.expander(f"📄 {key.upper()} — {len(df)} rows"):
            st.dataframe(df, width="stretch")
            buf = BytesIO()
            df.to_csv(buf, index=False)
            st.download_button(
                label=f"⬇ Download {key}.csv",
                data=buf.getvalue(),
                file_name=f"{key}.csv",
                mime="text/csv",
            )
