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
    detect_all_columns,
    REQUIRED_COLUMNS,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MojoPay Reconciliation Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME-AWARE STYLES
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        }
        h1, h2, h3 {
            color: var(--text-color);
        }
        .section-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-color);
            margin-bottom: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

AMOUNT_COLUMN = REQUIRED_COLUMNS["amount"]
STATUS_COLUMN = REQUIRED_COLUMNS["status"]
REFERENCE_COLUMN = REQUIRED_COLUMNS["reference"]
DATE_COLUMN = REQUIRED_COLUMNS["date"]


def normalize_dataframe(df):
    if isinstance(df, pd.DataFrame):
        return df
    return pd.DataFrame()


def download_buttons(df, filename_prefix, key_prefix):
    df = normalize_dataframe(df)
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"{filename_prefix}.csv",
        mime="text/csv",
        key=f"{key_prefix}-csv",
    )

    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    st.download_button(
        label="Download Excel",
        data=excel_buffer.getvalue(),
        file_name=f"{filename_prefix}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"{key_prefix}-xlsx",
    )


def render_dataframe_block(df, title, filename_prefix, key_prefix, show_title=True):
    df = normalize_dataframe(df)
    if show_title:
        st.markdown(f"**{title} ({len(df)} rows)**")
    if df.empty:
        st.caption("No records available.")
    st.dataframe(df, width="stretch")
    download_buttons(df, filename_prefix, key_prefix)


def render_dataframe_section(df, title, filename_prefix, key_prefix, expanded=False):
    df = normalize_dataframe(df)
    with st.expander(f"{title} ({len(df)} rows)", expanded=expanded):
        render_dataframe_block(
            df,
            title,
            filename_prefix,
            key_prefix,
            show_title=False,
        )


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "results" not in st.session_state:
    st.session_state["results"] = {}

if "step_status" not in st.session_state:
    st.session_state["step_status"] = "Not started"


def compute_step_status():
    has_csgmap = bool(st.session_state.get("csgmap_file"))
    has_partners = bool(st.session_state.get("partner_files"))
    has_cleaned = bool(st.session_state.get("cleaned_partners"))
    has_results = bool(st.session_state.get("results"))

    if has_results:
        return "Reconciliation complete"
    if has_cleaned:
        return "Cleaning complete"
    if has_csgmap and has_partners:
        return "Files uploaded"
    if has_csgmap or has_partners:
        return "Partial upload"
    return "Not started"

# ============================================================
# SIDEBAR - RUN STATUS
# ============================================================


def detect_csv_delimiter(sample_text):
    candidates = [("\t", "tab"), (",", "comma"), (";", "semicolon"), ("|", "pipe")]
    counts = {delim: sample_text.count(delim) for delim, _ in candidates}
    best_delim = max(counts, key=counts.get)
    return best_delim if counts[best_delim] > 0 else ","


def detect_header_row_from_lines(lines, delimiter, min_matches=2, min_columns=5):
    best_idx = None
    best_score = 0
    best_columns = 0
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        line_lower = line.lower()
        if "receipt no." in line_lower and "completion time" in line_lower:
            return idx
        cells = [cell.strip() for cell in line.split(delimiter)]
        nonempty_count = len([c for c in cells if c])
        if nonempty_count < min_matches:
            continue
        temp_df = pd.DataFrame(columns=cells)
        mapping, _ = detect_all_columns(temp_df, "preview")
        score = len(mapping)
        if score > best_score or (score == best_score and nonempty_count > best_columns):
            best_score = score
            best_idx = idx
            best_columns = nonempty_count
    if best_score >= min_matches:
        return int(best_idx)
    if best_columns >= min_columns and best_idx is not None:
        return int(best_idx)
    return 0


def read_csv_flexible(file, header_row, allow_skip, delimiter, skip_blank_lines):
    read_kwargs = {
        "sep": delimiter,
        "engine": "python",
        "header": header_row,
        "encoding": "utf-8-sig",
        "skip_blank_lines": skip_blank_lines,
    }
    if allow_skip:
        try:
            return pd.read_csv(file, on_bad_lines="skip", **read_kwargs)
        except TypeError:
            return pd.read_csv(file, error_bad_lines=False, **read_kwargs)
    return pd.read_csv(file, **read_kwargs)


def load_file(file):
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            file.seek(0)
            raw_bytes = file.read()
            raw_text = raw_bytes.decode("utf-8-sig", errors="ignore")
            delimiter = detect_csv_delimiter(raw_text[:2000])
            lines = raw_text.splitlines()[:50]
            header_row = detect_header_row_from_lines(lines, delimiter)
            st.session_state.setdefault("header_row_notes", {})[file.name] = {
                "header_row": header_row,
                "delimiter": delimiter,
            }
            file.seek(0)
            try:
                return read_csv_flexible(
                    file,
                    header_row=header_row,
                    allow_skip=False,
                    delimiter=delimiter,
                    skip_blank_lines=False,
                )
            except Exception:
                file.seek(0)
                st.warning(
                    f"{file.name} has inconsistent rows. Some lines were skipped during import."
                )
                return read_csv_flexible(
                    file,
                    header_row=header_row,
                    allow_skip=True,
                    delimiter=delimiter,
                    skip_blank_lines=False,
                )
        if name.endswith((".xlsx", ".xls")):
            file.seek(0)
            preview = pd.read_excel(file, header=None, nrows=25, dtype=str)
            header_row = detect_header_row(preview)
            st.session_state.setdefault("header_row_notes", {})[file.name] = {
                "header_row": header_row,
                "delimiter": None,
            }
            file.seek(0)
            return pd.read_excel(file, header=header_row)
        raise ValueError("Unsupported file format.")
    except Exception as e:
        st.error(f"Failed to load {file.name}: {e}")
        return None


def detect_header_row(df_preview, min_matches=2, min_columns=5):
    best_idx = None
    best_score = 0
    best_columns = 0
    for idx, row in df_preview.iterrows():
        values = [str(v) for v in row.tolist() if pd.notna(v)]
        if not values:
            continue
        nonempty_count = len([v for v in values if str(v).strip()])
        temp_df = pd.DataFrame(columns=values)
        mapping, _ = detect_all_columns(temp_df, "preview")
        score = len(mapping)
        if score > best_score or (score == best_score and nonempty_count > best_columns):
            best_score = score
            best_idx = idx
            best_columns = nonempty_count
    if best_score >= min_matches:
        return int(best_idx)
    if best_columns >= min_columns and best_idx is not None:
        return int(best_idx)
    return 0


def detect_partner_name(filename):
    name = filename.lower()
    for key in ["telecel", "ahan", "dd", "ep", "mtn", "ppt", "westom"]:
        if key in name:
            return key.upper()
    return "UNKNOWN"


def sidebar_status():
    with st.sidebar:
        st.image("https://mojo-pay.com/images/mojoLogo.svg", width=150)

        st.markdown("### Run Status")
        st.session_state["step_status"] = compute_step_status()
        st.write(f"Stage: {st.session_state.get('step_status', 'Not started')}")

        st.markdown("### Files")
        csgmap_file = st.session_state.get("csgmap_file")
        partner_files = st.session_state.get("partner_files", [])
        if csgmap_file:
            st.write(f"CSGMAP: {csgmap_file.name}")
        else:
            st.caption("CSGMAP: not uploaded")
        if partner_files:
            st.write(f"Partners: {len(partner_files)} file(s)")
            partner_names = sorted(
                {detect_partner_name(f.name) for f in partner_files}
            )
            st.caption(f"Detected: {', '.join(partner_names)}")
        else:
            st.caption("Partners: not uploaded")

        cleaned_partners = st.session_state.get("cleaned_partners", {})
        if cleaned_partners:
            st.markdown("### Cleaning")
            st.write(f"Partners cleaned: {len(cleaned_partners)}")

        results = st.session_state.get("results", {})
        if results:
            st.markdown("### Latest Run Summary")
            reconciled_count = len(results.get("reconciled", []))
            conflict_count = len(results.get("status_conflicts", []))
            col1, col2 = st.columns(2)
            col1.metric("Reconciled", reconciled_count)
            col2.metric("Conflicts", conflict_count)

        st.caption("MojoPay - Automated Financial Data Integrity Engine")


sidebar_status()

# ============================================================
# MAIN HEADER
# ============================================================

st.title("MojoPay Reconciliation Engine")
st.caption(
    "Upload files, clean data, and reconcile CSGMAP with partner statements. "
    "All outputs remain available for audit review."
)

# ============================================================
# UNIVERSAL FILE LOADER
# ============================================================


# ============================================================
# MAIN APP TABS
# ============================================================

tabs = st.tabs(["Upload Files", "Cleaning", "Reconciliation", "Results"])

# ============================================================
# TAB 1 - UPLOAD FILES
# ============================================================

with tabs[0]:
    st.header("Upload Files")
    st.write("Provide the CSGMAP file and one or more partner files.")

    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        st.subheader("CSGMAP File")
        csgmap_file = st.file_uploader(
            "Main CSGMAP file (CSV, XLSX, XLS)",
            type=["csv", "xlsx", "xls"],
        )
        if csgmap_file:
            st.write(f"Uploaded: {csgmap_file.name}")
            st.session_state["csgmap_file"] = csgmap_file

    with c2:
        st.subheader("Partner Files")
        partner_files = st.file_uploader(
            "Partner files (CSV, XLSX, XLS)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )
        if partner_files:
            st.write(f"Uploaded partner files: {len(partner_files)}")
            st.session_state["partner_files"] = partner_files

            file_summary = [
                {"File Name": f.name, "Detected Partner": detect_partner_name(f.name)}
                for f in partner_files
            ]
            st.markdown("#### File Summary")
            st.dataframe(pd.DataFrame(file_summary), width="stretch")

    if st.session_state.get("csgmap_file") and st.session_state.get("partner_files"):
        st.success("Files uploaded. Continue to Clean and Validate.")

# ============================================================
# TAB 2 - CLEANING
# ============================================================

with tabs[1]:
    st.header("Clean and Validate")
    st.write("Review file quality and duplicates before reconciliation.")

    if not st.session_state.get("csgmap_file") or not st.session_state.get("partner_files"):
        st.warning("Upload both the CSGMAP file and partner files in the Upload tab.")
        st.stop()

    st.subheader("CSGMAP Validation")
    csgmap_df = load_file(st.session_state["csgmap_file"])
    if csgmap_df is None:
        st.stop()
    csg_note = st.session_state.get("header_row_notes", {}).get(
        st.session_state["csgmap_file"].name
    )
    if csg_note:
        row_num = csg_note["header_row"] + 1
        delim = csg_note.get("delimiter")
        if delim == "\t":
            delim_label = "tab"
        elif delim == ",":
            delim_label = "comma"
        elif delim == ";":
            delim_label = "semicolon"
        elif delim == "|":
            delim_label = "pipe"
        else:
            delim_label = "unknown"
        note_parts = [f"Detected header row: {row_num}"]
        if delim is not None:
            note_parts.append(f"Delimiter: {delim_label}")
        st.caption(" | ".join(note_parts))

    csgmap_clean, csgmap_dupes, csgmap_summary, csgmap_diag = clean_csgmap(csgmap_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total rows", len(csgmap_df))
    col2.metric("Cleaned rows", csgmap_summary.get("total_rows_after_cleaning", 0))
    col3.metric(
        "Duplicate references",
        csgmap_summary.get("total_duplicates_detected", 0),
    )
    col4.metric("Duplicate rows", len(csgmap_dupes))

    render_dataframe_section(
        csgmap_dupes,
        "CSGMAP duplicate records",
        "csgmap_duplicates",
        "csgmap-duplicates",
    )

    st.session_state["csgmap_clean"] = csgmap_clean

    st.divider()
    st.subheader("Partner Validation")

    cleaned_partners = {}
    partner_summaries = []
    partner_dupes_map = {}

    for f in st.session_state["partner_files"]:
        pname = detect_partner_name(f.name)
        df = load_file(f)
        if df is None:
            continue
        partner_note = st.session_state.get("header_row_notes", {}).get(f.name)
        if partner_note:
            row_num = partner_note["header_row"] + 1
            delim = partner_note.get("delimiter")
            if delim == "\t":
                delim_label = "tab"
            elif delim == ",":
                delim_label = "comma"
            elif delim == ";":
                delim_label = "semicolon"
            elif delim == "|":
                delim_label = "pipe"
            else:
                delim_label = "unknown"
            note_parts = [f"Detected header row: {row_num}"]
            if delim is not None:
                note_parts.append(f"Delimiter: {delim_label}")
            st.caption(f"{pname}: " + " | ".join(note_parts))
        try:
            cleaned, dupes, summary, diag = clean_partner_sheet(
                df,
                pname,
                st.session_state.get("csgmap_clean"),
            )
        except TypeError:
            cleaned, dupes, summary, diag = clean_partner_sheet(df, pname)
        cleaned_partners[pname] = cleaned
        partner_dupes_map[pname] = dupes
        partner_summaries.append(
            {
                "Partner": pname,
                "Total rows": len(df),
                "Cleaned rows": summary.get("total_rows_after_cleaning", 0),
                "Duplicate references": summary.get("total_duplicates_detected", 0),
                "Duplicate rows": len(dupes),
            }
        )

    st.session_state["cleaned_partners"] = cleaned_partners

    if partner_summaries:
        st.markdown("#### Partner Validation Summary")
        st.dataframe(pd.DataFrame(partner_summaries), width="stretch")

        st.markdown("#### Duplicate Records by Partner")
        for row in partner_summaries:
            pname = row["Partner"]
            render_dataframe_section(
                partner_dupes_map.get(pname, pd.DataFrame()),
                f"Duplicate records - {pname}",
                f"{pname}_duplicates",
                f"{pname}-duplicates",
            )

    st.success("Cleaning complete. Continue to Reconciliation.")

# ============================================================
# TAB 3 - RECONCILIATION
# ============================================================

with tabs[2]:
    st.header("Reconciliation Process")
    st.write("Run the matching logic across partner files and CSGMAP.")

    if "csgmap_clean" not in st.session_state or "cleaned_partners" not in st.session_state:
        st.warning("Complete the Cleaning tab before running reconciliation.")
        st.stop()

    rec_list = []
    conflict_list = []
    partner_unmatch = []
    partner_reports = []
    csgmap_matched_views = []
    csgmap_non_success_views = []

    csgmap_remaining = st.session_state["csgmap_clean"].copy()

    for pname, df in st.session_state["cleaned_partners"].items():
        with st.spinner(f"Reconciling {pname}"):
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
        st.write(f"Completed reconciliation for {pname}.")

    # AHAN matching temporarily disabled in favor of standard reconciliation.
    ahan_match = ahan_amb = ahan_unmatch = pd.DataFrame()

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
    st.success("Reconciliation complete. Proceed to the Results tab.")

# ============================================================
# TAB 4 - RESULTS
# ============================================================

with tabs[3]:
    st.header("Reconciliation Results")

    if not st.session_state["results"]:
        st.write("Run reconciliation to view results.")
        st.stop()

    results = st.session_state["results"]
    results_tabs = st.tabs(["Summary", "Partners", "CSGMAP", "All Outputs"])

    with results_tabs[0]:
        st.subheader("Run Summary")

        overview_metrics = []
        if "reconciled" in results:
            overview_metrics.append(("Reconciled transactions", len(results.get("reconciled", []))))
        if "status_conflicts" in results:
            overview_metrics.append(("Status conflicts", len(results.get("status_conflicts", []))))

        if overview_metrics:
            cols = st.columns(len(overview_metrics))
            for col, (label, value) in zip(cols, overview_metrics):
                col.metric(label, value)

        st.markdown("#### Category Totals")
        category_rows = []
        for key, df in results.items():
            if key in {"partner_reports", "csgmap_summary"} or key.startswith("ahan_"):
                continue
            df = normalize_dataframe(df)
            amount_total = df[AMOUNT_COLUMN].sum() if AMOUNT_COLUMN in df.columns else None
            category_rows.append(
                {
                    "Category": key.replace("_", " ").title(),
                    "Rows": len(df),
                    "Total Amount": amount_total,
                }
            )
        if category_rows:
            st.dataframe(pd.DataFrame(category_rows), width="stretch")
        else:
            st.caption("No result categories available.")

    with results_tabs[1]:
        partner_reports = results.get("partner_reports", [])
        if not partner_reports:
            st.caption("No partner summaries available.")
        else:
            partner_entries = []
            summary_rows = []
            for idx, report in enumerate(partner_reports, start=1):
                pname = report.get("partner", "UNKNOWN")
                label = f"{pname} (report {idx})"
                partner_entries.append(
                    {"label": label, "report": report, "partner": pname, "index": idx}
                )
                summary_rows.append(
                    {
                        "Partner": pname,
                        "Matched": report.get("matched_total", 0),
                        "Unmatched": report.get("not_matched_total", 0),
                        "Partner status not success": report.get(
                            "matched_partner_non_success_status", 0
                        ),
                        "CSGMAP status not success": report.get(
                            "matched_csgmap_non_success_status", 0
                        ),
                        "Amount mismatches": report.get("amount_mismatch_total", 0),
                    }
                )

            st.markdown("#### Partner Summary")
            st.dataframe(pd.DataFrame(summary_rows), width="stretch")

            partner_tabs = st.tabs([entry["label"] for entry in partner_entries])
            for tab, entry in zip(partner_tabs, partner_entries):
                with tab:
                    report = entry["report"]
                    pname = entry["partner"]
                    idx = entry["index"]

                    st.subheader(f"Partner: {pname}")

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Matched transactions", report.get("matched_total", 0))
                    m2.metric("Unmatched transactions", report.get("not_matched_total", 0))
                    m3.metric(
                        "Partner status not success",
                        report.get("matched_partner_non_success_status", 0),
                    )
                    m4.metric(
                        "CSGMAP status not success",
                        report.get("matched_csgmap_non_success_status", 0),
                    )
                    m5.metric("Amount mismatches", report.get("amount_mismatch_total", 0))

                    detail_tabs = st.tabs(
                        [
                            "Matched",
                            "Unmatched",
                            "Partner status not success",
                            "CSGMAP status not success",
                            "Amount mismatches",
                        ]
                    )

                    with detail_tabs[0]:
                        render_dataframe_block(
                            report.get("df_matched", pd.DataFrame()),
                            "Matched transactions",
                            f"{pname}_matched",
                            f"{pname}-matched-{idx}",
                        )
                    with detail_tabs[1]:
                        render_dataframe_block(
                            report.get("df_not_matched", pd.DataFrame()),
                            "Unmatched partner transactions",
                            f"{pname}_unmatched",
                            f"{pname}-unmatched-{idx}",
                        )
                    with detail_tabs[2]:
                        render_dataframe_block(
                            report.get("df_partner_non_success_status", pd.DataFrame()),
                            "Matched with partner status not success",
                            f"{pname}_partner_non_success",
                            f"{pname}-partner-non-success-{idx}",
                        )
                    with detail_tabs[3]:
                        render_dataframe_block(
                            report.get("df_csgmap_non_success_status", pd.DataFrame()),
                            "Matched with CSGMAP status not success",
                            f"{pname}_csgmap_non_success",
                            f"{pname}-csgmap-non-success-{idx}",
                        )
                    with detail_tabs[4]:
                        render_dataframe_block(
                            report.get("df_amount_mismatch", pd.DataFrame()),
                            "Matched with amount mismatch",
                            f"{pname}_amount_mismatch",
                            f"{pname}-amount-mismatch-{idx}",
                        )

    with results_tabs[2]:
        csg_summary = results.get("csgmap_summary", {})
        if not csg_summary:
            st.caption("No CSGMAP summary available.")
        else:
            st.subheader("CSGMAP Summary")
            df_csg_all = csg_summary.get("df_csgmap_all", pd.DataFrame())
            df_csg_non_success = csg_summary.get(
                "df_csgmap_non_success_status", pd.DataFrame()
            )
            df_csg_unused = csg_summary.get("df_csgmap_unused_or_unmatched", pd.DataFrame())

            c1, c2, c3 = st.columns(3)
            c1.metric("Mapped transactions", len(df_csg_all))
            c2.metric("Mapped with status not success", len(df_csg_non_success))
            c3.metric("Unmatched or unused", len(df_csg_unused))

            csg_tabs = st.tabs(
                [
                    "Mapped",
                    "Mapped status not success",
                    "Unmatched or unused",
                    "Unmatched by status",
                ]
            )

            with csg_tabs[0]:
                render_dataframe_block(
                    df_csg_all,
                    "CSGMAP transactions linked to partners",
                    "csgmap_mapped",
                    "csgmap-mapped",
                )
            with csg_tabs[1]:
                render_dataframe_block(
                    df_csg_non_success,
                    "CSGMAP matched with status not success",
                    "csgmap_non_success",
                    "csgmap-non-success",
                )
            with csg_tabs[2]:
                render_dataframe_block(
                    df_csg_unused,
                    "CSGMAP unmatched or unused",
                    "csgmap_unmatched",
                    "csgmap-unmatched",
                )
            with csg_tabs[3]:
                unused_by_status = csg_summary.get("df_csgmap_unused_by_status", {})
                if not unused_by_status:
                    st.caption("No unmatched records grouped by status.")
                else:
                    status_options = list(unused_by_status.keys())
                    selected_status = st.selectbox(
                        "Select status",
                        options=status_options,
                        key="csgmap-unmatched-status-select",
                    )
                    render_dataframe_block(
                        unused_by_status.get(selected_status, pd.DataFrame()),
                        f"CSGMAP unmatched status: {selected_status}",
                        f"csgmap_unmatched_status_{selected_status}",
                        f"csgmap-unmatched-status-{selected_status}",
                    )

    with results_tabs[3]:
        st.subheader("All Outputs")
        other_keys = [
            key
            for key in results.keys()
            if key not in {"partner_reports", "csgmap_summary"}
            and not key.startswith("ahan_")
        ]
        if not other_keys:
            st.caption("No datasets available.")
        for key in other_keys:
            df = results.get(key, pd.DataFrame())
            render_dataframe_section(
                df,
                key.replace("_", " ").title(),
                key,
                f"{key}-dataset",
            )
