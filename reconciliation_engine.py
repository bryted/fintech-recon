# ======================================================================
# MojoPay Reconciliation Engine (Final Production Version)
# ======================================================================

import pandas as pd
import numpy as np
import re
import warnings
from rapidfuzz import fuzz

# ======================================================================
# CANONICAL FIELDS
# ======================================================================

REQUIRED_COLUMNS = {
    "reference": "MojoPay Reference No",
    "date": "Transaction Date",
    "amount": "Transacted Amount",
    "status": "Status",
    "merchant": "Merchant Ref Code",
    "mobile": "Payer Mobile",
}

FUZZY_THRESHOLD = 80   # strict fuzzy tolerance
SUCCESS_STATUS_VALUES = {"SUCCESS", "COMPLETED", "SUCCESSFUL"}


# ======================================================================
# UTILITIES — NORMALIZATION
# ======================================================================

def normalize_string(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    return re.sub(r"[^a-z0-9]+", "", s).strip()


def find_column_case_insensitive(df: pd.DataFrame, target: str):
    target_norm = target.strip().lower()
    for col in df.columns:
        if col.strip().lower() == target_norm:
            return col
    return None


def standardize_reference_series(series: pd.Series) -> pd.Series:
    def normalize(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, np.integer)):
            return str(val)
        if isinstance(val, (float, np.floating)):
            if float(val).is_integer():
                return str(int(val))
            return str(val).strip()
        text = str(val).strip()
        if not text or text.lower() in {"nan", "none"}:
            return np.nan
        return text

    normalized = series.apply(normalize)
    return normalized


def concat_or_empty(frames):
    valid = [
        df
        for df in frames
        if df is not None and not df.empty and not df.isna().all().all()
    ]
    if valid:
        return pd.concat(valid, ignore_index=True).drop_duplicates().reset_index(drop=True)
    return pd.DataFrame()


def filter_valid_frames(frames):
    return [
        df
        for df in frames
        if isinstance(df, pd.DataFrame)
        and not df.empty
        and not df.isna().all().all()
    ]


# ======================================================================
# 1. DETERMINISTIC COLUMN DETECTION
# ======================================================================

COLUMN_RULES = {
    "reference": ["mojopay", "reference", "ref_no", "rrn"],
    "date": ["date", "transdate", "transactiondate", "datetime"],
    "amount": ["amount", "amt", "transactedamount", "transactionamount"],
    "status": ["status", "state", "transstatus", "responsecode"],
    "merchant": ["merchant", "merchantref", "merchantreference", "refcode", "trans.id"],
    "mobile": ["mobile", "msisdn", "payer", "payermobile", "phonenumber"],
}

def detect_column_by_rules(col_name: str):
    col_norm = normalize_string(col_name)
    for key, patterns in COLUMN_RULES.items():
        for pat in patterns:
            if pat in col_norm:
                return key
    return None


# ======================================================================
# 2. SECONDARY FUZZY MATCHING
# ======================================================================

def best_fuzzy_match(col_name: str, candidates: dict):
    norm_col = normalize_string(col_name)
    best_score = 0
    best_key = None

    for key, canonical in candidates.items():
        score = fuzz.ratio(norm_col, normalize_string(canonical))
        if score > best_score:
            best_score = score
            best_key = key

    if best_score >= FUZZY_THRESHOLD:
        return best_key
    return None


# ======================================================================
# 3. MASTER COLUMN DETECTOR
# ======================================================================

def detect_all_columns(df: pd.DataFrame, sheet_name: str):
    diagnostics = {
        "sheet": sheet_name,
        "columns_detected": df.columns.tolist(),
        "mapping": {},
        "missing_after_detection": [],
    }

    mapping = {}

    # Pass 1: deterministic
    for col in df.columns:
        rule_match = detect_column_by_rules(col)
        if rule_match:
            mapping.setdefault(rule_match, []).append(col)

    # Pass 2: fuzzy fallback for unmatched canonical fields
    for key in REQUIRED_COLUMNS.keys():
        if key not in mapping:
            for col in df.columns:
                if detect_column_by_rules(col):
                    continue
                fuzzy_key = best_fuzzy_match(col, REQUIRED_COLUMNS)
                if fuzzy_key == key:
                    mapping.setdefault(key, []).append(col)

    def pick_best_column(key, cols):
        canonical = normalize_string(REQUIRED_COLUMNS[key])
        best_col = cols[0]
        best_score = -1
        for col in cols:
            score = fuzz.ratio(normalize_string(col), canonical)
            if score > best_score or (score == best_score and normalize_string(col) == canonical):
                best_score = score
                best_col = col
        return best_col

    # Resolve collisions — pick best column in each group
    resolved = {}
    for key, cols in mapping.items():
        if len(cols) == 1:
            resolved[key] = cols[0]
        else:
            resolved[key] = pick_best_column(key, cols)

    # Track missing
    for key in REQUIRED_COLUMNS.keys():
        if key not in resolved:
            diagnostics["missing_after_detection"].append(REQUIRED_COLUMNS[key])

    diagnostics["mapping"] = resolved
    return resolved, diagnostics


# ======================================================================
# 4. CANONICAL FIELD INJECTION (Option A)
# ======================================================================

def inject_canonical_columns(df: pd.DataFrame, mapping: dict):
    """
    Adds canonical columns (REQUIRED_COLUMNS values) into df.
    Does NOT drop original columns.
    """

    for key, canon in REQUIRED_COLUMNS.items():
        if key in mapping:
            df[canon] = df[mapping[key]]
        else:
            df[canon] = np.nan

    return df


# ======================================================================
# 5. DATA TYPE NORMALIZATION
# ======================================================================

def normalize_types(df: pd.DataFrame):
    status_col = REQUIRED_COLUMNS["status"]
    df[REQUIRED_COLUMNS["date"]] = pd.to_datetime(
        df[REQUIRED_COLUMNS["date"]], dayfirst=True, errors="coerce"
    )
    df[REQUIRED_COLUMNS["amount"]] = pd.to_numeric(
        df[REQUIRED_COLUMNS["amount"]], errors="coerce"
    )
    df[status_col] = (
        df[status_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    df.loc[df[status_col].isin({"", "NAN", "NONE"}), status_col] = np.nan
    df.loc[df[status_col].isin(SUCCESS_STATUS_VALUES), status_col] = "SUCCESS"
    return df


# ======================================================================
# 6. DUPLICATE DETECTION + RESOLUTION
# ======================================================================

def detect_duplicates(df, subset_col=None):
    if subset_col is None:
        subset_col = REQUIRED_COLUMNS["reference"]
    subset = [subset_col] if isinstance(subset_col, str) else list(subset_col)
    return df[df.duplicated(subset=subset, keep=False)].copy()


def resolve_duplicate_group(group):
    status_col = REQUIRED_COLUMNS["status"]
    date_col = REQUIRED_COLUMNS["date"]

    success_rows = group[group[status_col] == "SUCCESS"]
    if len(success_rows) == 1:
        return success_rows.iloc[0]
    if len(success_rows) > 1:
        success_rows[date_col] = pd.to_datetime(success_rows[date_col], dayfirst=True, errors="coerce")
        return success_rows.sort_values(date_col).iloc[-1]

    group[date_col] = pd.to_datetime(group[date_col], dayfirst=True, errors="coerce")
    return group.sort_values(date_col).iloc[-1]


def resolve_duplicates(df, subset_col=None):
    key = subset_col if subset_col is not None else REQUIRED_COLUMNS["reference"]

    dupes = detect_duplicates(df, key)
    if dupes.empty:
        return df.copy(), dupes

    cleaned = df.groupby(key, group_keys=False).apply(resolve_duplicate_group)
    return cleaned.reset_index(drop=True), dupes


# ======================================================================
# 7. CLEANING PIPELINES
# ======================================================================

def clean_csgmap(df_raw: pd.DataFrame):
    mapping, diagnostics = detect_all_columns(df_raw, "CSGMAP")
    df = inject_canonical_columns(df_raw.copy(), mapping)

    # Now verify required canonical fields exist
    missing = [col for col in REQUIRED_COLUMNS.values() if col not in df.columns]
    if missing:
        raise ValueError(f"CSGMAP missing required columns: {missing}")

    df = normalize_types(df)
    cleaned, dupes = resolve_duplicates(df)
    ref_col = REQUIRED_COLUMNS["reference"]
    dupe_count = dupes[ref_col].nunique() if not dupes.empty else 0
    summary = {
        "sheet": "CSGMAP",
        "total_rows_after_cleaning": len(cleaned),
        "total_duplicates_detected": dupe_count,
        "rows_by_status": cleaned[REQUIRED_COLUMNS["status"]]
        .value_counts(dropna=False)
        .to_dict(),
        "amount_by_status": cleaned.groupby(
            REQUIRED_COLUMNS["status"], dropna=False
        )[REQUIRED_COLUMNS["amount"]].sum(min_count=1).to_dict(),
    }

    return cleaned, dupes, summary, diagnostics


def clean_partner_sheet(df_raw: pd.DataFrame, sheet_name: str):
    mapping, diagnostics = detect_all_columns(df_raw, sheet_name)
    df = inject_canonical_columns(df_raw.copy(), mapping)

    if df[REQUIRED_COLUMNS["status"]].isna().all():
        df[REQUIRED_COLUMNS["status"]] = "SUCCESS"

    # Special rule for PPT — assume STATUS = SUCCESS
    if sheet_name == "PPT":
        df[REQUIRED_COLUMNS["status"]] = "SUCCESS"
        invoice_col = find_column_case_insensitive(df_raw, "Invoice Amount")
        if invoice_col:
            df[REQUIRED_COLUMNS["amount"]] = df_raw[invoice_col]

    # Fallback Merchant Ref logic
    if df[REQUIRED_COLUMNS["merchant"]].isna().all():
        try:
            df[REQUIRED_COLUMNS["merchant"]] = (
                df_raw["Trans.ID"]
                if "Trans.ID" in df_raw.columns
                else df[REQUIRED_COLUMNS["reference"]]
            )
        except:
            df[REQUIRED_COLUMNS["merchant"]] = df[REQUIRED_COLUMNS["reference"]]

    df = normalize_types(df)

    dedup_col = REQUIRED_COLUMNS["reference"]
    if sheet_name == "TELECEL":
        receipt_col = find_column_case_insensitive(df, "Receipt No.")
        if receipt_col:
            dedup_col = receipt_col

    cleaned, dupes = resolve_duplicates(df, dedup_col)
    ref_col = dedup_col if isinstance(dedup_col, str) else REQUIRED_COLUMNS["reference"]
    dupe_count = dupes[ref_col].nunique() if not dupes.empty and ref_col in dupes.columns else len(dupes)
    summary = {
        "sheet": sheet_name,
        "total_rows_after_cleaning": len(cleaned),
        "total_duplicates_detected": dupe_count,
        "rows_by_status": cleaned[REQUIRED_COLUMNS["status"]]
        .value_counts(dropna=False)
        .to_dict(),
        "amount_by_status": cleaned.groupby(
            REQUIRED_COLUMNS["status"], dropna=False
        )[REQUIRED_COLUMNS["amount"]].sum(min_count=1).to_dict(),
    }

    return cleaned, dupes, summary, diagnostics


# ======================================================================
# 8. MERGING — NEW SAFE PRE-SUFFIX METHOD
# ======================================================================

def reconcile_sheet(csgmap_df, partner_df, sheet_name):
    ref = REQUIRED_COLUMNS["reference"]
    amt = REQUIRED_COLUMNS["amount"]
    date_col = REQUIRED_COLUMNS["date"]

    # Pre-suffix (prevents pandas MergeError)
    csg = csgmap_df.add_suffix("_csgmap")
    prt = partner_df.add_suffix("_partner")

    # Rename reference columns back so they match
    csg = csg.rename(columns={f"{ref}_csgmap": ref})
    prt = prt.rename(columns={f"{ref}_partner": ref})

    # enforce consistent reference key formatting to avoid merge errors/mismatches
    for frame in (csg, prt):
        frame[ref] = standardize_reference_series(frame[ref])

    merged = prt.merge(csg, on=ref, how="outer", indicator=True)

    status_partner = REQUIRED_COLUMNS["status"] + "_partner"
    status_csgmap = REQUIRED_COLUMNS["status"] + "_csgmap"
    amt_partner = f"{amt}_partner"
    amt_csgmap = f"{amt}_csgmap"

    matched = merged[merged["_merge"] == "both"].copy()
    matched["source"] = sheet_name

    # Only consider matches as valid when transaction amounts align
    partner_amount = pd.to_numeric(matched[amt_partner], errors="coerce")
    csgmap_amount = pd.to_numeric(matched[amt_csgmap], errors="coerce")
    amount_equal = (
        partner_amount.notna()
        & csgmap_amount.notna()
        & np.isclose(partner_amount, csgmap_amount, rtol=0.0, atol=0.01)
    )
    matched_amount = matched[amount_equal].copy()
    mismatched_amount = matched[~amount_equal].copy()

    partner_unmatched = merged[merged["_merge"] == "left_only"].copy()
    csgmap_unmatched = merged[merged["_merge"] == "right_only"].copy()

    if not mismatched_amount.empty:
        concat_candidates = filter_valid_frames(
            [csgmap_unmatched, mismatched_amount]
        )
        if concat_candidates:
            csgmap_unmatched = pd.concat(concat_candidates, ignore_index=True)
        else:
            csgmap_unmatched = pd.DataFrame(columns=csgmap_unmatched.columns)

    partner_unmatched["source"] = sheet_name
    csgmap_unmatched["source"] = "CSGMAP"

    # Determine conflicts vs reconciled based on status availability
    partner_status_available = matched_amount[status_partner].notna()
    csg_status_available = matched_amount[status_csgmap].notna()
    both_status_available = partner_status_available & csg_status_available
    both_success = (
        matched_amount[status_partner] == "SUCCESS"
    ) & (matched_amount[status_csgmap] == "SUCCESS")
    status_conflict_mask = both_status_available & ~both_success

    conflicts = matched_amount[status_conflict_mask].copy()
    reconciled = matched_amount[~status_conflict_mask].copy()

    for df in [conflicts, reconciled, partner_unmatched, csgmap_unmatched]:
        if "_merge" in df.columns:
            df.drop(columns=["_merge"], inplace=True)

    def format_csg_rows(df, include_partner_status=True):
        desired_cols = [
            ref,
            date_col,
            amt,
            status_csgmap,
        ]
        if include_partner_status:
            desired_cols.append(status_partner)
        desired_cols.append("source")

        if df.empty:
            return pd.DataFrame(columns=desired_cols)

        rename_map = {}
        subset_cols = []
        if ref in df.columns:
            subset_cols.append(ref)
        for field in [date_col, amt]:
            col_name = f"{field}_csgmap"
            if col_name in df.columns:
                subset_cols.append(col_name)
                rename_map[col_name] = field
        if status_csgmap in df.columns:
            subset_cols.append(status_csgmap)
        if include_partner_status and status_partner in df.columns:
            subset_cols.append(status_partner)
        if "source" in df.columns:
            subset_cols.append("source")

        formatted = df[subset_cols].copy()
        formatted.rename(columns=rename_map, inplace=True)
        for col in desired_cols:
            if col not in formatted.columns:
                formatted[col] = np.nan
        return formatted[desired_cols]

    def format_partner_rows(df, include_status=False):
        desired_cols = [ref, date_col, amt]
        if include_status:
            desired_cols.extend([status_partner, status_csgmap])
        desired_cols.append("source")
        if df.empty:
            return pd.DataFrame(columns=desired_cols)

        rename_map = {}
        subset_cols = []
        if ref in df.columns:
            subset_cols.append(ref)
        for field in [date_col, amt]:
            col_name = f"{field}_partner"
            if col_name in df.columns:
                subset_cols.append(col_name)
                rename_map[col_name] = field
        if include_status and status_partner in df.columns:
            subset_cols.append(status_partner)
        if include_status and status_csgmap in df.columns:
            subset_cols.append(status_csgmap)
        if "source" in df.columns:
            subset_cols.append("source")

        formatted = df[subset_cols].copy()
        formatted.rename(columns=rename_map, inplace=True)
        for col in desired_cols:
            if col not in formatted.columns:
                formatted[col] = np.nan
        return formatted[desired_cols]

    # Build a CSGMAP-focused view for reconciled/conflict matches and tidy unmatched sets
    reconciled_view = format_csg_rows(reconciled, include_partner_status=True)
    conflicts_view = format_csg_rows(conflicts, include_partner_status=True)

    partner_unmatched_view = format_partner_rows(partner_unmatched)

    # strip the temporary _csgmap suffixes so the next reconciliation pass
    # sees canonical column names again (retain all canonical fields for AHAN logic)
    canonical_csg_cols = [
        f"{col}_csgmap" for col in REQUIRED_COLUMNS.values()
        if f"{col}_csgmap" in csgmap_unmatched.columns
    ]
    keep_cols = [col for col in ["source", ref] if col in csgmap_unmatched.columns]
    trimmed_cols = keep_cols + canonical_csg_cols
    csgmap_unmatched_clean = csgmap_unmatched[trimmed_cols].copy()
    rename_map = {
        f"{col}_csgmap": col for col in REQUIRED_COLUMNS.values()
        if f"{col}_csgmap" in csgmap_unmatched_clean.columns
    }
    csgmap_unmatched_clean.rename(columns=rename_map, inplace=True)

    # Partner-level metrics & dataframes
    matched_partner_view = format_partner_rows(matched_amount, include_status=True)
    not_matched_partner_view = format_partner_rows(partner_unmatched, include_status=True)
    amount_mismatch_partner_view = format_partner_rows(mismatched_amount, include_status=True)

    partner_non_success_mask = matched_amount[status_partner].notna() & (
        matched_amount[status_partner] != "SUCCESS"
    )
    partner_non_success_df = format_partner_rows(
        matched_amount[partner_non_success_mask],
        include_status=True,
    )

    csgmap_non_success_mask = matched_amount[status_csgmap].notna() & (
        matched_amount[status_csgmap] != "SUCCESS"
    )
    csgmap_non_success_df = format_csg_rows(
        matched_amount[csgmap_non_success_mask],
        include_partner_status=True,
    )

    partner_report = {
        "partner": sheet_name,
        "matched_total": len(matched_partner_view),
        "not_matched_total": len(not_matched_partner_view),
        "matched_partner_non_success_status": len(partner_non_success_df),
        "matched_csgmap_non_success_status": len(csgmap_non_success_df),
        "amount_mismatch_total": len(amount_mismatch_partner_view),
        "df_matched": matched_partner_view,
        "df_not_matched": not_matched_partner_view,
        "df_partner_non_success_status": partner_non_success_df,
        "df_csgmap_non_success_status": csgmap_non_success_df,
        "df_amount_mismatch": amount_mismatch_partner_view,
    }

    csgmap_matched_view = format_csg_rows(
        matched_amount,
        include_partner_status=True,
    )

    return (
        reconciled_view,
        conflicts_view,
        partner_unmatched_view,
        csgmap_unmatched_clean,
        partner_report,
        csgmap_matched_view,
        csgmap_non_success_df,
    )


# ======================================================================
# 9. AHAN MATCHING
# ======================================================================

def match_ahan_transactions(csgmap_unmatched_df, ahan_df):
    amt = REQUIRED_COLUMNS["amount"]
    mob = REQUIRED_COLUMNS["mobile"]
    ref = REQUIRED_COLUMNS["reference"]

    ahan_df["ahan_key"] = ahan_df[amt].astype(str) + "|" + ahan_df[mob].astype(str)
    csgmap_unmatched_df["csg_key"] = csgmap_unmatched_df[amt].astype(str) + "|" + csgmap_unmatched_df[mob].astype(str)

    merged = ahan_df.merge(
        csgmap_unmatched_df,
        left_on="ahan_key",
        right_on="csg_key",
        how="left",
        suffixes=("_ahan", "_csg"),
    )

    matched_rows = []
    ambiguous = []
    unmatched = []

    for key, group in merged.groupby("ahan_key"):
        matches = group[group["csg_key"].notna()]

        if len(matches) == 1:
            row = matches.iloc[0].copy()
            row["source"] = "AHAN"
            matched_rows.append(row)

        elif len(matches) > 1:
            ambiguous.append(group.iloc[0])

        else:
            unmatched.append(group.iloc[0])

    matched_df = pd.DataFrame(matched_rows)
    amb_df = pd.DataFrame(ambiguous)
    unmatch_df = pd.DataFrame(unmatched)

    if not matched_df.empty:
        matched_refs = matched_df[ref].unique()
        new_unmatched = csgmap_unmatched_df[
            ~csgmap_unmatched_df[ref].isin(matched_refs)
        ]
    else:
        new_unmatched = csgmap_unmatched_df.copy()

    return matched_df, amb_df, unmatch_df, new_unmatched


# ======================================================================
# 10. FINAL CONSOLIDATION
# ======================================================================

def consolidate_reconciliation_results(
    reconciled_list,
    conflicts_list,
    partner_unmatched_list,
    final_csgmap_unmatched,
    ahan_matched,
    ahan_amb,
    ahan_unmatched,
    partner_reports=None,
    csgmap_matched_views=None,
    csgmap_non_success_views=None,
):
    partner_reports = partner_reports or []
    csgmap_matched_views = csgmap_matched_views or []
    csgmap_non_success_views = csgmap_non_success_views or []

    reconciled_frames = filter_valid_frames(reconciled_list + [ahan_matched])
    if reconciled_frames:
        reconciled = (
            pd.concat(reconciled_frames, ignore_index=True)
            .drop_duplicates()
        )
    else:
        reconciled = pd.DataFrame()

    conflict_frames = filter_valid_frames(conflicts_list)
    conflicts = (
        pd.concat(conflict_frames, ignore_index=True)
        if conflict_frames
        else pd.DataFrame()
    )

    partner_unmatched_frames = filter_valid_frames(partner_unmatched_list)
    partner_unmatched = (
        pd.concat(partner_unmatched_frames, ignore_index=True)
        if partner_unmatched_frames
        else pd.DataFrame()
    )

    csgmap_all = concat_or_empty(csgmap_matched_views)
    csgmap_non_success = concat_or_empty(csgmap_non_success_views)
    csgmap_unused = final_csgmap_unmatched.copy()

    status_col = REQUIRED_COLUMNS["status"]
    csgmap_unused_by_status = {}
    if status_col in csgmap_unused.columns:
        for status_value, group in csgmap_unused.groupby(status_col, dropna=False):
            key = "NaN" if pd.isna(status_value) else str(status_value)
            csgmap_unused_by_status[key] = group.copy()
    else:
        csgmap_unused_by_status["ALL"] = csgmap_unused.copy()

    csgmap_summary = {
        "df_csgmap_all": csgmap_all,
        "df_csgmap_non_success_status": csgmap_non_success,
        "df_csgmap_unused_or_unmatched": csgmap_unused,
        "df_csgmap_unused_by_status": csgmap_unused_by_status,
    }

    return {
        "reconciled": reconciled,
        "status_conflicts": conflicts,
        "partner_unmatched": partner_unmatched,
        "ahan_ambiguous": ahan_amb,
        "ahan_unmatched": ahan_unmatched,
        "csgmap_unmatched": final_csgmap_unmatched,
        "partner_reports": partner_reports,
        "csgmap_summary": csgmap_summary,
    }
