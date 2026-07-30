# =============================================================================
# profiling.py — DataGuard Agent: Data Profiling & Statistical Anomaly Engine
# =============================================================================
# Provides:
#   • Column-level statistical profiling (mean, std, min, max, nulls, uniques)
#   • Null-rate anomaly detection (configurable threshold)
#   • Duplicate / primary-key violation detection
#   • Outlier detection via Z-Score and Tukey IQR fence
#   • Categorical label consistency checks
#   • Aggregated AnomalyReport dataclass
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

import config


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    """Statistical snapshot of a single column."""
    column_name: str
    dtype: str
    total_rows: int
    null_count: int
    null_rate: float
    unique_count: int
    # Numeric only
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    # Categorical only
    top_values: Optional[dict] = None


@dataclass
class AnomalyRecord:
    """A single detected anomaly."""
    anomaly_type: str          # "NULL_RATE" | "DUPLICATE" | "OUTLIER_ZSCORE" | "OUTLIER_IQR" | "CATEGORY_INCONSISTENCY"
    column: str
    severity: str              # "HIGH" | "MEDIUM" | "LOW"
    description: str
    affected_rows: int
    affected_row_indices: list = field(default_factory=list)
    metric_value: Optional[float] = None
    threshold_used: Optional[float] = None
    suggested_sql: Optional[str] = None


@dataclass
class ProfilingReport:
    """Full profiling output for one table."""
    table_name: str
    total_rows: int
    total_columns: int
    column_profiles: list[ColumnProfile] = field(default_factory=list)
    anomalies: list[AnomalyRecord] = field(default_factory=list)
    duplicate_count: int = 0
    duplicate_pks: list[Any] = field(default_factory=list)

    @property
    def anomaly_count(self) -> int:
        return len(self.anomalies)

    @property
    def high_severity_count(self) -> int:
        return sum(1 for a in self.anomalies if a.severity == "HIGH")


# ---------------------------------------------------------------------------
# Column Profiling
# ---------------------------------------------------------------------------

def profile_column(series: pd.Series, total_rows: int) -> ColumnProfile:
    """Compute statistics for a single pandas Series."""
    null_count = int(series.isna().sum())
    null_rate = round(null_count / total_rows, 4) if total_rows > 0 else 0.0
    unique_count = int(series.nunique(dropna=True))

    profile = ColumnProfile(
        column_name=series.name,
        dtype=str(series.dtype),
        total_rows=total_rows,
        null_count=null_count,
        null_rate=null_rate,
        unique_count=unique_count,
    )

    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean) > 0:
            profile.mean = round(float(clean.mean()), 4)
            profile.std = round(float(clean.std()), 4)
            profile.min_val = round(float(clean.min()), 4)
            profile.max_val = round(float(clean.max()), 4)
            profile.q1 = round(float(clean.quantile(0.25)), 4)
            profile.q3 = round(float(clean.quantile(0.75)), 4)
            profile.iqr = round(float(profile.q3 - profile.q1), 4)
            profile.skewness = round(float(stats.skew(clean)), 4)
    else:
        # Categorical / text
        profile.top_values = (
            series.value_counts(dropna=True).head(5).to_dict()
        )

    return profile


def profile_dataframe(df: pd.DataFrame, table_name: str = "uploaded_table") -> ProfilingReport:
    """Profile all columns in a DataFrame and return a ProfilingReport."""
    total_rows = len(df)
    report = ProfilingReport(
        table_name=table_name,
        total_rows=total_rows,
        total_columns=len(df.columns),
    )
    for col in df.columns:
        report.column_profiles.append(profile_column(df[col], total_rows))
    return report


# ---------------------------------------------------------------------------
# Anomaly Detectors
# ---------------------------------------------------------------------------

def detect_null_anomalies(
    df: pd.DataFrame,
    report: ProfilingReport,
    threshold: float = config.NULL_RATE_THRESHOLD,
) -> None:
    """Flag columns whose null rate exceeds the threshold."""
    for cp in report.column_profiles:
        if cp.null_rate > threshold:
            severity = "HIGH" if cp.null_rate > 0.20 else "MEDIUM"
            report.anomalies.append(AnomalyRecord(
                anomaly_type="NULL_RATE",
                column=cp.column_name,
                severity=severity,
                description=(
                    f"Column '{cp.column_name}' has {cp.null_rate*100:.1f}% null values "
                    f"({cp.null_count:,} / {cp.total_rows:,} rows) — "
                    f"exceeds threshold of {threshold*100:.0f}%."
                ),
                affected_rows=cp.null_count,
                metric_value=cp.null_rate,
                threshold_used=threshold,
                suggested_sql=(
                    f"-- Inspect null records\n"
                    f"SELECT * FROM {report.table_name} WHERE {cp.column_name} IS NULL;"
                ),
            ))


def detect_duplicates(
    df: pd.DataFrame,
    report: ProfilingReport,
    pk_column: Optional[str] = None,
) -> None:
    """
    Detect duplicated rows.
    If pk_column is provided, check for primary-key violations specifically.
    """
    if pk_column and pk_column in df.columns:
        dup_mask = df.duplicated(subset=[pk_column], keep="first")
        dup_values = df.loc[dup_mask, pk_column].tolist()
        dup_count = int(dup_mask.sum())
        report.duplicate_count = dup_count
        report.duplicate_pks = dup_values[:50]  # cap for display

        if dup_count > 0:
            severity = "HIGH" if dup_count > 5 else "MEDIUM"
            report.anomalies.append(AnomalyRecord(
                anomaly_type="DUPLICATE",
                column=pk_column,
                severity=severity,
                description=(
                    f"Primary-key column '{pk_column}' has {dup_count} duplicate value(s). "
                    f"Sample duplicates: {dup_values[:5]}"
                ),
                affected_rows=dup_count,
                affected_row_indices=df.index[dup_mask].tolist(),
                metric_value=float(dup_count),
                suggested_sql=(
                    f"-- Find duplicate PKs\n"
                    f"SELECT {pk_column}, COUNT(*) AS cnt\n"
                    f"FROM {report.table_name}\n"
                    f"GROUP BY {pk_column}\n"
                    f"HAVING COUNT(*) > 1\n"
                    f"ORDER BY cnt DESC;"
                ),
            ))
    else:
        # Full-row duplicates
        dup_mask = df.duplicated(keep="first")
        dup_count = int(dup_mask.sum())
        report.duplicate_count = dup_count
        if dup_count > 0:
            report.anomalies.append(AnomalyRecord(
                anomaly_type="DUPLICATE",
                column="(all columns)",
                severity="MEDIUM",
                description=f"{dup_count} fully duplicated row(s) detected.",
                affected_rows=dup_count,
                affected_row_indices=df.index[dup_mask].tolist(),
                metric_value=float(dup_count),
                suggested_sql=(
                    f"-- Count rows per group to find full duplicates\n"
                    f"SELECT *, COUNT(*) AS dup_count\n"
                    f"FROM {report.table_name}\n"
                    f"GROUP BY {', '.join(df.columns.tolist())}\n"
                    f"HAVING COUNT(*) > 1;"
                ),
            ))


def detect_outliers_zscore(
    df: pd.DataFrame,
    report: ProfilingReport,
    threshold: float = config.ZSCORE_THRESHOLD,
) -> None:
    """Flag rows with Z-score beyond ±threshold for each numeric column."""
    for cp in report.column_profiles:
        col = cp.column_name
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        clean = df[col].dropna()
        if len(clean) < 10 or cp.std is None or cp.std == 0:
            continue

        z_scores = np.abs(stats.zscore(clean))
        outlier_indices = clean.index[z_scores > threshold].tolist()
        outlier_count = len(outlier_indices)

        if outlier_count > 0:
            report.anomalies.append(AnomalyRecord(
                anomaly_type="OUTLIER_ZSCORE",
                column=col,
                severity="HIGH" if outlier_count > 10 else "MEDIUM",
                description=(
                    f"Column '{col}' has {outlier_count} outlier(s) with |Z-score| > {threshold}. "
                    f"Range in data: [{cp.min_val}, {cp.max_val}], Mean: {cp.mean}, Std: {cp.std}."
                ),
                affected_rows=outlier_count,
                affected_row_indices=outlier_indices[:50],
                metric_value=float(outlier_count),
                threshold_used=threshold,
                suggested_sql=(
                    f"-- Identify Z-score outliers for '{col}'\n"
                    f"WITH stats AS (\n"
                    f"  SELECT AVG({col}) AS mean_val, \n"
                    f"         ({cp.std}) AS std_val\n"
                    f"  FROM {report.table_name}\n"
                    f")\n"
                    f"SELECT * FROM {report.table_name}, stats\n"
                    f"WHERE ABS(({col} - mean_val) / NULLIF(std_val, 0)) > {threshold};"
                ),
            ))


def detect_outliers_iqr(
    df: pd.DataFrame,
    report: ProfilingReport,
    multiplier: float = config.IQR_MULTIPLIER,
) -> None:
    """Flag rows outside the Tukey IQR fence [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    for cp in report.column_profiles:
        col = cp.column_name
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if cp.q1 is None or cp.q3 is None or cp.iqr is None:
            continue
        if cp.iqr == 0:
            continue

        lower = cp.q1 - multiplier * cp.iqr
        upper = cp.q3 + multiplier * cp.iqr

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        outlier_indices = df.index[outlier_mask & df[col].notna()].tolist()
        outlier_count = len(outlier_indices)

        if outlier_count > 0:
            report.anomalies.append(AnomalyRecord(
                anomaly_type="OUTLIER_IQR",
                column=col,
                severity="MEDIUM",
                description=(
                    f"Column '{col}' has {outlier_count} value(s) outside Tukey fence "
                    f"[{lower:.2f}, {upper:.2f}] (IQR×{multiplier}). "
                    f"Q1={cp.q1}, Q3={cp.q3}, IQR={cp.iqr}."
                ),
                affected_rows=outlier_count,
                affected_row_indices=outlier_indices[:50],
                metric_value=float(outlier_count),
                threshold_used=multiplier,
                suggested_sql=(
                    f"-- IQR fence filter for '{col}'\n"
                    f"SELECT * FROM {report.table_name}\n"
                    f"WHERE {col} < {lower:.4f} OR {col} > {upper:.4f};"
                ),
            ))


def detect_category_inconsistencies(
    df: pd.DataFrame,
    report: ProfilingReport,
    min_variant_ratio: float = 0.02,
) -> None:
    """
    Flag categorical columns where normalised (lower-stripped) value counts
    collapse into fewer categories — signalling mixed-case / typos.
    """
    for cp in report.column_profiles:
        col = cp.column_name
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        col_data = df[col].dropna().astype(str)
        if col_data.nunique() == 0:
            continue

        raw_unique = col_data.nunique()
        normalised = col_data.str.strip().str.lower()
        norm_unique = normalised.nunique()

        if raw_unique > norm_unique:
            # Find which raw values map to the same normalised token (pandas 3-compatible)
            norm_series = pd.Series(normalised.values, index=col_data.values, name="norm")
            mapping = {}
            for norm_val, group in col_data.groupby(normalised):
                mapping[norm_val] = list(group.unique())
            inconsistent = {k: v for k, v in mapping.items() if len(v) > 1}
            affected = int(
                col_data.isin(
                    [item for lst in inconsistent.values() for item in lst
                     if item != lst[0]]
                ).sum()
            )
            if affected > 0:
                report.anomalies.append(AnomalyRecord(
                    anomaly_type="CATEGORY_INCONSISTENCY",
                    column=col,
                    severity="LOW",
                    description=(
                        f"Column '{col}' has case/spelling inconsistencies: "
                        f"{raw_unique} raw variants collapse to {norm_unique} after normalisation. "
                        f"Examples: { {k: v for k, v in list(inconsistent.items())[:3]} }"
                    ),
                    affected_rows=affected,
                    metric_value=float(raw_unique - norm_unique),
                    suggested_sql=(
                        f"-- Count raw variants in '{col}'\n"
                        f"SELECT {col}, COUNT(*) AS cnt\n"
                        f"FROM {report.table_name}\n"
                        f"GROUP BY {col}\n"
                        f"ORDER BY LOWER(TRIM({col})), cnt DESC;"
                    ),
                ))


# ---------------------------------------------------------------------------
# Master Run Function
# ---------------------------------------------------------------------------

def run_full_profiling(
    df: pd.DataFrame,
    table_name: str = "uploaded_table",
    pk_column: Optional[str] = None,
) -> ProfilingReport:
    """
    Execute all profiling and anomaly checks on *df*.

    Parameters
    ----------
    df          : Input DataFrame
    table_name  : Name used in suggested SQL snippets
    pk_column   : Primary-key column name for duplicate checks (optional)

    Returns
    -------
    ProfilingReport
    """
    report = profile_dataframe(df, table_name=table_name)
    detect_null_anomalies(df, report)
    detect_duplicates(df, report, pk_column=pk_column)
    detect_outliers_zscore(df, report)
    detect_outliers_iqr(df, report)
    detect_category_inconsistencies(df, report)
    return report


def anomalies_to_dataframe(report: ProfilingReport) -> pd.DataFrame:
    """Convert anomaly records to a flat DataFrame for display."""
    rows = []
    for a in report.anomalies:
        rows.append({
            "Type":           a.anomaly_type,
            "Column":         a.column,
            "Severity":       a.severity,
            "Affected Rows":  a.affected_rows,
            "Metric Value":   a.metric_value,
            "Description":    a.description,
            "Suggested SQL":  a.suggested_sql,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Type", "Column", "Severity", "Affected Rows", "Metric Value", "Description", "Suggested SQL"]
    )
