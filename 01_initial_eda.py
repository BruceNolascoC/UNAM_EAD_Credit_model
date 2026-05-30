#%%
"""
Initial EDA for the UNAM EAD credit model dataset.

Run this file interactively in VS Code, Spyder, or Jupyter using the #%% cells.
The analysis is intentionally focused on the EAD-factor target:

    Factor = EI / STOTAL at t0

and on the state variables highlighted in modelling_context.md:
current balance, utilization, delinquency, seasoning, and moving averages.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

try:
    import seaborn as sns
except ImportError as exc:
    raise ImportError(
        "This EDA script uses seaborn. Install it with `pip install seaborn` "
        "or adapt the plotting cells to plain matplotlib."
    ) from exc


warnings.filterwarnings("ignore", category=FutureWarning)

try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()
DATA_PATHS = [PROJECT_DIR / "data.csv", PROJECT_DIR / "dat.csv"]
DICTIONARY_PATH = PROJECT_DIR / "data_dictionary.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "eda_initial"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:,.6f}".format)

sns.set_theme(style="whitegrid", context="notebook")
RANDOM_STATE = 42


#%%
# Load data and dictionary

data_path = next((path for path in DATA_PATHS if path.exists()), None)
if data_path is None:
    raise FileNotFoundError(
        "Could not find the base dataset. Expected one of: "
        + ", ".join(str(path.name) for path in DATA_PATHS)
    )

df = pd.read_csv(data_path)

try:
    dictionary = pd.read_csv(DICTIONARY_PATH, encoding="utf-8")
except UnicodeDecodeError:
    dictionary = pd.read_csv(DICTIONARY_PATH, encoding="latin1")

print(f"Loaded data from: {data_path.name}")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
display(df.head())
display(dictionary)


#%%
# Column groups from the data dictionary and modelling context

TARGET = "Factor"
BALANCE_COL = "STOTAL"
UTIL_COL = "utilizacion"
DELINQ_COL = "DiasMora"
AGE_COL = "df_Life_M"
SEGMENT_COL = "mar_emp"

current_state_cols = [DELINQ_COL, BALANCE_COL, AGE_COL, UTIL_COL]
balance_cols = [BALANCE_COL, "prom_saldo_6m", "prom_saldo_12m"]
delinquency_cols = [DELINQ_COL, "prom_diasmora_6m", "prom_diasmora_12m"]
age_cols = [AGE_COL, "prom_df_life_M_6m", "prom_df_life_M_12m"]
utilization_cols = [UTIL_COL, "prom_utilizacion_6m", "prom_utilizacion_12m"]
moving_average_cols = [
    col
    for col in df.columns
    if col.startswith("prom_")
]
numeric_cols = df.select_dtypes(include="number").columns.tolist()
feature_cols = [col for col in numeric_cols if col != TARGET]

missing_expected = [
    col
    for col in [TARGET, BALANCE_COL, UTIL_COL, DELINQ_COL, AGE_COL, SEGMENT_COL]
    if col not in df.columns
]
if missing_expected:
    raise ValueError(f"Missing expected columns: {missing_expected}")

print("Current-state columns:", current_state_cols)
print("Moving-average columns:", moving_average_cols)


#%%
# Basic structure, missing values, duplicates, and data-quality checks

overview = pd.DataFrame(
    {
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": df.isna().sum(),
        "missing_rate": df.isna().mean(),
        "n_unique": df.nunique(dropna=True),
    }
)
display(overview)
overview.to_csv(OUTPUT_DIR / "column_overview.csv")

duplicate_rows = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_rows:,} ({duplicate_rows / len(df):.2%})")

quality_flags = pd.Series(
    {
        "non_positive_STOTAL": (df[BALANCE_COL] <= 0).sum(),
        "negative_Factor": (df[TARGET] < 0).sum(),
        "zero_Factor": (df[TARGET] == 0).sum(),
        "negative_utilizacion": (df[UTIL_COL] < 0).sum(),
        "utilizacion_above_1": (df[UTIL_COL] > 1).sum(),
        "negative_DiasMora": (df[DELINQ_COL] < 0).sum(),
        "negative_df_Life_M": (df[AGE_COL] < 0).sum(),
    }
).to_frame("row_count")
quality_flags["row_rate"] = quality_flags["row_count"] / len(df)
display(quality_flags)
quality_flags.to_csv(OUTPUT_DIR / "quality_flags.csv")


#%%
# Descriptive statistics with tail-focused quantiles

quantiles = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
summary = df[numeric_cols].describe(percentiles=quantiles[1:-1]).T
summary["skew"] = df[numeric_cols].skew(numeric_only=True)
summary["kurtosis"] = df[numeric_cols].kurtosis(numeric_only=True)
display(summary)
summary.to_csv(OUTPUT_DIR / "numeric_summary.csv")

display(df[numeric_cols].quantile(quantiles).T)


#%%
# Target engineering: raw estimated EI and interpretable target bands

df_eda = df.copy()
df_eda["estimated_EI"] = df_eda[BALANCE_COL] * df_eda[TARGET]
df_eda["log_STOTAL"] = np.log1p(df_eda[BALANCE_COL].clip(lower=0))
df_eda["log_estimated_EI"] = np.log1p(df_eda["estimated_EI"].clip(lower=0))
df_eda["log_Factor"] = np.log(df_eda[TARGET].where(df_eda[TARGET] > 0))

factor_bins = [-np.inf, 0, 0.5, 0.95, 1.05, 1.5, 2.0, np.inf]
factor_labels = [
    "<= 0",
    "(0, 0.5]",
    "(0.5, 0.95]",
    "near 1",
    "(1.05, 1.5]",
    "(1.5, 2.0]",
    "> 2.0",
]
df_eda["factor_band"] = pd.cut(df_eda[TARGET], bins=factor_bins, labels=factor_labels)

factor_band_summary = (
    df_eda["factor_band"]
    .value_counts(dropna=False, normalize=False)
    .rename("count")
    .to_frame()
)
factor_band_summary["rate"] = factor_band_summary["count"] / len(df_eda)
display(factor_band_summary)
factor_band_summary.to_csv(OUTPUT_DIR / "factor_band_summary.csv")

display(
    df_eda[[TARGET, "estimated_EI", BALANCE_COL, UTIL_COL, DELINQ_COL, AGE_COL]]
    .quantile(quantiles)
    .T
)


#%%
# Distribution plots for target and central predictors

plot_cols = [TARGET, "estimated_EI", BALANCE_COL, UTIL_COL, DELINQ_COL, AGE_COL]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(), plot_cols):
    sns.histplot(df_eda[col].dropna(), bins=50, kde=True, ax=ax)
    ax.set_title(f"Distribution: {col}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "key_variable_distributions.png", dpi=160)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(df_eda["log_Factor"].dropna(), bins=50, kde=True, ax=axes[0])
axes[0].set_title("log(Factor), positive values only")
sns.histplot(df_eda["log_STOTAL"], bins=50, kde=True, ax=axes[1])
axes[1].set_title("log1p(STOTAL)")
sns.histplot(df_eda["log_estimated_EI"], bins=50, kde=True, ax=axes[2])
axes[2].set_title("log1p(estimated EI)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "log_scale_distributions.png", dpi=160)
plt.show()


#%%
# Segment summaries: target by product/channel flag and business-relevant bands

df_eda["utilization_band"] = pd.cut(
    df_eda[UTIL_COL],
    bins=[-np.inf, 0.05, 0.25, 0.50, 0.75, 0.90, 1.00, np.inf],
    labels=["<=5%", "5-25%", "25-50%", "50-75%", "75-90%", "90-100%", ">100%"],
)
df_eda["delinquency_band"] = pd.cut(
    df_eda[DELINQ_COL],
    bins=[-np.inf, 0, 30, 60, 90, 120, 180, np.inf],
    labels=["0", "1-30", "31-60", "61-90", "91-120", "121-180", ">180"],
)
df_eda["seasoning_band"] = pd.cut(
    df_eda[AGE_COL],
    bins=[-np.inf, 6, 12, 24, 48, 72, 120, np.inf],
    labels=["<=6m", "7-12m", "13-24m", "25-48m", "49-72m", "73-120m", ">120m"],
)


def summarize_target_by(group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df_eda.groupby(group_cols, observed=True)
        .agg(
            n=(TARGET, "size"),
            factor_mean=(TARGET, "mean"),
            factor_median=(TARGET, "median"),
            factor_p10=(TARGET, lambda s: s.quantile(0.10)),
            factor_p90=(TARGET, lambda s: s.quantile(0.90)),
            ei_mean=("estimated_EI", "mean"),
            stotal_mean=(BALANCE_COL, "mean"),
            utilization_mean=(UTIL_COL, "mean"),
        )
        .reset_index()
    )
    grouped["share"] = grouped["n"] / len(df_eda)
    return grouped.sort_values(group_cols)


for group_cols in [[SEGMENT_COL], ["utilization_band"], ["delinquency_band"], ["seasoning_band"]]:
    summary_by_group = summarize_target_by(group_cols)
    display(summary_by_group)
    summary_by_group.to_csv(OUTPUT_DIR / f"target_by_{'_'.join(group_cols)}.csv", index=False)


#%%
# Target by utilization and delinquency together

util_mora_summary = summarize_target_by(["utilization_band", "delinquency_band"])
display(util_mora_summary)
util_mora_summary.to_csv(OUTPUT_DIR / "target_by_utilization_and_delinquency.csv", index=False)

heatmap_data = util_mora_summary.pivot(
    index="delinquency_band",
    columns="utilization_band",
    values="factor_median",
)

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "Median Factor"})
plt.title("Median Factor by utilization and delinquency bands")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "factor_heatmap_utilization_delinquency.png", dpi=160)
plt.show()


#%%
# Current value versus moving-average behaviour

eps = 1e-9
df_eda["saldo_vs_6m_ratio"] = df_eda[BALANCE_COL] / (df_eda["prom_saldo_6m"].abs() + eps)
df_eda["saldo_vs_12m_ratio"] = df_eda[BALANCE_COL] / (df_eda["prom_saldo_12m"].abs() + eps)
df_eda["util_vs_6m_diff"] = df_eda[UTIL_COL] - df_eda["prom_utilizacion_6m"]
df_eda["util_vs_12m_diff"] = df_eda[UTIL_COL] - df_eda["prom_utilizacion_12m"]
df_eda["mora_vs_6m_diff"] = df_eda[DELINQ_COL] - df_eda["prom_diasmora_6m"]
df_eda["mora_vs_12m_diff"] = df_eda[DELINQ_COL] - df_eda["prom_diasmora_12m"]
df_eda["util_6m_vs_12m_diff"] = df_eda["prom_utilizacion_6m"] - df_eda["prom_utilizacion_12m"]
df_eda["saldo_6m_vs_12m_ratio"] = df_eda["prom_saldo_6m"] / (df_eda["prom_saldo_12m"].abs() + eps)

behaviour_cols = [
    "saldo_vs_6m_ratio",
    "saldo_vs_12m_ratio",
    "util_vs_6m_diff",
    "util_vs_12m_diff",
    "mora_vs_6m_diff",
    "mora_vs_12m_diff",
    "util_6m_vs_12m_diff",
    "saldo_6m_vs_12m_ratio",
]

behaviour_summary = df_eda[behaviour_cols + [TARGET]].describe(percentiles=quantiles[1:-1]).T
display(behaviour_summary)
behaviour_summary.to_csv(OUTPUT_DIR / "behaviour_delta_summary.csv")


#%%
# Relationship plots for target against key state variables

sample_size = min(10_000, len(df_eda))
plot_sample = df_eda.sample(sample_size, random_state=RANDOM_STATE)

scatter_specs = [
    ("log_STOTAL", TARGET, "STOTAL on log scale"),
    (UTIL_COL, TARGET, "Utilization"),
    (DELINQ_COL, TARGET, "Days past due"),
    (AGE_COL, TARGET, "Account age in months"),
    ("util_vs_12m_diff", TARGET, "Current utilization minus 12m average"),
    ("saldo_vs_12m_ratio", TARGET, "Current balance / 12m average balance"),
]

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, (x_col, y_col, x_label) in zip(axes.ravel(), scatter_specs):
    sns.scatterplot(
        data=plot_sample,
        x=x_col,
        y=y_col,
        hue=SEGMENT_COL if SEGMENT_COL in plot_sample.columns else None,
        alpha=0.35,
        s=15,
        ax=ax,
        legend=False,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_label}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "factor_relationship_scatterplots.png", dpi=160)
plt.show()


#%%
# Correlation analysis: Pearson and Spearman

analysis_cols = [
    TARGET,
    "estimated_EI",
    BALANCE_COL,
    UTIL_COL,
    DELINQ_COL,
    AGE_COL,
    SEGMENT_COL,
] + moving_average_cols + behaviour_cols
analysis_cols = [col for col in analysis_cols if col in df_eda.columns]

pearson_corr = df_eda[analysis_cols].corr(method="pearson")
spearman_corr = df_eda[analysis_cols].corr(method="spearman")

target_corr = pd.DataFrame(
    {
        "pearson_with_Factor": pearson_corr[TARGET].sort_values(ascending=False),
        "spearman_with_Factor": spearman_corr[TARGET].sort_values(ascending=False),
    }
)
display(target_corr)
target_corr.to_csv(OUTPUT_DIR / "target_correlations.csv")

plt.figure(figsize=(14, 10))
sns.heatmap(spearman_corr, cmap="coolwarm", center=0, linewidths=0.2)
plt.title("Spearman correlation heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "spearman_correlation_heatmap.png", dpi=160)
plt.show()


#%%
# Outlier and influence review for the target and exposure scale

outlier_cols = [TARGET, "estimated_EI", BALANCE_COL, UTIL_COL, DELINQ_COL] + behaviour_cols
outlier_cutoffs = df_eda[outlier_cols].quantile([0.001, 0.01, 0.99, 0.999]).T
outlier_cutoffs.columns = ["p001", "p01", "p99", "p999"]
display(outlier_cutoffs)
outlier_cutoffs.to_csv(OUTPUT_DIR / "outlier_cutoffs.csv")

top_factor = df_eda.nlargest(20, TARGET)[
    [TARGET, "estimated_EI", BALANCE_COL, UTIL_COL, DELINQ_COL, AGE_COL, SEGMENT_COL]
]
bottom_factor = df_eda.nsmallest(20, TARGET)[
    [TARGET, "estimated_EI", BALANCE_COL, UTIL_COL, DELINQ_COL, AGE_COL, SEGMENT_COL]
]
top_exposure = df_eda.nlargest(20, "estimated_EI")[
    [TARGET, "estimated_EI", BALANCE_COL, UTIL_COL, DELINQ_COL, AGE_COL, SEGMENT_COL]
]

display(top_factor)
display(bottom_factor)
display(top_exposure)

top_factor.to_csv(OUTPUT_DIR / "top_20_factor.csv", index=False)
bottom_factor.to_csv(OUTPUT_DIR / "bottom_20_factor.csv", index=False)
top_exposure.to_csv(OUTPUT_DIR / "top_20_estimated_ei.csv", index=False)


#%%
# Boxplots by bands, with target clipped for visual readability

clip_upper = df_eda[TARGET].quantile(0.99)
df_eda["Factor_clipped_p99"] = df_eda[TARGET].clip(upper=clip_upper)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(data=df_eda, x="utilization_band", y="Factor_clipped_p99", ax=axes[0])
axes[0].tick_params(axis="x", rotation=45)
axes[0].set_title("Factor by utilization band")

sns.boxplot(data=df_eda, x="delinquency_band", y="Factor_clipped_p99", ax=axes[1])
axes[1].tick_params(axis="x", rotation=45)
axes[1].set_title("Factor by delinquency band")

sns.boxplot(data=df_eda, x="seasoning_band", y="Factor_clipped_p99", ax=axes[2])
axes[2].tick_params(axis="x", rotation=45)
axes[2].set_title("Factor by account seasoning")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "factor_boxplots_by_bands.png", dpi=160)
plt.show()


#%%
# Missingness and pairwise availability of moving averages

ma_availability = df_eda[moving_average_cols].notna().mean().sort_values(ascending=False)
display(ma_availability.to_frame("availability_rate"))
ma_availability.to_csv(OUTPUT_DIR / "moving_average_availability.csv")

plt.figure(figsize=(10, 4))
sns.barplot(x=ma_availability.index, y=ma_availability.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Non-missing rate")
plt.title("Moving-average variable availability")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "moving_average_availability.png", dpi=160)
plt.show()


#%%
# Export enriched EDA frame and a compact modelling-readiness report

eda_export_cols = [
    col
    for col in df_eda.columns
    if col in df.columns
    or col
    in [
        "estimated_EI",
        "factor_band",
        "utilization_band",
        "delinquency_band",
        "seasoning_band",
        "log_STOTAL",
        "log_estimated_EI",
        "log_Factor",
    ]
    + behaviour_cols
]
try:
    df_eda[eda_export_cols].to_parquet(OUTPUT_DIR / "eda_enriched_dataset.parquet", index=False)
except ImportError:
    df_eda[eda_export_cols].to_csv(OUTPUT_DIR / "eda_enriched_dataset.csv", index=False)

readiness_report = pd.DataFrame(
    {
        "item": [
            "source_file",
            "rows",
            "columns",
            "duplicate_rows",
            "target_missing",
            "target_non_positive",
            "stotal_non_positive",
            "utilization_above_1",
            "factor_p50",
            "factor_p90",
            "factor_p99",
            "estimated_ei_p50",
            "estimated_ei_p90",
            "estimated_ei_p99",
        ],
        "value": [
            data_path.name,
            len(df_eda),
            df_eda.shape[1],
            duplicate_rows,
            df_eda[TARGET].isna().sum(),
            (df_eda[TARGET] <= 0).sum(),
            (df_eda[BALANCE_COL] <= 0).sum(),
            (df_eda[UTIL_COL] > 1).sum(),
            df_eda[TARGET].quantile(0.50),
            df_eda[TARGET].quantile(0.90),
            df_eda[TARGET].quantile(0.99),
            df_eda["estimated_EI"].quantile(0.50),
            df_eda["estimated_EI"].quantile(0.90),
            df_eda["estimated_EI"].quantile(0.99),
        ],
    }
)
display(readiness_report)
readiness_report.to_csv(OUTPUT_DIR / "modelling_readiness_report.csv", index=False)

print(f"EDA outputs written to: {OUTPUT_DIR}")
