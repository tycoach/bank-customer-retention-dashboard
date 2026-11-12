"""KPI definitions, segment tables and attrited-vs-existing comparison (Tasks 1 & 3)."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def compute_kpis(df: pd.DataFrame) -> dict:
    """Seven headline KPIs. `risk_band` is optional (added by src.risk)."""
    k = {
        "Attrition rate (%)": df["churned"].mean() * 100,
        "Active customer ratio (%)": (1 - df["churned"].mean()) * 100,
        "Average credit limit ($)": df["credit_limit"].mean(),
        "Average utilisation ratio (%)": df["utilisation_ratio"].mean() * 100,
        "Total transaction amount, 12m ($)": df["total_trans_amt"].sum(),
        "Average transaction count, 12m": df["total_trans_ct"].mean(),
        "Average products held": df["total_relationship_count"].mean(),
    }
    if "risk_band" in df:
        high = df["risk_band"] == "High"
        k["High-risk customers (count)"] = int(high.sum())
        k["High-risk share of customers (%)"] = high.mean() * 100
        k["Transaction value at risk (%)"] = df.loc[high, "total_trans_amt"].sum() / df["total_trans_amt"].sum() * 100
    return k


def segment_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Customers, attrited customers and attrition rate by a categorical column."""
    t = (df.groupby(col, observed=True)
           .agg(customers=("churned", "size"), attrited=("churned", "sum"),
                avg_credit_limit=("credit_limit", "mean"), avg_trans_amt=("total_trans_amt", "mean"))
           .assign(attrition_rate_pct=lambda x: x["attrited"] / x["customers"] * 100)
           .round(1))
    return t.reset_index()


COMPARE_VARS = {
    "total_trans_ct": "Transactions (12m)",
    "total_trans_amt": "Transaction amount ($)",
    "total_revolving_bal": "Revolving balance ($)",
    "utilisation_ratio": "Utilisation ratio",
    "total_ct_chng_q4_q1": "Txn-count change Q4/Q1",
    "total_relationship_count": "Products held",
    "months_inactive_12_mon": "Months inactive",
    "contacts_count_12_mon": "Contacts with bank (12m)",
    "credit_limit": "Credit limit ($)",
    "customer_age": "Age",
}


def compare_groups(df: pd.DataFrame, variables: dict = COMPARE_VARS) -> pd.DataFrame:
    """Existing vs attrited: means, medians, % difference and Mann-Whitney U p-value.

    Mann-Whitney is used (not a t-test) because most of these variables are
    skewed or bounded, so a rank-based test is the safer choice.
    """
    rows = []
    ex, at = df[df.churned == 0], df[df.churned == 1]
    for col, label in variables.items():
        p = mannwhitneyu(ex[col], at[col], alternative="two-sided").pvalue
        rows.append({
            "variable": label,
            "existing_mean": ex[col].mean(), "attrited_mean": at[col].mean(),
            "existing_median": ex[col].median(), "attrited_median": at[col].median(),
            "pct_difference": (at[col].mean() / ex[col].mean() - 1) * 100 if ex[col].mean() else np.nan,
            "p_value": p,
        })
    out = pd.DataFrame(rows)
    out["significant_5pct"] = out["p_value"] < 0.05
    return out.round(4)
