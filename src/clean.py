"""Data profiling, cleaning and transformation (Task 2).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "BankChurners.csv"
CLEAN = ROOT / "data" / "processed" / "bank_churners_clean.csv"
QUALITY = ROOT / "outputs" / "data_quality_report.csv"

EDU_ORDER = ["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate", "Unknown"]
INCOME_ORDER = ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"]
CARD_ORDER = ["Blue", "Silver", "Gold", "Platinum"]
UNKNOWN_COLS = ["education_level", "marital_status", "income_category"]


def load_raw(path: Path = RAW) -> pd.DataFrame:
    return pd.read_csv(path)


def UTIL_COL(d: pd.DataFrame) -> str:
    """Utilisation column name differs before/after the rename in clean()."""
    return "avg_utilization_ratio" if "avg_utilization_ratio" in d else "utilisation_ratio"


def _iqr_flags(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def quality_checks(df: pd.DataFrame) -> dict:
    """Return a dict of data-quality metrics. Works on raw OR cleaned frames."""
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    out = {
        "rows": len(d),
        "columns": d.shape[1],
        "missing_values_total": int(d.isna().sum().sum()),
        "duplicate_rows": int(d.duplicated().sum()),
        "duplicate_customer_ids": int(d["clientnum"].duplicated().sum()),
        "leakage_columns_present": int(sum(c.startswith("naive_bayes") for c in d.columns)),
    }
    for c in ["education_level", "marital_status", "income_category"]:
        out[f"unknown_{c}"] = int((d[c] == "Unknown").sum())
    out["open_to_buy_mismatches"] = int(
        ((d["avg_open_to_buy"] - (d["credit_limit"] - d["total_revolving_bal"])).abs() > 1).sum()
    )
    out["utilisation_mismatches"] = int(
        ((d[UTIL_COL(d)] - d["total_revolving_bal"] / d["credit_limit"]).abs() > 0.01).sum()
    )
    out["tenure_exceeds_adult_life"] = int((d["months_on_book"] > (d["customer_age"] - 18) * 12).sum())
    for c in ["credit_limit", "total_trans_amt", "total_trans_ct"]:
        out[f"iqr_outliers_{c}"] = int(_iqr_flags(d[c]).sum())
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]

    # 1. Drop the two Naive-Bayes columns shipped with the Kaggle file.
    #    They are the output of a classifier trained on the attrition label,
    #    so using them as inputs would leak the target.
    d = d.drop(columns=[c for c in d.columns if c.startswith("naive_bayes")])

    # 2. Identifier as text (it is a label, not a quantity); trim label whitespace.
    d["clientnum"] = d["clientnum"].astype(str)
    for c in d.select_dtypes(include=["object", "string"]).columns:
        d[c] = d[c].str.strip()

    # 3. Target: binary flag + readable label.
    d["churned"] = (d["attrition_flag"] == "Attrited Customer").astype(int)
    d["customer_status"] = d["churned"].map({1: "Attrited", 0: "Existing"})
    d = d.drop(columns="attrition_flag")

    # 4. 'Unknown' labels are KEPT as an explicit category (not imputed).
    #    Evidence: churn among 'Unknown' income customers (16.8%) is almost
    #    identical to the overall rate (16.1%), so imputing the mode would add
    #    bias without adding information. Flags let analysts exclude them.
    for c in UNKNOWN_COLS:
        d[f"{c}_unknown"] = (d[c] == "Unknown").astype(int)

    # 5. Ordered categoricals so charts and tables sort sensibly.
    d["education_level"] = pd.Categorical(d["education_level"], EDU_ORDER, ordered=True)
    d["income_category"] = pd.Categorical(d["income_category"], INCOME_ORDER, ordered=True)
    d["card_category"] = pd.Categorical(d["card_category"], CARD_ORDER, ordered=True)

    # 6. Outliers are FLAGGED, not removed or capped. High credit limits and
    #    transaction totals belong to genuine (often Gold/Platinum) customers;
    #    removing them would erase the bank's most valuable segment.
    d["outlier_credit_limit"] = _iqr_flags(d["credit_limit"]).astype(int)
    d["outlier_trans_amt"] = _iqr_flags(d["total_trans_amt"]).astype(int)

    # 7. Derived analytical fields.
    d["age_band"] = pd.cut(
        d["customer_age"], [0, 34, 44, 54, 64, 120],
        labels=["Under 35", "35-44", "45-54", "55-64", "65+"],
    )
    d["income_segment"] = d["income_category"].map({
        "Less than $40K": "Low income", "$40K - $60K": "Lower-middle", "$60K - $80K": "Lower-middle",
        "$80K - $120K": "Upper-middle/High", "$120K +": "Upper-middle/High", "Unknown": "Unknown",
    }).astype("category")
    d["tenure_band"] = pd.cut(
        d["months_on_book"], [0, 24, 36, 48, 120],
        labels=["13-24 months", "25-36 months", "37-48 months", "49+ months"],
    )
    d["utilisation_band"] = pd.cut(
        d["avg_utilization_ratio"], [-0.001, 0, 0.3, 0.6, 1.0],
        labels=["None (0%)", "Low (0-30%)", "Medium (30-60%)", "High (60%+)"],
    )
    d["transaction_frequency_band"] = pd.cut(
        d["total_trans_ct"], [0, 50, 80, 1000],
        labels=["Low (<=50)", "Medium (51-80)", "High (81+)"],
    )
    d["spend_trend"] = pd.cut(
        d["total_ct_chng_q4_q1"], [-0.001, 0.6, 1.0, 100],
        labels=["Sharp decline (<0.6)", "Mild decline (0.6-1.0)", "Stable/growing (>1.0)"],
    )
    d["customer_value_class"] = pd.qcut(
        d["total_trans_amt"], 3, labels=["Low value", "Mid value", "High value"]
    )
    d["products_held_band"] = pd.cut(
        d["total_relationship_count"], [0, 2, 4, 6], labels=["1-2 products", "3-4 products", "5-6 products"],
    )

    # Plain British-English name for the utilisation column.
    d = d.rename(columns={"avg_utilization_ratio": "utilisation_ratio"})
    return d


def main() -> pd.DataFrame:
    raw = load_raw()
    before = quality_checks(raw)
    cleaned = clean(raw)
    after = quality_checks(cleaned)

    report = pd.DataFrame({"check": list(before), "before_cleaning": list(before.values()),
                           "after_cleaning": [after.get(k, "n/a") for k in before]})
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    QUALITY.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN, index=False)
    report.to_csv(QUALITY, index=False)
    print(report.to_string(index=False))
    print(f"\nSaved {CLEAN.relative_to(ROOT)}  shape={cleaned.shape}")
    return cleaned


if __name__ == "__main__":
    main()
