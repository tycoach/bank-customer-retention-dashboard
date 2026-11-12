"""Build the interactive Plotly dashboard .
"""
import json
from pathlib import Path

import pandas as pd
import plotly.offline as po

ROOT = Path(__file__).resolve().parents[1]
SCORED = ROOT / "data" / "processed" / "bank_churners_scored.csv"
TEMPLATE = Path(__file__).with_name("dashboard_template.html")
OUT = ROOT / "outputs" / "retention_dashboard.html"

ORDERS = {
    "card_category": ["Blue", "Silver", "Gold", "Platinum"],
    "income_segment": ["Low income", "Lower-middle", "Upper-middle/High", "Unknown"],
    "age_band": ["Under 35", "35-44", "45-54", "55-64", "65+"],
    "gender": ["F", "M"],
    "risk_band": ["Low", "Medium", "High"],
    "transaction_frequency_band": ["Low (<=50)", "Medium (51-80)", "High (81+)"],
    "spend_trend": ["Sharp decline (<0.6)", "Mild decline (0.6-1.0)", "Stable/growing (>1.0)"],
    "products_held_band": ["1-2 products", "3-4 products", "5-6 products"],
    "utilisation_band": ["None (0%)", "Low (0-30%)", "Medium (30-60%)", "High (60%+)"],
}
COLUMNS = [
    "clientnum", "churned", "card_category", "income_segment", "age_band", "gender", "risk_band",
    "risk_score", "churn_probability", "transaction_frequency_band", "spend_trend", "utilisation_band",
    "products_held_band", "total_trans_ct", "total_trans_amt", "credit_limit", "utilisation_ratio",
    "total_relationship_count", "total_revolving_bal", "total_ct_chng_q4_q1",
    "months_inactive_12_mon", "contacts_count_12_mon",
]


def _rate(df, col, value):
    s = df[df[col] == value]
    return s["churned"].mean() * 100, len(s)


def build_insights(df: pd.DataFrame) -> dict:
    """Data-driven sentences shown under each chart (full portfolio)."""
    port = df["churned"].mean() * 100
    lo, _ = _rate(df, "transaction_frequency_band", "Low (<=50)")
    hi, _ = _rate(df, "transaction_frequency_band", "High (81+)")
    r_low, _ = _rate(df, "risk_band", "Low")
    r_high, n_high = _rate(df, "risk_band", "High")
    high_mask = df["risk_band"] == "High"
    hi_share = high_mask.mean() * 100
    hi_value = df.loc[high_mask, "total_trans_amt"].sum() / df["total_trans_amt"].sum() * 100
    hi_caught = df.loc[high_mask, "churned"].sum() / df["churned"].sum() * 100

    cards = {k: df[df.card_category == k].churned.mean() * 100 for k in ORDERS["card_category"]}
    n_plat = int((df.card_category == "Platinum").sum())

    big = lambda col: [df[df[col] == v].churned.mean() * 100 for v in ORDERS[col]
                       if (df[col] == v).sum() >= 300]
    age_r, inc_r = big("age_band"), big("income_segment")
    spread = max(max(age_r) - min(age_r), max(inc_r) - min(inc_r))

    cell = df[(df.transaction_frequency_band == "Low (<=50)") & (df.spend_trend == "Sharp decline (<0.6)")]
    cell_rate, cell_n = cell.churned.mean() * 100, len(cell)
    cell_share = cell.churned.sum() / df.churned.sum() * 100
    calm = df[(df.transaction_frequency_band == "High (81+)") & (df.spend_trend != "Sharp decline (<0.6)")]

    at, ex = df[df.churned == 1], df[df.churned == 0]
    q_at = (at.total_trans_ct <= 60).mean() * 100
    q_ex = (ex.total_trans_ct <= 60).mean() * 100

    idx = lambda c: at[c].mean() / ex[c].mean() * 100
    p12, _ = _rate(df, "products_held_band", "1-2 products")
    p56, _ = _rate(df, "products_held_band", "5-6 products")
    u0, _ = _rate(df, "utilisation_band", "None (0%)")
    u_other = df[df.utilisation_band != "None (0%)"].churned.mean() * 100

    return {
        "A": f"Customers making 50 or fewer transactions a year leave at {lo:.1f}%, about {lo / hi:.0f} times the "
             f"{hi:.1f}% rate of customers with 81 or more. Transaction frequency is the strongest single warning signal in the data.",
        "B": f"The score separates customers cleanly: {r_low:.1f}% attrition in Low against {r_high:.1f}% in High. "
             f"The High band is {hi_share:.1f}% of customers ({n_high:,}) yet contains {hi_caught:.0f}% of all attrited customers, "
             f"so it is the natural target list for retention calls.",
        "C": f"Card tier is a weak driver: Blue {cards['Blue']:.1f}%, Silver {cards['Silver']:.1f}%, Gold {cards['Gold']:.1f}%. "
             f"Platinum ({cards['Platinum']:.0f}%) rests on only {n_plat} customers, so it should not drive strategy.",
        "D": f"Across groups with 300+ customers, attrition varies by at most {spread:.1f} percentage points by age or income. "
             f"Demographics barely explain who leaves, so retention should target behaviour, not who the customer is.",
        "E": f"Customers who are both low-activity and cutting spend churn at {cell_rate:.1f}% ({cell_n:,} customers), "
             f"and account for {cell_share:.0f}% of all attrition. Highly active customers without a sharp decline churn at "
             f"{calm.churned.mean() * 100:.1f}%.",
        "F": f"{q_at:.0f}% of attrited customers made 60 or fewer transactions, against {q_ex:.0f}% of existing customers. "
             f"Attrited customers are already far less active than the rest, so activity-based alerts can flag accounts at risk.",
        "G": f"Attrited customers transact {100 - idx('total_trans_ct'):.0f}% less often and hold "
             f"{100 - idx('total_revolving_bal'):.0f}% less revolving balance than existing customers, "
             f"yet contact the bank {idx('contacts_count_12_mon') - 100:.0f}% more. They are disengaging while contacting the bank more, "
             f"which may indicate unresolved service issues (a hypothesis to test, as the data holds no complaint records).",
        "H": f"Customers with 1-2 products churn at {p12:.1f}% against {p56:.1f}% with 5-6 products. Customers who never "
             f"use their revolving credit churn at {u0:.1f}% against {u_other:.1f}% for everyone else. "
             f"Depth of relationship and card usage both protect retention.",
        "I": f"High-risk customers hold only {hi_value:.1f}% of transaction value. Churn is concentrated in low-activity accounts, "
             f"so a low-cost digital campaign suits most of them, with personal outreach reserved for the highest-value names.",
        "portfolio_rate": port,
    }


def build(path_out: Path = OUT) -> Path:
    df = pd.read_csv(SCORED)
    data = {c: df[c].tolist() for c in COLUMNS}
    for c in ("utilisation_ratio", "total_ct_chng_q4_q1", "churn_probability"):
        data[c] = [round(v, 4) for v in data[c]]
    meta = {"orders": ORDERS, "insights": build_insights(df), "n_total": len(df)}

    def js(obj):
        return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")

    html = (TEMPLATE.read_text(encoding="utf-8")
            .replace("__PLOTLYJS__", po.get_plotlyjs())
            .replace("__DATA__", js(data))
            .replace("__META__", js(meta)))
    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_out.write_text(html, encoding="utf-8")
    print(f"Saved {path_out.relative_to(ROOT)}  ({path_out.stat().st_size / 1e6:.1f} MB)")
    return path_out


if __name__ == "__main__":
    build()
