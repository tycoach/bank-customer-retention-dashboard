"""Run the full analytical pipeline: clean -> score -> KPI/EDA tables.

    python run_pipeline.py
"""
from pathlib import Path
import pandas as pd
from src import clean as C, kpis as K, risk as R

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

cleaned = C.main()
scored = R.add_risk_columns(cleaned)
scored.to_csv(ROOT / "data" / "processed" / "bank_churners_scored.csv", index=False)

print("\n=== HEADLINE KPIs ===")
kp = K.compute_kpis(scored)
for k, v in kp.items():
    print(f"{k:38s} {v:>14,.2f}")
pd.Series(kp, name="value").to_csv(OUT / "kpis.csv")

print("\n=== ATTRITED vs EXISTING ===")
cmp_ = K.compare_groups(scored)
print(cmp_[["variable", "existing_mean", "attrited_mean", "pct_difference", "p_value"]].to_string(index=False))
cmp_.to_csv(OUT / "attrited_vs_existing.csv", index=False)

for col in ["transaction_frequency_band", "spend_trend", "utilisation_band", "products_held_band",
            "age_band", "income_segment", "card_category", "customer_value_class", "risk_band"]:
    t = K.segment_table(scored, col)
    t.to_csv(OUT / f"segment_{col}.csv", index=False)
    print(f"\n-- {col}"); print(t[[col, "customers", "attrited", "attrition_rate_pct"]].to_string(index=False))

print("\n=== RISK SCORE VALIDATION (held-out 30% test) ===")
ev = R.evaluate(scored)
print(f"AUC rule-based score : {ev['auc_rule_test']:.3f}")
print(f"AUC logistic model   : {ev['auc_logit_test']:.3f}")
print("\nBands - TRAIN\n", ev["bands_train"], "\n\nBands - TEST\n", ev["bands_test"])
print("\nTop logistic coefficients\n", ev["logit_top_coefficients"])
ev["bands_train"].to_csv(OUT / "risk_bands_train.csv"); ev["bands_test"].to_csv(OUT / "risk_bands_test.csv")
pd.Series({"auc_rule_test": ev["auc_rule_test"], "auc_logit_test": ev["auc_logit_test"]}).to_csv(OUT / "model_metrics.csv")
