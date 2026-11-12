# MIT 8339 - Customer Retention Intelligence for a Commercial Bank

BI/DSS virtual lab: attrition analysis, retention-risk scoring and a Plotly dashboard
built on the Kaggle *Credit Card Customers* (BankChurners) dataset.

## Reproduce
```bash
pip install -r requirements.txt
# place BankChurners.csv in data/raw/  (Goyal, S. - Kaggle, see report references)
python run_pipeline.py        # clean -> derive fields -> KPIs -> risk score -> outputs/
python -m src.dashboard       # interactive dashboard HTML
python -m src.export_dashboard  # optional: screenshots + PDF (needs Playwright + Chrome)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_end_to_end.ipynb
```

## Layout
| Path | Purpose |
|---|---|
| `notebooks/end_to_end.ipynb` | Full analysis, executed with outputs (Tasks 1-5 evidence) |
| `src/clean.py` | Profiling, cleaning, derived fields, before/after quality report |
| `src/kpis.py` | KPI definitions, segment tables, attrited-vs-existing tests |
| `src/risk.py` | Rule-based risk score + logistic-regression validation |
| `src/dashboard.py` | Builds the interactive Plotly dashboard (`outputs/retention_dashboard.html`) |
| `src/export_dashboard.py` | Headless-Chrome screenshots and PDF export of the dashboard |
| `data/processed/` | Cleaned and scored datasets |
| `outputs/` | Quality report, KPI/segment tables, model metrics, figures |
