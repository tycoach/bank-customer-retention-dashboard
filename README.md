# MIT 8339 - Customer Retention Intelligence for a Commercial Bank

BI/DSS virtual lab: attrition analysis, retention-risk scoring and a Plotly Dash dashboard
built on the Kaggle *Credit Card Customers* (BankChurners) dataset.

## Run the dashboard
```bash
pip install -r requirements.txt
python app.py                  # then open http://127.0.0.1:8050
```
Two pages (executive overview, customer retention analysis), eight KPI cards, eight charts, two tables and five
filters (card category, income segment, age band, gender, risk band) that recompute every KPI, chart and table.
Each chart has a "View as table" toggle. The app reads `data/processed/bank_churners_scored.csv`, which ships with
the repository.

## Reproduce the analysis
```bash
# place BankChurners.csv in data/raw/  (Goyal, S. - Kaggle, see report references)
python run_pipeline.py            # clean -> derive fields -> KPIs -> risk score -> outputs/
python -m src.export_dashboard    # optional: screenshots + PDF from the Dash app (needs Playwright + Chrome)
python -m src.validate_dashboard  # optional: Dash KPI tiles vs pandas under 4 filter combinations
python src/report_figures.py      # optional: report Figures 1-3
jupyter nbconvert --to notebook --execute --inplace notebooks/01_end_to_end.ipynb
```

## Layout
| Path | Purpose |
|---|---|
| `app.py` | The Plotly Dash dashboard (layout, filters, callbacks) |
| `src/dash_charts.py` | The eight charts, KPI tiles and tables used by the app |
| `assets/dashboard.css` | Dashboard styling (loaded automatically by Dash) |
| `notebooks/01_end_to_end.ipynb` | Full analysis, executed with outputs (Tasks 1-5 evidence) |
| `src/clean.py` | Profiling, cleaning, derived fields, before/after quality report |
| `src/kpis.py` | KPI definitions, segment tables, attrited-vs-existing tests |
| `src/risk.py` | Rule-based risk score + logistic-regression validation |
| `src/dashboard.py` | Data-driven insight sentences, plus a static single-file HTML copy of the dashboard (`outputs/retention_dashboard.html`) for offline viewing |
| `src/export_dashboard.py` | Headless-Chrome screenshots and PDF export of the Dash app |
| `src/validate_dashboard.py` | Checks dashboard KPI tiles against pandas under four filter combinations |
| `src/report_figures.py` | Regenerates report Figures 1-3 (behaviour, behavioural rates, demographics) |
| `data/processed/` | Cleaned and scored datasets |
| `outputs/` | Quality report, KPI/segment tables, model metrics, figures, dashboard exports |
