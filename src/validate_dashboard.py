"""Check the dashboard's KPI tiles against an independent pandas calculation under four filter combinations.

    python -m src.validate_dashboard            # the Dash app (starts it on port 8051 for the test)
    python -m src.validate_dashboard --html     # the static HTML export (outputs/retention_dashboard.html)

Writes outputs/dashboard_filter_validation.csv. Needs Playwright and a local Chrome/Chromium.
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "outputs" / "retention_dashboard.html"
CHROME = os.environ.get("CHROME_PATH", "/usr/bin/google-chrome")
PORT = 8051

COMBOS = {
    "1. No filters": {},
    "2. Risk band = High": {"risk_band": "High"},
    "3. Card = Blue, Gender = F": {"card_category": "Blue", "gender": "F"},
    "4. Age 45-54, Low income, Risk = Medium": {"age_band": "45-54", "income_segment": "Low income", "risk_band": "Medium"},
}
LABELS = ["Attrition rate", "Active customer ratio", "Average credit limit", "Average utilisation",
          "Total transaction amount", "Average products held", "High-risk customers", "Transaction value at risk"]
GENDER_NAMES = {"F": "Female", "M": "Male"}


def expected(df):
    high = df.risk_band == "High"
    amt = df.total_trans_amt.sum()
    return [f"{df.churned.mean() * 100:.1f}%", f"{100 - df.churned.mean() * 100:.1f}%",
            f"${df.credit_limit.mean():,.0f}", f"{df.utilisation_ratio.mean() * 100:.1f}%",
            f"${amt / 1e6:.1f}M", f"{df.total_relationship_count.mean():.2f}", f"{int(high.sum()):,}",
            f"{df.loc[high, 'total_trans_amt'].sum() / amt * 100:.1f}%"]


def wait_for_server(url, tries=60):
    for _ in range(tries):
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Dash app did not start")


def set_filter_dash(pg, key, value):
    pg.click(f"#f_{key}")
    pg.click(f"[role=option]:has-text('{GENDER_NAMES.get(value, value)}')")
    pg.wait_for_timeout(500)


def main(use_html=False):
    df = pd.read_csv(ROOT / "data" / "processed" / "bank_churners_scored.csv")
    server = None
    if not use_html:
        server = subprocess.Popen([sys.executable, "-c", f"import app; app.app.run(port={PORT}, debug=False)"], cwd=ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_for_server(f"http://127.0.0.1:{PORT}/")
    rows = []
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"], headless=True)
            pg = b.new_page(viewport={"width": 1360, "height": 900})
            pg.goto(HTML.as_uri() if use_html else f"http://127.0.0.1:{PORT}/")
            pg.wait_for_timeout(3500)
            for name, flt in COMBOS.items():
                pg.click("button.reset" if use_html else "#reset")
                pg.wait_for_timeout(700)
                for k, v in flt.items():
                    if use_html:
                        pg.select_option(f"#f_{k}", v)
                        pg.wait_for_timeout(300)
                    else:
                        set_filter_dash(pg, k, v)
                pg.wait_for_timeout(900)
                shown = pg.eval_on_selector_all("#kpis .kpi .v", "els => els.map(e => e.textContent.trim())")
                sub = df
                for k, v in flt.items():
                    sub = sub[sub[k] == v]
                for lab, s, e in zip(LABELS, shown, expected(sub)):
                    rows.append({"filter_combination": name, "customers_in_view": len(sub), "kpi": lab,
                                 "dashboard": s, "pandas": e, "match": "PASS" if s == e else "FAIL"})
            b.close()
    finally:
        if server:
            server.terminate()
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "dashboard_filter_validation.csv", index=False)
    print(out.groupby("filter_combination").match.apply(lambda s: f"{(s == 'PASS').sum()}/{len(s)} KPIs match"))
    print("All pass:", (out.match == "PASS").all(), "|", "HTML export" if use_html else "Dash app")


if __name__ == "__main__":
    main("--html" in sys.argv)
