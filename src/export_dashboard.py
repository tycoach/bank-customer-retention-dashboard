"""Render the Dash app in headless Chrome: 

Starts the app on port 8052 for the capture. Needs Playwright and a local Chrome/Chromium
(set CHROME_PATH if it is not at /usr/bin/google-chrome).
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
CHROME = os.environ.get("CHROME_PATH", "/usr/bin/google-chrome")
PORT = 8052
URL = f"http://127.0.0.1:{PORT}/"


def shoot(page, name):
    path = FIG / name
    page.screenshot(path=str(path), full_page=True)
    return path


def pick(page, key, text):
    page.click(f"#f_{key}")
    page.click(f"[role=option]:has-text('{text}')")
    page.wait_for_timeout(1200)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    errors = []
    server = subprocess.Popen([sys.executable, "-c", f"import app; app.app.run(port={PORT}, debug=False)"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(URL, timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"], headless=True)
            pg = b.new_page(viewport={"width": 1360, "height": 900}, device_scale_factor=1.5)
            pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(URL)
            pg.wait_for_timeout(3500)
            p1 = shoot(pg, "dashboard_page1_overview.png")
            pg.click("text=2 · Customer retention analysis")
            pg.wait_for_timeout(2500)
            p2 = shoot(pg, "dashboard_page2_retention.png")
            # Filter evidence: Risk band = High recomputes every KPI, chart and table.
            pick(pg, "risk_band", "High")
            shoot(pg, "dashboard_page2_filtered_high_risk.png")
            pg.click("text=1 · Executive overview")
            pg.wait_for_timeout(2000)
            shoot(pg, "dashboard_page1_filtered_high_risk.png")
            # "View as table" toggle on the first chart.
            pg.click("#tg_A")
            pg.wait_for_timeout(1000)
            shoot(pg, "dashboard_page1_table_view.png")
            b.close()
    finally:
        server.terminate()
    imgs = [Image.open(x).convert("RGB") for x in (p1, p2)]
    imgs[0].save(ROOT / "outputs" / "retention_dashboard.pdf", save_all=True, append_images=imgs[1:], resolution=150)
    print("Saved screenshots + outputs/retention_dashboard.pdf")
    if errors:
        print("BROWSER ERRORS:", *errors, sep="\n  ")
    return errors


if __name__ == "__main__":
    main()
