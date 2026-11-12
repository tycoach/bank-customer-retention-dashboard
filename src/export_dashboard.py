"""Render the dashboard in headless Chrome: screenshots and a PDF export.

  
"""
import os
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "outputs" / "retention_dashboard.html"
FIG = ROOT / "outputs" / "figures"
CHROME = os.environ.get("CHROME_PATH", "/usr/bin/google-chrome")


def shoot(page, name, full=True):
    path = FIG / name
    page.screenshot(path=str(path), full_page=full)
    return path


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"], headless=True)
        pg = b.new_page(viewport={"width": 1360, "height": 900}, device_scale_factor=1.5)
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(HTML.as_uri()); pg.wait_for_timeout(2500)
        p1 = shoot(pg, "dashboard_page1_overview.png")
        pg.click("#tab2"); pg.wait_for_timeout(2000)
        p2 = shoot(pg, "dashboard_page2_retention.png")
        # filtered view example: Blue card, High risk -> evidence that filters work
        pg.select_option("#f_risk_band", "High"); pg.wait_for_timeout(1500)
        shoot(pg, "dashboard_page2_filtered_high_risk.png")
        pg.click("#tab1"); pg.wait_for_timeout(1500)
        shoot(pg, "dashboard_page1_filtered_high_risk.png")
        b.close()
    imgs = [Image.open(x).convert("RGB") for x in (p1, p2)]
    imgs[0].save(ROOT / "outputs" / "retention_dashboard.pdf", save_all=True, append_images=imgs[1:], resolution=150)
    print("Saved screenshots + outputs/retention_dashboard.pdf")
    if errors:
        print("BROWSER ERRORS:", *errors, sep="\n  ")
    return errors


if __name__ == "__main__":
    main()
