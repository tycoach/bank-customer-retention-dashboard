"""Report Figures 1-3 (behaviour, behavioural attrition rates, demographics), drawn from the scored data.

    python src/report_figures.py        # from the repo root
    from src import report_figures      # in the notebook: report_figures.main()

Writes outputs/figures/report_fig{1_behaviour,2_behavioural_rates,3_demographics}.png and returns their paths.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
OR, BL, GR = "#E8692F", "#4A90D9", "#555555"   # attrited orange, existing blue, text grey
RC = {"font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.spines.top": False,
      "axes.spines.right": False, "axes.spines.left": False}
LABEL_BOX = dict(fc="white", ec="none", pad=1.2)
PCT60 = ([0, 20, 40, 60], ["0%", "20%", "40%", "60%"])


def _rate_bars(ax, df, col, title, order, base, xlim=60, ticks=PCT60):
    g = df.groupby(col, observed=True).churned.agg(["mean", "size"]).loc[order]
    y = np.arange(len(g))[::-1]
    ax.barh(y, g["mean"] * 100, color=OR, height=.6)
    ax.axvline(base * 100, color="#888", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(g.index, fontsize=8.5)
    for yi, (m, n) in zip(y, g.values):
        ax.text(m * 100 + 1, yi, f"{m * 100:.1f}%  (n={int(n):,})", va="center", fontsize=8, color=GR, bbox=LABEL_BOX, zorder=5)
    ax.set_xlim(0, xlim)
    ax.set_xticks(ticks[0])
    ax.set_xticklabels(ticks[1])
    ax.set_title(title, loc="left", fontsize=9.5, color=GR)
    ax.tick_params(left=False)


def fig_behaviour(d, out):
    """Figure 1: distributions of three behavioural measures, attrited versus existing."""
    spec = [("total_trans_ct", "Transactions in last 12 months", None), ("total_revolving_bal", "Revolving balance ($)", None),
            ("total_ct_chng_q4_q1", "Transaction-count change, Q4 to Q1", (0, 1.75))]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, 3, figsize=(10, 2.9), dpi=200)
        for a, (c, t, r) in zip(ax, spec):
            x = d[c] if r is None else d[c].clip(*r)
            bins = np.histogram_bin_edges(x, bins=30)
            a.hist(x[d.churned == 0], bins=bins, density=True, color=BL, alpha=.65, label="Existing")
            a.hist(x[d.churned == 1], bins=bins, density=True, color=OR, alpha=.65, label="Attrited")
            a.set_title(t, loc="left", fontsize=9.5, color=GR)
            a.set_yticks([])
            a.grid(axis="x", color="#e5e5e5", lw=.6)
            a.set_axisbelow(True)
            m1, m0 = d[c][d.churned == 1].median(), d[c][d.churned == 0].median()
            txt = (f"Median: attrited {m1:.2f}\nexisting {m0:.2f}" if c == "total_ct_chng_q4_q1"
                   else f"Median: attrited {m1:,.0f}\nexisting {m0:,.0f}")
            a.text(.98, .97, txt, transform=a.transAxes, ha="right", va="top", fontsize=7.5, color=GR)
        ax[0].legend(frameon=False, loc="center right", fontsize=8)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()


def fig_rates(d, out, base):
    """Figure 2: attrition rate by transaction frequency, utilisation, products held and bank contacts."""
    d = d.copy()
    d["contacts_grp"] = d.contacts_count_12_mon.astype(int).map(lambda v: "1 contact" if v == 1 else f"{v} contacts")
    with plt.rc_context(RC):
        fig, ax = plt.subplots(2, 2, figsize=(10, 4.6), dpi=200)
        _rate_bars(ax[0, 0], d, "transaction_frequency_band", "Transactions in 12 months", ["Low (<=50)", "Medium (51-80)", "High (81+)"], base)
        _rate_bars(ax[0, 1], d, "utilisation_band", "Card utilisation", ["None (0%)", "Low (0-30%)", "Medium (30-60%)", "High (60%+)"], base)
        _rate_bars(ax[1, 0], d, "products_held_band", "Products held", ["1-2 products", "3-4 products", "5-6 products"], base)
        _rate_bars(ax[1, 1], d, "contacts_grp", "Bank contacts in 12 months",
                   ["0 contacts", "1 contact"] + [f"{k} contacts" for k in range(2, 7)], base, xlim=130,
                   ticks=([0, 50, 100], ["0%", "50%", "100%"]))
        fig.text(.995, .005, "Dashed line: portfolio attrition 16.1%. Contacts panel uses a wider axis.", ha="right", fontsize=7.5, color=GR)
        plt.tight_layout(rect=(0, .02, 1, 1))
        plt.savefig(out)
        plt.close()


def fig_demographics(d, out, base):
    """Figure 3: attrition rate by age band (under 65), income segment and gender on one shared axis."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, 3, figsize=(10, 2.5), dpi=200)
        _rate_bars(ax[0], d[d.age_band.isin(["Under 35", "35-44", "45-54", "55-64"])], "age_band", "Age band",
                   ["Under 35", "35-44", "45-54", "55-64"], base)
        _rate_bars(ax[1], d, "income_segment", "Income segment", ["Low income", "Lower-middle", "Upper-middle/High", "Unknown"], base)
        _rate_bars(ax[2], d, "gender", "Gender", ["F", "M"], base)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()


def main():
    d = pd.read_csv(ROOT / "data" / "processed" / "bank_churners_scored.csv")
    base = d.churned.mean()
    FIG.mkdir(parents=True, exist_ok=True)
    paths = [FIG / "report_fig1_behaviour.png", FIG / "report_fig2_behavioural_rates.png", FIG / "report_fig3_demographics.png"]
    fig_behaviour(d, paths[0])
    fig_rates(d, paths[1], base)
    fig_demographics(d, paths[2], base)
    return paths


if __name__ == "__main__":
    print(*[p.relative_to(ROOT) for p in main()], sep="\n")
