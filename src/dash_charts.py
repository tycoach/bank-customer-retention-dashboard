"""Chart, KPI and table builders for the Dash app (Task 4).

Every function here is a Python port of the original JavaScript dashboard, so the Dash app shows the same
graphics: same colours, bar labels, portfolio reference line, heatmap, scatter and indexed comparison.
Each builder takes the filtered DataFrame and returns a Plotly figure plus the rows for its table view.
"""
import math

import pandas as pd
import plotly.graph_objects as go

from src.dashboard import ORDERS

C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
         exist="#2a78d6", attr="#eb6834")
FONT = '-apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'


def fmt(n) -> str:
    return f"{round(n):,}"


def pct(x, d=1) -> str:
    return f"{x:.{d}f}%"


# ---------------------------------------------------------------- shared layout helpers
def base_layout(**extra) -> dict:
    layout = dict(paper_bgcolor=C["surface"], plot_bgcolor=C["surface"], font=dict(family=FONT, color=C["ink2"], size=12),
                  margin=dict(l=10, r=20, t=14, b=40), showlegend=False, hoverlabel=dict(font=dict(family=FONT, size=12)))
    layout.update(extra)
    return layout


def ax(**o) -> dict:
    a = dict(gridcolor=C["grid"], zeroline=False, linecolor=C["grid"], tickfont=dict(color=C["muted"]), automargin=True)
    a.update(o)
    return a


def empty_figure(text="No customers match the current filters.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**base_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                    annotations=[dict(text=text, showarrow=False, font=dict(size=13, color=C["muted"]))]))
    return fig


def rate_by(df: pd.DataFrame, col: str, order: list, label_prefix: str = "", label_suffix: str = "") -> list:
    rows = []
    for o in order:
        s = df[df[col] == o]
        n, a = len(s), int(s["churned"].sum())
        rows.append(dict(label=f"{label_prefix}{o}{label_suffix}", n=n, a=a, r=(a / n * 100) if n else None))
    return rows


def hbar(rows: list, port_rate: float):
    vals = [0 if r["r"] is None else r["r"] for r in rows]
    mx = max([port_rate] + vals)
    xmax = mx * 2.05
    # Rates cannot exceed 100%, so ticks stop at 100% even though the axis leaves room for the labels.
    step = next((s for s in (5, 10, 20, 25, 50) if mx / s <= 6), 50)
    top = min(100, math.ceil(mx / step) * step)
    ticks = list(range(0, int(top) + 1, step))
    fig = go.Figure(go.Bar(
        orientation="h", x=vals, y=[r["label"] for r in rows], marker=dict(color=C["attr"], cornerradius=4),
        customdata=[[r["n"], r["a"]] for r in rows],
        hovertemplate="<b>%{y}</b><br>Attrition rate: %{x:.1f}%<br>%{customdata[0]:,} customers, %{customdata[1]:,} attrited<extra></extra>"))
    # Value labels sit in a fixed right-hand column so they never collide with bars or the reference line.
    notes = [dict(xref="x", yref="y", x=xmax, y=r["label"], xanchor="right", yanchor="middle", showarrow=False, align="right",
                  font=dict(size=12, color=C["ink2"]),
                  text=(f"<b>{r['r']:.1f}%</b>  n={fmt(r['n'])}{' (low n)' if r['n'] < 30 else ''}" if r["n"] else "no customers"))
             for r in rows]
    notes.append(dict(x=port_rate, y=1, yref="paper", yanchor="bottom", text=f"Portfolio {pct(port_rate)}", showarrow=False,
                      font=dict(size=11, color=C["muted"])))
    fig.update_layout(**base_layout(
        bargap=0.5, margin=dict(l=10, r=20, t=22, b=36),
        xaxis=ax(range=[0, xmax], tickmode="array", tickvals=ticks, ticktext=[f"{v}%" for v in ticks],
                 title=dict(text="Attrition rate", font=dict(size=11))),
        yaxis=ax(autorange="reversed", showgrid=False, tickfont=dict(color=C["ink"], size=12)),
        shapes=[dict(type="line", x0=port_rate, x1=port_rate, y0=0, y1=1, yref="paper", layer="below",
                     line=dict(color=C["muted"], width=1.5, dash="dot"))],
        annotations=notes))
    table = dict(cols=["Group", "Customers", "Attrited", "Attrition rate"],
                 rows=[[r["label"], fmt(r["n"]), fmt(r["a"]), pct(r["r"]) if r["n"] else "–"] for r in rows])
    return fig, table


# ---------------------------------------------------------------- the eight charts
def chart_a(df, port):
    return hbar(rate_by(df, "transaction_frequency_band", ORDERS["transaction_frequency_band"]), port)


def chart_b(df, port):
    return hbar(rate_by(df, "risk_band", ORDERS["risk_band"], label_suffix=" risk"), port)


def chart_c(df, port):
    return hbar(rate_by(df, "card_category", ORDERS["card_category"]), port)


def chart_d(df, port):
    rows = (rate_by(df, "age_band", ORDERS["age_band"], "Age: ")
            + rate_by(df, "income_segment", ORDERS["income_segment"], "Income: "))
    return hbar(rows, port)


def chart_e(df, port):
    xs, ys = ORDERS["spend_trend"], ORDERS["transaction_frequency_band"]
    n = [[0] * len(xs) for _ in ys]
    a = [[0] * len(xs) for _ in ys]
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            s = df[(df["transaction_frequency_band"] == y) & (df["spend_trend"] == x)]
            n[r][c], a[r][c] = len(s), int(s["churned"].sum())
    z = [[(a[r][c] / n[r][c] * 100) if n[r][c] else None for c in range(len(xs))] for r in range(len(ys))]
    flat = [v for row in z for v in row if v is not None]
    zmax = max([40] + flat)
    notes = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            v = z[r][c]
            notes.append(dict(x=x, y=y, showarrow=False,
                              text="–" if v is None else f"<b>{v:.1f}%</b><br>n={fmt(n[r][c])}",
                              font=dict(size=12, color="#ffffff" if (v is not None and v > zmax * 0.45) else C["ink"])))
    fig = go.Figure(go.Heatmap(
        x=xs, y=ys, z=z, zmin=0, zmax=zmax, xgap=2, ygap=2, hoverongaps=False,
        colorscale=[[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]],
        colorbar=dict(title=dict(text="Attrition %", font=dict(size=11)), thickness=10, len=0.8, tickfont=dict(color=C["muted"])),
        customdata=n,
        hovertemplate="Transactions: %{y}<br>Spend trend: %{x}<br>Attrition: %{z:.1f}%<br>%{customdata:,} customers<extra></extra>"))
    fig.update_layout(**base_layout(
        margin=dict(l=10, r=10, t=10, b=50), annotations=notes,
        xaxis=ax(side="bottom", showgrid=False, tickfont=dict(color=C["ink"]),
                 title=dict(text="Change in transaction count, Q4 vs Q1", font=dict(size=11))),
        yaxis=ax(autorange="reversed", showgrid=False, tickfont=dict(color=C["ink"]),
                 title=dict(text="Transactions in last 12 months", font=dict(size=11)))))
    rows = [[y, x, fmt(n[r][c]), fmt(a[r][c]), "–" if z[r][c] is None else pct(z[r][c])]
            for r, y in enumerate(ys) for c, x in enumerate(xs)]
    return fig, dict(cols=["Transactions", "Spend trend", "Customers", "Attrited", "Attrition rate"], rows=rows)


def chart_f(df, port):
    ex, at = df[df["churned"] == 0], df[df["churned"] == 1]

    def trace(d, name, color, opacity):
        return go.Scatter(mode="markers", name=name, x=d["total_trans_ct"], y=d["total_trans_amt"],
                          marker=dict(color=color, size=5, opacity=opacity, line=dict(width=0)),
                          hovertemplate=name + "<br>Transactions: %{x}<br>Amount: $%{y:,}<extra></extra>")

    fig = go.Figure([trace(ex, "Existing", C["exist"], 0.45), trace(at, "Attrited", C["attr"], 0.6)])
    fig.update_layout(**base_layout(
        showlegend=True, legend=dict(orientation="h", x=0, y=1.12, font=dict(color=C["ink"])),
        margin=dict(l=10, r=20, t=30, b=44),
        xaxis=ax(title=dict(text="Transactions in last 12 months", font=dict(size=11))),
        yaxis=ax(tickprefix="$", title=dict(text="Total transaction amount, 12m", font=dict(size=11)))))

    def grp(d):
        return [fmt(len(d)), fmt(d["total_trans_ct"].mean()), "$" + fmt(d["total_trans_amt"].mean())] if len(d) else ["0", "–", "–"]

    return fig, dict(cols=["Group", "Customers", "Avg transactions", "Avg amount"],
                     rows=[["Existing"] + grp(ex), ["Attrited"] + grp(at)])


INDEX_VARS = [("total_trans_ct", "Transactions (12m)"), ("total_trans_amt", "Transaction amount"),
              ("total_revolving_bal", "Revolving balance"), ("utilisation_ratio", "Utilisation ratio"),
              ("total_ct_chng_q4_q1", "Txn-count change Q4/Q1"), ("total_relationship_count", "Products held"),
              ("months_inactive_12_mon", "Months inactive"), ("contacts_count_12_mon", "Contacts with bank")]


def chart_g(df, port):
    ex, at = df[df["churned"] == 0], df[df["churned"] == 1]
    if ex.empty or at.empty:
        return None, None
    rows = []
    for col, label in INDEX_VARS:
        e, a = ex[col].mean(), at[col].mean()
        rows.append(dict(label=label, e=e, a=a, idx=(a / e * 100) if e else None))
    fig = go.Figure(go.Bar(
        orientation="h", y=[r["label"] for r in rows], x=[r["idx"] - 100 for r in rows], base=100,
        text=["" if r["idx"] is None else str(round(r["idx"])) for r in rows], textposition="outside", cliponaxis=False,
        textfont=dict(color=C["ink2"]), marker=dict(color=C["attr"], cornerradius=4),
        customdata=[[r["e"], r["a"]] for r in rows],
        hovertemplate="<b>%{y}</b><br>Attrited average: %{customdata[1]:,.2f}<br>Existing average: %{customdata[0]:,.2f}"
                      "<br>Index: %{text} (existing = 100)<extra></extra>"))
    fig.update_layout(**base_layout(
        bargap=0.5, margin=dict(l=10, r=30, t=22, b=44),
        xaxis=ax(range=[30, 150], title=dict(text="Attrited customers as % of the existing-customer average (existing = 100)",
                                             font=dict(size=11))),
        yaxis=ax(autorange="reversed", showgrid=False, tickfont=dict(color=C["ink"])),
        shapes=[dict(type="line", x0=100, x1=100, y0=0, y1=1, yref="paper", line=dict(color=C["exist"], width=2))],
        annotations=[dict(x=100, y=1, yref="paper", yanchor="bottom", text="Existing average = 100", showarrow=False,
                          font=dict(size=11, color=C["exist"]))]))
    table = dict(cols=["Variable", "Existing avg", "Attrited avg", "Index (existing = 100)"],
                 rows=[[r["label"], f"{r['e']:.2f}", f"{r['a']:.2f}", "–" if r["idx"] is None else f"{r['idx']:.0f}"] for r in rows])
    return fig, table


def chart_h(df, port):
    rows = (rate_by(df, "products_held_band", ORDERS["products_held_band"], "Products: ")
            + rate_by(df, "utilisation_band", ORDERS["utilisation_band"], "Utilisation: "))
    return hbar(rows, port)


BUILDERS = dict(A=chart_a, B=chart_b, C=chart_c, D=chart_d, E=chart_e, F=chart_f, G=chart_g, H=chart_h)


# ---------------------------------------------------------------- KPIs and tables
def kpi_calc(df: pd.DataFrame) -> dict:
    n = len(df)
    amt = df["total_trans_amt"].sum()
    high = df["risk_band"] == "High"
    nan = float("nan")
    return dict(
        n=n, attr=df["churned"].mean() * 100 if n else nan, active=100 - df["churned"].mean() * 100 if n else nan,
        lim=df["credit_limit"].mean() if n else nan, util=df["utilisation_ratio"].mean() * 100 if n else nan, amt=amt,
        prod=df["total_relationship_count"].mean() if n else nan, hi=int(high.sum()),
        hi_share=high.mean() * 100 if n else nan, hi_value=df.loc[high, "total_trans_amt"].sum() / amt * 100 if amt else nan)


def kpi_tiles(df: pd.DataFrame, port: dict, active: bool) -> list:
    k = kpi_calc(df)
    nan = lambda v: isinstance(v, float) and math.isnan(v)
    hi_sub = "" if nan(k["hi_share"]) else f"{pct(k['hi_share'])} of customers · {'–' if nan(k['hi_value']) else pct(k['hi_value'])} of txn value"
    return [
        ("Attrition rate", "–" if nan(k["attr"]) else pct(k["attr"]), "Share of customers who left", pct(port["attr"])),
        ("Active customer ratio", "–" if nan(k["active"]) else pct(k["active"]), "Share still with the bank", pct(port["active"])),
        ("Average credit limit", "–" if nan(k["lim"]) else "$" + fmt(k["lim"]), "Mean limit per customer", "$" + fmt(port["lim"])),
        ("Average utilisation", "–" if nan(k["util"]) else pct(k["util"]), "Revolving balance / credit limit", pct(port["util"])),
        ("Total transaction amount", f"${k['amt'] / 1e6:.1f}M", "Sum over the last 12 months", f"${port['amt'] / 1e6:.1f}M"),
        ("Average products held", "–" if nan(k["prod"]) else f"{k['prod']:.2f}", "Bank products per customer", f"{port['prod']:.2f}"),
        ("High-risk customers", fmt(k["hi"]), hi_sub, fmt(port["hi"])),
        ("Transaction value at risk", "–" if nan(k["hi_value"]) else pct(k["hi_value"]), "Txn amount held by High-risk customers", pct(port["hi_value"])),
    ]


def risk_table(df: pd.DataFrame):
    tot = df["total_trans_amt"].sum()
    rows = []
    for b in ORDERS["risk_band"]:
        s = df[df["risk_band"] == b]
        n, a, amt = len(s), int(s["churned"].sum()), s["total_trans_amt"].sum()
        rows.append([b, fmt(n), fmt(a), pct(a / n * 100) if n else "–", "$" + fmt(s["credit_limit"].mean()) if n else "–",
                     "$" + fmt(amt / n) if n else "–", pct(amt / tot * 100) if tot else "–"])
    return ["Risk band", "Customers", "Attrited", "Attrition rate", "Avg credit limit", "Avg txn amount", "Share of txn value"], rows


def outreach_table(df: pd.DataFrame):
    top = df[df["risk_band"] == "High"].sort_values("total_trans_amt", ascending=False, kind="stable").head(10)
    rows = [[str(r.clientnum), r.card_category, int(r.risk_score), pct(r.churn_probability * 100, 0), int(r.total_trans_ct),
             "$" + fmt(r.total_trans_amt), "$" + fmt(r.credit_limit), int(r.months_inactive_12_mon)] for r in top.itertuples()]
    cols = ["Customer ID", "Card", "Risk score (0-12)", "Model churn probability", "Txns (12m)", "Txn amount", "Credit limit", "Months inactive"]
    return cols, rows
