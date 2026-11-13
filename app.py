"""Customer Retention Intelligence Dashboard: a Plotly Dash app (MIT 8339, Task 4).

    python app.py            ->  http://127.0.0.1:8050

Two pages (executive overview, customer retention analysis), eight KPI cards, eight charts, two tables,
five filters that recompute every KPI, chart and table, and a "View as table" toggle on each chart.
Reads data/processed/bank_churners_scored.csv (run `python run_pipeline.py` first).
"""
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, ctx, dcc, html

from src.dash_charts import (BUILDERS, empty_figure, fmt, kpi_calc, kpi_tiles, outreach_table, pct, risk_table)
from src.dashboard import ORDERS, build_insights

ROOT = Path(__file__).resolve().parent
DF = pd.read_csv(ROOT / "data" / "processed" / "bank_churners_scored.csv")
N_TOTAL = len(DF)
PORT_KPI = kpi_calc(DF)
PORT_RATE = PORT_KPI["attr"]
INS = build_insights(DF)

FILTERS = [("card_category", "Card category", None), ("income_segment", "Income segment", None),
           ("age_band", "Age band", None), ("gender", "Gender", {"F": "Female", "M": "Male"}),
           ("risk_band", "Risk band", None)]
FILTER_KEYS = [f[0] for f in FILTERS]

CARDS = [
    dict(id="A", page=1, title="Attrition rate by transaction frequency",
         sub="Customers grouped by number of transactions in the last 12 months",
         why="Why this chart: a short ranked bar chart compares three ordered groups against a portfolio reference line without distorting scale."),
    dict(id="B", page=1, title="Attrition rate by retention-risk band",
         sub="Validates the risk score: attrition should rise from Low to High",
         why="Why this chart: bars on a common zero baseline show whether the score truly separates customers; sample sizes are printed beside each bar."),
    dict(id="C", page=1, title="Attrition rate by card category", sub="Product tier from Blue to Platinum",
         why="Why this chart: sample sizes are shown so a tiny tier (Platinum) is not over-read as a trend."),
    dict(id="D", page=1, title="Attrition rate by age band and income segment",
         sub="Do customer demographics explain attrition?",
         why="Why this chart: one shared axis and the portfolio line make small differences look as small as they are."),
    dict(id="E", page=2, title="Where attrition concentrates: activity x spending trend",
         sub="Attrition rate (colour) and customers (n) for each combination",
         why="Why this chart: a heatmap suits a two-way grid of ordered categories; the numbers in each cell remove any need to judge shade."),
    dict(id="F", page=2, title="Transaction count against transaction amount", sub="Each dot is one customer",
         why="Why this chart: a scatter shows the shape of the two groups, which averages would hide. Two colours only, kept consistent across the dashboard."),
    dict(id="G", page=2, title="How attrited customers differ from existing customers",
         sub="Average of each measure, indexed so existing customers = 100",
         why="Why this chart: indexing puts measures with different units on one axis, so no dual-axis chart is needed."),
    dict(id="H", page=2, title="Attrition by relationship depth and card usage",
         sub="Products held and credit-utilisation band",
         why="Why this chart: the same bar form as the overview lets a reader compare engagement drivers directly."),
]
OUTREACH_NOTE = ("These are the accounts where a personal call protects the most value. Customer IDs are shown for "
                 "operational use only; this list should be access-controlled.")


def table_html(cols, rows):
    return html.Table(className="t", children=[
        html.Thead(html.Tr([html.Th(c) for c in cols])),
        html.Tbody([html.Tr([html.Td(v) for v in r]) for r in rows])])


def note(text, why=None):
    kids = [html.Div(why, className="why")] if why else []
    kids.append(html.Div([html.B("Insight (full portfolio): "), text]))
    return html.Div(kids, className="note")


def chart_card(c):
    cid = c["id"]
    return html.Div(className="card", children=[
        html.Div(className="head", children=[
            html.Div([html.H3(c["title"]), html.Div(c["sub"], className="sub")]),
            html.Button("View as table", id=f"tg_{cid}", className="toggle", n_clicks=0)]),
        dcc.Graph(id=f"chart_{cid}", className="chart", config=dict(displayModeBar=False, responsive=True),
                  style=dict(height="320px")),
        html.Div(id=f"table_{cid}", className="tview", style=dict(display="none")),
        note(INS[cid], c["why"])])


def table_card(div_id, title, sub, text):
    return html.Div(className="card wide", children=[
        html.H3(title), html.Div(sub, className="sub"), html.Div(id=div_id, className="tview full"), note(text)])


app = Dash(__name__, title="Customer Retention Intelligence Dashboard", suppress_callback_exceptions=False)
server = app.server

app.layout = html.Div(className="wrap", children=[
    html.Header([
        html.H1("Customer Retention Intelligence Dashboard"),
        html.P("MIT 8339 Business Intelligence and Decision Support Systems · BankChurners dataset, 10,127 credit-card "
               "customers · Prepared for the Chief Data Officer")]),
    html.Div(className="filters", id="filters", children=[
        html.Div(className="fl", children=[
            html.Div(label, className="flabel"),
            dcc.Dropdown(id=f"f_{key}", options=[dict(label=(names or {}).get(v, v), value=v) for v in ORDERS[key]],
                         value=None, placeholder="All", clearable=True, searchable=False, className="fdrop")])
        for key, label, names in FILTERS] + [
        html.Div(className="spacer"),
        html.Div(id="count", className="count"),
        html.Button("Reset filters", id="reset", className="reset", n_clicks=0)]),
    dcc.Tabs(id="tabs", value="page1", className="tabs", children=[
        dcc.Tab(label="1 · Executive overview", value="page1", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="2 · Customer retention analysis", value="page2", className="tab", selected_className="tab--selected")]),
    html.Div(id="page1", children=[
        html.Div(id="kpis", className="kpis"),
        html.Div(className="grid", children=[chart_card(c) for c in CARDS if c["page"] == 1])]),
    html.Div(id="page2", style=dict(display="none"), children=[
        html.Div(className="grid", children=[chart_card(c) for c in CARDS if c["page"] == 2] + [
            table_card("tbl_risk", "Customer segmentation: retention-risk bands",
                       "Size, attrition and value of each band (responds to filters)", INS["I"]),
            table_card("tbl_list", "Priority outreach list: highest-value high-risk customers",
                       "Top 10 customers in the High band by 12-month transaction amount (responds to filters)", OUTREACH_NOTE)])]),
    html.Footer("Source: Goyal, S. (2020). Credit Card Customers [Data set]. Kaggle. Filters apply to every KPI, chart and table on "
                "both pages. Insight sentences under each chart describe the full portfolio. Risk bands come from a transparent "
                "points score (see report)."),
])


def apply_filters(values) -> pd.DataFrame:
    df = DF
    for key, v in zip(FILTER_KEYS, values):
        if v:
            df = df[df[key] == v]
    return df


filter_inputs = [Input(f"f_{k}", "value") for k in FILTER_KEYS]


@app.callback(Output("count", "children"), Output("kpis", "children"), Output("tbl_risk", "children"),
              Output("tbl_list", "children"), *filter_inputs)
def update_summary(*values):
    df = apply_filters(values)
    active = any(values)
    tiles = [html.Div(className="kpi", children=[html.Div(lab, className="l"), html.Div(val, className="v"),
                                                 html.Div(("Portfolio: " + port) if active else sub, className="s")])
             for lab, val, sub, port in kpi_tiles(df, PORT_KPI, active)]
    count = [html.B(fmt(len(df))), f" of {fmt(N_TOTAL)} customers in view"]
    cols, rows = risk_table(df)
    ocols, orows = outreach_table(df)
    outreach = table_html(ocols, orows) if orows else html.Div("No high-risk customers match the current filters.", className="empty")
    return count, tiles, table_html(cols, rows), outreach


def register_card(cid):
    @app.callback(Output(f"chart_{cid}", "figure"), Output(f"chart_{cid}", "style"), Output(f"table_{cid}", "children"),
                  Output(f"table_{cid}", "style"), Output(f"tg_{cid}", "children"),
                  *filter_inputs, Input(f"tg_{cid}", "n_clicks"))
    def update_card(*args):
        *values, clicks = args
        df = apply_filters(values)
        as_table = (clicks or 0) % 2 == 1
        label = "View as chart" if as_table else "View as table"
        fig, table = BUILDERS[cid](df, PORT_RATE) if len(df) else (None, None)
        if fig is None:
            return empty_figure(), dict(height="320px"), "", dict(display="none"), label
        if as_table:
            return fig, dict(height="320px", display="none"), table_html(table["cols"], table["rows"]), dict(), label
        return fig, dict(height="320px"), "", dict(display="none"), label


for _c in CARDS:
    register_card(_c["id"])


@app.callback(*[Output(f"f_{k}", "value") for k in FILTER_KEYS], Input("reset", "n_clicks"), prevent_initial_call=True)
def reset_filters(_):
    return [None] * len(FILTER_KEYS)


@app.callback(Output("page1", "style"), Output("page2", "style"), Input("tabs", "value"))
def switch_page(tab):
    return (dict(), dict(display="none")) if tab == "page1" else (dict(display="none"), dict())


# Charts drawn while hidden need a resize event once their page becomes visible.
app.clientside_callback(
    "function(v){ setTimeout(function(){ window.dispatchEvent(new Event('resize')); }, 80); return window.dash_clientside.no_update; }",
    Output("tabs", "className"), Input("tabs", "value"))

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
