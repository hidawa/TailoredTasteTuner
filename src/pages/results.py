import pandas as pd
from dash import html, register_page
import dash_chart_editor as dce
register_page(__name__, path="/results")  # type: ignore

df = pd.read_csv(
    'https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')

layout = html.Div([
    html.H2(
        "📊 モデル結果",
        # className="text-light"
    ),
    html.P(
        "過去の実験結果を表やグラフで表示します。",
        # className="text-secondary"
    ),
    dce.DashChartEditor(dataSources=df.to_dict("list")),
])
