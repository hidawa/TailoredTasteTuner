from dash import html, dcc, register_page, Input, Output, callback # noqa: F401
import dash_bootstrap_components as dbc
# from src.app import app
from typing import Union

register_page(__name__, path="/experiment")

layout = html.Div([
    html.H2("🧪 コーヒーブレンド実験", className="text-light mb-4"),
    dbc.Tabs(
        [
            dbc.Tab(label="ブレンド作成", tab_id="tab-blend"),
            dbc.Tab(label="評価入力", tab_id="tab-eval"),
        ],
        id="experiment-tabs",
        active_tab="tab-blend",
        className="mb-3"
    ),
    html.Div(id="experiment-tab-content")
])

@callback(
    Output("experiment-tab-content", "children"),
    Input("experiment-tabs", "active_tab")
)
def update_tab_content(active_tab: str) -> Union[html.Div, None]:
    if active_tab == "tab-blend":
        return html.Div([
            html.H4("☕ ブレンド作成", className="text-light"),
            html.P("ここでコーヒー豆の配合や焙煎を設定します。", className="text-secondary"),
            # フォーム・スライダー等配置予定
        ])
    elif active_tab == "tab-eval":
        return html.Div([
            html.H4("📋 評価入力", className="text-light"),
            html.P("ブレンドの味・香りなどのスコアを入力してください。", className="text-secondary"),
            # 入力フォーム等配置予定
        ])
    return html.Div()