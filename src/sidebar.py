from dash import html
import dash_bootstrap_components as dbc

sidebar = html.Div(
    [
        html.Hr(),
        html.H4(
            "メニュー", 
            # className="text-white p-3"
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("🏠 ホーム", href="/", active="exact"),
                dbc.NavLink("🧪 実験計画", href="/experiment", active="exact"),
                dbc.NavLink("🛠️ 最適化", href="/optimize", active="exact"),
                dbc.NavLink("📊 結果", href="/results", active="exact"),
                dbc.NavLink("⚙️ 設定", href="/settings", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
    ],
    # style={
    #     "position": "fixed",
    #     "top": 0,
    #     "left": 0,
    #     "bottom": 0,
    #     "width": "16rem",
    #     "padding": "2rem 1rem",
    #     # "background-color": "#222",
    # },
)