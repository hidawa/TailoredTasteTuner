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
                dbc.NavLink("🧪 次の実験", href="/experiment", active="exact"),
                dbc.NavLink("🛠️ 評価・履歴", href="/evaluation", active="exact"),
                dbc.NavLink("📊 分析", href="/analysis", active="exact"),
                dbc.NavLink("⚙️ 設定", href="/settings", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
    ],
)
