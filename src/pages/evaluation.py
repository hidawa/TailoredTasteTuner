from dash import html, register_page
register_page(__name__, path="/evaluation")  # type: ignore

layout = html.Div([
    html.H2(
        "🛠️ 評価・履歴",
        # className="text-light"
    ),
    html.P(
        "過去に試したブレンドの評価や履歴を確認できます。",
        # className="text-secondary"
    ),
])
