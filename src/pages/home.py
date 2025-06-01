from dash import html, register_page
register_page(__name__, path="/")  # type: ignore

layout = html.Div([
    html.H2(
        "🏠 ホーム",
        # className="text-light"
    ),
    html.P(
        "このアプリは最適なコーヒーブレンドを探索します。",
        # className="text-secondary"
    ),
])
