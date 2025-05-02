from dash import html, register_page
register_page(__name__, path="/optimize") # type: ignore

layout = html.Div([
    html.H2("🤖 最適化", className="text-light"),
    html.P("現在の評価をもとに、次に試すべきブレンドを提案します。", className="text-secondary"),
])