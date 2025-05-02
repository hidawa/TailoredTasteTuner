from dash import html, register_page
register_page(__name__, path="/results") # type: ignore

layout = html.Div([
    html.H2("📊 結果", className="text-light"),
    html.P("過去の実験結果を表やグラフで表示します。", className="text-secondary"),
])