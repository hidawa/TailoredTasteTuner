from dash import html, register_page
register_page(__name__, path="/settings") # type: ignore

layout = html.Div([
    html.H2("⚙️ 設定", className="text-light"),
    html.P("豆の種類、評価軸、初期値などをここで設定します。", className="text-secondary"),
])