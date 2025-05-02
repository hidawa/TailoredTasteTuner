import dash
from dash import html, page_container
import dash_bootstrap_components as dbc
from src.navbar import navbar
from src.sidebar import sidebar

app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.DARKLY],
)
server = app.server

app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col(sidebar, width=2),
            dbc.Col(
                html.Div([
                    navbar,
                    html.Div(page_container, className="p-4"),
                ]),
                width=10,
            ),
        ], className="g-0")
    ],
    fluid=True,
    className="bg-dark text-white",
)

if __name__ == "__main__":
    app.run(debug=False)