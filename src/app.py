import dash
from dash import html, page_container, Input, Output, State
import dash_bootstrap_components as dbc
from src.navbar import navbar


app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.SANDSTONE],
)
server = app.server

app.layout = html.Div(
    [
        navbar,
        dbc.Container(
            [
                html.Hr(),
                html.Div(page_container),
            ],
            fluid=True,
        )
    ]
)


@app.callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),
    State("navbar-collapse", "is_open"),
)
def toggle_navbar(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
