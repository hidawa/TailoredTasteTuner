import dash
from dash import html, page_container
import dash_bootstrap_components as dbc
from src.navbar import navbar
from src.sidebar import sidebar


app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.SANDSTONE],
    # external_scripts=["https://cdn.plot.ly/plotly-2.18.2.min.js"]
)
server = app.server

app.layout = dbc.Container(
    [
        navbar,
        dbc.Row(
            [
                dbc.Col(sidebar, width=2),
                dbc.Col(
                    html.Div([
                        # navbar,
                        html.Hr(),
                        html.Div(
                            page_container,
                            # className="p-4"
                        ),
                    ]),
                    width=10,
                ),
            ],
            # className="g-0"
        ),
    ],
    fluid=True,
    # className="bg-coffee",
)

if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=8080, debug=False)
