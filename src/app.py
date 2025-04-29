import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
 
app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col(html.Div("Data"), width=6),
            dbc.Col(html.Div("Plot"), width=6),
        ]),
        # dbc.Row([
        #     dbc.Col(dcc.Graph(id='example-graph-1'), width=6),
        #     dbc.Col(dcc.Graph(id='example-graph-2'), width=6),
        # ])
    ]
)
 
if __name__ == '__main__':
    app.run(debug=None, port=8050)
