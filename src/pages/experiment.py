from dash import html, dcc, register_page, Input, Output, callback  # noqa: F401
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from typing import Union
import pandas as pd
from src.model.return_candidates import CreateCandidates, CandidatesRequest

register_page(__name__, path="/experiment")

df = pd.read_csv(
    "../data/input_test.csv",
)

grid = dag.AgGrid(
    id="grid-callback-candidates",
    dashGridOptions={
        "rowSelection": "multiple",
        "suppressRowClickSelection": True,
        "animateRows": False
    },
    columnDefs=[
        {"field": x, } for x in df.columns
    ] + [
        {
            "headerName": "試してみる",   # チェックボックスの列
            "checkboxSelection": True,
            "headerCheckboxSelection": True,
            # "width": 50
        }
    ],
    rowData=[],
)

del df


layout = html.Div([
    html.H2(
        "🧪 コーヒーブレンド実験",
        # className="text-light mb-4"
    ),
    dbc.Tabs(
        [
            dbc.Tab(label="ブレンド作成", tab_id="tab-blend"),
            dbc.Tab(label="評価入力", tab_id="tab-eval"),
        ],
        id="experiment-tabs",
        active_tab="tab-blend",
        className="mb-3"
    ),
    html.Div(id="experiment-tab-content")
])


@callback(
    Output("experiment-tab-content", "children"),
    Input("experiment-tabs", "active_tab")
)
def update_tab_content(active_tab: str) -> Union[html.Div, None]:
    if active_tab == "tab-blend":
        return html.Div([
            html.H4(
                "☕ ブレンド作成",
                # className="text-light"
            ),
            html.Button(
                "おすすめブレンドを計算",
                id="create-blend-button",
                className="btn btn-primary mb-3"
            ),
            dcc.Loading(grid),
            html.Button(
                "チェックしたブレンドを保存",
                id="save-blend-button",
                # className="btn btn-primary mb-3"
            ),
        ])
    elif active_tab == "tab-eval":
        return html.Div([
            html.H4(
                "📋 評価入力",
                # className="text-light"
            ),
            html.P(
                "ブレンドの味・香りなどのスコアを入力してください。",
                # className="text-secondary"
            ),
            # 入力フォーム等配置予定
        ])
    return html.Div()


@callback(
    Output("grid-callback-candidates", "rowData", allow_duplicate=True),
    Input("create-blend-button", "n_clicks"),
    prevent_initial_call=True
)
def create_blend(n_clicks: int) -> Union[list, None]:
    if n_clicks is None:
        return [
        ]

    df = pd.read_csv(
        "../data/input_test.csv",
    )
    request_data = CandidatesRequest(
        X_train=df.drop(columns=["美味しさ"]),
        Y_train=df[["美味しさ"]],
        num_candidates=3,
        bounds=None
    )

    create_candidates = CreateCandidates()
    response = create_candidates(request_data)
    df_output = pd.concat(
        [
            response.candidates,
            response.predictions
        ],
        axis=1
    )
    new_blend_data = df_output.to_dict("records")

    return new_blend_data
