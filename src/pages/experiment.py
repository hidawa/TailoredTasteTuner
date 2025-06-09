from dash import html, dcc, register_page, Input, Output, callback  # noqa: F401
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from typing import Union
import pandas as pd
from src.model.return_candidates import CreateCandidates, CandidatesRequest
from src.config import DATA_DIR

register_page(__name__, path="/experiment")


def create_candidate_blocks(df_candidates):
    blocks = []

    for idx, row in df_candidates.iterrows():
        blend_id = idx + 1
        score = row["美味しさ"]

        header_row = dbc.Row(
            [
                dbc.Col(
                    html.H5(f"候補 {blend_id}：美味しさ {score:.2f}"), width="auto"),
                dbc.Col(
                    dbc.Button(
                        "試してみる",
                        id={"type": "try-button", "index": blend_id},
                        color="success",
                        className="ms-3"
                    ),
                    width="auto"
                ),
            ],
            align="center",
            className="mb-2"
        )

        # 材料のカラムだけ抽出（例: ["マンデリン", "グアテマラ", ...]）
        ingredients = row.drop("美味しさ").to_dict()
        row_data = [{"材料名": k, "分量": v} for k, v in ingredients.items()]

        grid = dag.AgGrid(
            columnDefs=[
                {"field": "材料名", "headerName": "材料名"},
                {"field": "分量", "headerName": "分量"},
            ],
            rowData=row_data,
            dashGridOptions={"domLayout": "autoHeight"},
            style={"width": "100%"},
        )
        block = html.Div([header_row, grid], className="mb-4")
        blocks.append(block)

    return blocks


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
            dcc.Loading(
                id="loading-candidates",
                type="circle",
                children=html.Div(
                    id="candidate-blocks"
                ),
            ),
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
    Output("candidate-blocks", "children"),
    Input("create-blend-button", "n_clicks"),
    prevent_initial_call=False
)
def create_blend(n_clicks: int) -> Union[list, None]:
    if n_clicks is None:
        return [
            html.P(
                "ブレンドを作成するにはボタンをクリックしてください。"
            )
        ]

    df = pd.read_csv(
        DATA_DIR / "input_test.csv"
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

    return create_candidate_blocks(df_output)
