from dash import html, dcc, register_page, Input, Output, callback  # noqa: F401
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from typing import Union
from datetime import datetime
import pandas as pd
from dash.exceptions import PreventUpdate
from src.model.return_candidates import CreateCandidates, CandidatesRequest
from src.config import BQ_DATASET_ID, BQ_TABLE_ID, SAMPLE_USER_ID, SAMPLE_EXPERIMENT_TYPE
from dash.dependencies import MATCH, State
from src.services.insert_to_bigquery import InsertRecordData, insert_record_to_bigquery
from src.services.extract_dataframe_from_bigquery import extract_dataframe_from_bigquery

register_page(__name__, path="/experiment")


def create_candidate_blocks(df_candidates):
    blocks = []

    for idx, row in df_candidates.iterrows():
        blend_id = idx + 1
        score = row["美味しさ"]

        header_row = dbc.Row(
            [
                dbc.Col(
                    html.H5(
                        f"候補 {blend_id}：美味しさ {score:.2f}",
                        style={"margin": 0, "padding": "0.375rem 0"}
                    ),
                    width="auto",
                    className="d-flex align-items-center",
                ),
                dbc.Col(
                    dbc.Button(
                        "試してみる",
                        id={"type": "try-button", "index": blend_id},
                        color="success",
                        className="ms-3"
                    ),
                    className="d-flex align-items-center",
                    width="auto"
                ),
            ],
            align="center",
            className="mb-2"
        )

        # 評価フォーム（初期は非表示）
        rating_form = dbc.Collapse(
            dbc.Row(
                id={"type": "rating-form", "index": blend_id},
                children=[
                    dbc.Col(html.Label("美味しさを評価（1〜10）"),
                            width="auto", className="pt-2"),
                    dbc.Col(
                        dcc.Input(
                            id={"type": "rating-input", "index": blend_id},
                            type="number",
                            min=1,
                            max=10,
                            step=0.1,
                            style={"width": "100px"}
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dcc.Loading(
                            dbc.Button(
                                "送信",
                                id={"type": "submit-rating", "index": blend_id},
                                color="primary"
                            ),
                            id={"type": "submit-rating-loading",
                                "index": blend_id},
                        ),
                        width="auto",
                        className="d-flex justify-content-end"
                    ),
                ],
                align="center",
                justify="between",
            ),
            id={"type": "rating-collapse", "index": blend_id},
            is_open=False
        )

        # 材料のカラムだけ抽出（例: ["マンデリン", "グアテマラ", ...]）
        ingredients = row.drop("美味しさ").to_dict()
        row_data = [{"材料名": k, "分量": v} for k, v in ingredients.items()]

        grid = dag.AgGrid(
            id={"type": "candidate-grid", "index": blend_id},
            columnDefs=[
                {"field": "材料名", "headerName": "材料名"},
                {"field": "分量", "headerName": "分量"},
            ],
            rowData=row_data,
            dashGridOptions={"domLayout": "autoHeight"},
            style={"width": "100%"},
        )
        block = html.Div(
            id={"type": "candidate-block", "index": blend_id},
            children=[
                header_row,
                rating_form,
                grid
            ],
            className="mb-4",
        )
        blocks.append(block)

    return blocks


layout = html.Div([
    html.H2(
        "🧪 ブレンド作成",
        # className="text-light mb-4"
    ),
    html.Div([
        html.Button(
            "おすすめブレンドを計算",
            id="create-blend-button",
            className="btn btn-primary mb-3"
        ),
        dcc.Loading(
            id="loading-candidates",
            type="circle",
            color="#007bff",
            children=html.Div(
                id="multi-candidate-blocks"
            ),
            target_components={
                "multi-candidate-blocks": "children",
            },
            show_initially=False,
        ),
    ])
])


@callback(
    Output("multi-candidate-blocks", "children"),
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

    df = extract_dataframe_from_bigquery(
        dataset_id=BQ_DATASET_ID,
        table_id=BQ_TABLE_ID,
        user_id=SAMPLE_USER_ID,
        experiment_type=SAMPLE_EXPERIMENT_TYPE
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

    candidate_blocks = create_candidate_blocks(df_output)

    return candidate_blocks


@callback(
    Output({"type": "rating-collapse", "index": MATCH}, "is_open"),
    Input({"type": "try-button", "index": MATCH}, "n_clicks"),
    State({"type": "rating-collapse", "index": MATCH}, "is_open"),
    prevent_initial_call=True
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(
    [
        Output({"type": "candidate-block", "index": MATCH},
               "children"),  # 送信ボタンの表示変更でフィードバック
        Output({"type": "submit-rating", "index": MATCH}, "children"),
    ],
    Input({"type": "submit-rating", "index": MATCH}, "n_clicks"),
    State({"type": "rating-input", "index": MATCH}, "value"),
    State({"type": "candidate-grid", "index": MATCH}, "rowData"),
    prevent_initial_call=True
)
def submit_rating(n_clicks, rating, row_data):
    if n_clicks is None or rating is None or row_data is None:
        raise PreventUpdate

    print(
        f"Rating submitted: {rating}, Row data: {row_data}"
    )
    record = InsertRecordData(
        dataset_id=BQ_DATASET_ID,
        table_id=BQ_TABLE_ID,
        user_id=SAMPLE_USER_ID,
        timestamp=datetime.now().isoformat(),
        experiment_type=SAMPLE_EXPERIMENT_TYPE,
        rating=float(rating),
        ingredients=row_data  # これは [{'材料名': '〜', '分量': x}, ...] の形式
    )

    insert_record_to_bigquery(record)

    return_candidate_block = [
        html.Div(
            [
                dbc.Alert(
                    "✅ 送信が完了しました！",
                    color="success",
                    className="mb-2",
                    style={"backgroundColor": "#d4edda",
                           "color": "#155724", "borderColor": "#c3e6cb"}
                ),
                html.Div(
                    [
                        html.Span("再評価は別ページからお願いします。")
                    ]
                )
            ]
        )
    ]
    return_submit_rating = []

    return [
        return_candidate_block,
        return_submit_rating,
    ]
