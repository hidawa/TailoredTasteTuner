import pandas as pd
from dash import html, register_page, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from src.config import DATA_DIR
from src.model.create_gridded_prediction_data import GriddedPredictionDataRequest, CreateGriddedPredictionData


register_page(__name__, path="/analysis")  # type: ignore


df_actual = pd.read_csv(
    DATA_DIR / "input_test.csv"
)
list_target_col = ["美味しさ"]

feature_cols = [col for col in df_actual.columns if col not in list_target_col]

# 初期値設定
default_x_i = feature_cols[0]
default_x_j = feature_cols[1]

dropdown_options = [{"label": col, "value": col}
                    for col in df_actual.columns if col not in list_target_col]

# === スライダー生成 ===
slider_components = []
for col in feature_cols:
    # TODO: 20250613 boundsから取得する
    col_min = 0  # int(df_actual[col].min())
    col_max = 1  # int(df_actual[col].max())
    step = 0.25
    slider_components.append(
        html.Div([
            html.Label(f"{col}", className="fw-bold"),
            dcc.RangeSlider(
                id=f"slider-{col}",
                min=col_min,
                max=col_max,
                step=step,
                value=[col_min, col_max],  # ←初期状態：すべて含む
                marks={i: str(i) for i in [0, 0.25, 0.5, 0.75, 1.0]},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ],
        )
    )

layout = html.Div([
    html.H2(
        "📊 モデル結果",
    ),
    dbc.Col([
        html.Label("X軸:"),
        dcc.Dropdown(id="x-i-col", options=[{"label": c, "value": c}
                                            for c in feature_cols], value=default_x_i),
        html.Label("Y軸:"),
        dcc.Dropdown(id="x-j-col", options=[{"label": c, "value": c}
                                            for c in feature_cols], value=default_x_j),
        html.Hr(),
        html.Div(slider_components)
    ],),
    dbc.Col(
        dcc.Graph(
            id="surface-plot",
        ),
    ),
])


@callback(
    Output("surface-plot", "figure"),
    [
        Input("x-i-col", "value"),
        Input("x-j-col", "value"),
    ] + [

        Input(f"slider-{col}", "value") for col in feature_cols
    ],
    # prevent_initial_call=True
)
def update_surface(x_i_col, x_j_col, *slider_ranges):
    # スライダー範囲でフィルタリング
    actual_col = list_target_col[0]  # 実際の値の列名
    bounds = [
        [
            0 for _ in range(
                df_actual.drop(columns=list_target_col).shape[1]
            )
        ],
        [
            1 for _ in range(
                df_actual.drop(columns=list_target_col).shape[1]
            )
        ]
    ]
    gredded_request = GriddedPredictionDataRequest(
        X_train=df_actual.drop(columns=list_target_col),
        Y_train=df_actual[list_target_col],
        bounds=bounds,  # ここは適宜設定
        gridded_levels=[round(x, 2) for x in np.arange(0, 1.01, 0.05)]
    )
    create_gridded_prediction_data = CreateGriddedPredictionData()
    gridded_response = create_gridded_prediction_data(gredded_request)

    df_filtered = gridded_response.df_gridded.copy()
    for col, (vmin, vmax) in zip(feature_cols, slider_ranges):
        if col in [x_i_col, x_j_col]:
            continue
        df_filtered = df_filtered[(df_filtered[col] >= vmin) & (
            df_filtered[col] <= vmax)]

    print(df_filtered)
    # 表示データがないとき
    if df_filtered.empty:
        return go.Figure()

    df_plot = df_filtered.groupby(
        [x_i_col, x_j_col], as_index=False
    ).mean()
    print(df_plot)

    fig = go.Figure()
    # GP平均
    mean_x, mean_y, mean_z = reshape_surface(df_plot, x_i_col, x_j_col, "mean")
    fig.add_trace(go.Surface(
        z=mean_z,
        x=mean_x,
        y=mean_y,
        colorscale='Rdbu',
        reversescale=True,
        opacity=0.3,
        name='予測平均',
    ))

    # 上下限
    upper_x, upper_y, upper_z = reshape_surface(
        df_plot, x_i_col, x_j_col, "upper")
    fig.add_trace(go.Surface(
        z=upper_z,
        x=upper_x,
        y=upper_y,
        opacity=0.6,
        colorscale='Reds',
        autocolorscale=False,
        name='Mean + 1.645σ',
        showscale=False
    ))

    lower_x, lower_y, lower_z = reshape_surface(
        df_plot, x_i_col, x_j_col, "lower")
    fig.add_trace(go.Surface(
        z=lower_z,
        x=lower_x,
        y=lower_y,
        opacity=0.6,
        colorscale='Blues',
        autocolorscale=False,
        reversescale=True,
        name='Mean - 1.645',
        showscale=False
    ))

    fig.add_trace(go.Scatter3d(
        z=df_actual[actual_col],
        x=df_actual[x_i_col],
        y=df_actual[x_j_col],
        mode='markers',
        marker=dict(
            size=5,
            color='black',
            symbol='circle',
        ),
        name='試行点'
    ))

    title_name = f"GP予測（平均 ± 1.645σ）と試行点：{x_i_col} × {x_j_col}"
    fig.update_layout(
        title=title_name,
        scene=dict(
            xaxis_title=x_i_col,
            yaxis_title=x_j_col,
            zaxis_title=actual_col,
            xaxis=dict(
                range=[
                    df_actual[x_i_col].min(),
                    df_actual[x_i_col].max()
                ]
            ),
            yaxis=dict(
                range=[
                    df_actual[x_j_col].min(),
                    df_actual[x_j_col].max()
                ]
            ),
            zaxis=dict(
                range=[
                    np.floor(df_plot["lower"].min()),
                    np.ceil(df_plot["upper"].max())
                ]
            ),
            camera=dict(
                eye=dict(x=-2, y=-2, z=2)  # 手前から見る視点
            )
        ),
    )

    return fig


def reshape_surface(df, x_i_col_name, x_j_col_name, z_col_name):
    df_pivot = df.pivot(
        index=x_j_col_name,
        columns=x_i_col_name,
        values=z_col_name
    )
    x = np.sort(df_pivot.columns.to_numpy())  # 横軸
    y = np.sort(df_pivot.index.to_numpy())    # 縦軸
    z = df_pivot.loc[y, x].values             # ソートされた順に z を取得

    return x, y, z
