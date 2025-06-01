import pandas as pd
from dash import html, register_page, dcc, callback, Input, Output
# import dash_chart_editor as dce
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from src.config import DATA_DIR


register_page(__name__, path="/analysis")  # type: ignore


df_grid = pd.read_csv(
    DATA_DIR / "grid_predict.csv"
)
df_actual = pd.read_csv(
    DATA_DIR / "input_test.csv"
)
actual_col = "美味しさ"

# スライダーにする対象の特徴量（x, y, z軸以外）
predict_cols = [
    "mean",
    "std",
    "upper",
    "lower",
]
feature_cols = [col for col in df_grid.columns if col not in predict_cols]

# 初期値設定
default_x_i = feature_cols[0]
default_x_j = feature_cols[1]

dropdown_options = [{"label": col, "value": col}
                    for col in df_grid.columns if col not in predict_cols]

# === スライダー生成 ===
slider_components = []
for col in feature_cols:
    col_min = int(df_grid[col].min())
    col_max = int(df_grid[col].max())
    step = 10
    slider_components.append(
        html.Div([
            html.Label(f"{col}", className="fw-bold"),
            dcc.RangeSlider(
                id=f"slider-{col}",
                min=col_min,
                max=col_max,
                step=step,
                value=[col_min, col_max],  # ←初期状態：すべて含む
                marks={i: str(i) for i in range(col_min, col_max + 1, step)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ],
        )
    )

layout = html.Div([
    html.H2(
        "📊 モデル結果",
    ),
    html.P(
        "実験結果を表やグラフで表示します。",
    ),
    dbc.Row([
        dbc.Col([
            html.Label("X軸:"),
            dcc.Dropdown(id="x-i-col", options=[{"label": c, "value": c}
                         for c in feature_cols], value=default_x_i),
            html.Label("Y軸:"),
            dcc.Dropdown(id="x-j-col", options=[{"label": c, "value": c}
                         for c in feature_cols], value=default_x_j),
            html.Hr(),
            html.Div(slider_components)
        ], width=4),
        dbc.Col(
            dcc.Graph(
                id="surface-plot",
            ),
            width=8
        ),
    ]),
    # dce.DashChartEditor(dataSources=df_grid.to_dict("list")),
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
    df_filtered = df_grid.copy()
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

    # meshgridの準備
    x_vals = np.sort(df_plot[x_i_col].unique())
    y_vals = np.sort(df_plot[x_j_col].unique())
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing='ij')

    fig = go.Figure()
    # GP平均
    fig.add_trace(go.Surface(
        z=reshape_surface(df_plot, x_i_col, x_j_col, "mean"),
        x=x_mesh,
        y=y_mesh,
        colorscale='Rdbu',
        reversescale=True,
        opacity=0.3,
        name='予測平均',
    ))

    # 上下限
    fig.add_trace(go.Surface(
        z=reshape_surface(df_plot, x_i_col, x_j_col, "upper"),
        x=x_mesh,
        y=y_mesh,
        opacity=0.6,
        colorscale='Reds',
        autocolorscale=False,
        name='Mean + 1.645σ',
        showscale=False
    ))

    fig.add_trace(go.Surface(
        z=reshape_surface(df_plot, x_i_col, x_j_col, "lower"),
        x=x_mesh,
        y=y_mesh,
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
    z = df.pivot(
        index=x_i_col_name,
        columns=x_j_col_name,
        values=z_col_name
    ).values
    return z
