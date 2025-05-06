# %%
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
import pandas as pd
import plotly.graph_objects as go
import itertools

# %%
# デバイス設定（必要に応じて）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# トレーニングデータ
df = pd.read_csv(
    "../../data/input_test.csv")

x_col = df.columns.drop("美味しさ")
y_col = "美味しさ"
X_train = torch.tensor(df[x_col].values, dtype=torch.float32).to(device)
Y_train = torch.tensor(
    df[y_col].values, dtype=torch.float32).unsqueeze(-1).to(device)

# %%
# モデル構築とフィッティング
bounds = torch.stack([
    X_train.min(dim=0).values,
    X_train.max(dim=0).values
]).to(device)

X_train_norm = normalize(X_train, bounds)

model = SingleTaskGP(
    X_train_norm, Y_train, outcome_transform=Standardize(m=1)).to(device)
mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device)
fit_gpytorch_mll(mll)

# %%

# # === 可視化軸設定 ===
# x_i_idx = 0  # ケニア_ml
# x_j_idx = 2  # コスタリカ_ml

# x_i_vals = torch.linspace(0, 50, 21)
# x_j_vals = torch.linspace(0, 50, 21)
# x_i_grid, x_j_grid = torch.meshgrid(x_i_vals, x_j_vals, indexing='ij')

# # ベース入力（他の変数は平均固定）
# base_input = X_train.median(dim=0).values.cpu()
# inputs = base_input.repeat(x_i_grid.numel(), 1)
# inputs[:, x_i_idx] = x_i_grid.flatten()
# inputs[:, x_j_idx] = x_j_grid.flatten()

# # inputs（予測用グリッド）も同じ bounds で正規化
# inputs_norm = normalize(inputs.to(device), bounds)

# %%

# 各次元の範囲を10刻みで生成
grid_ranges = [
    np.arange(start, end+1.0, 10.0)
    for start, end in zip(bounds[0].cpu(), bounds[1].cpu())
]
# すべての組み合わせを生成
grid_points = list(itertools.product(*grid_ranges))
grid_points
# DataFrameに変換
df_grid = pd.DataFrame(grid_points, columns=x_col)
df_grid
X_grid = torch.tensor(df_grid[x_col].values, dtype=torch.float32).to(device)
X_grid_norm = normalize(X_grid, bounds)
# %%
# === 予測 ===
model.eval()
posterior = model.posterior(X_grid_norm.to(device))
mean = posterior.mean.detach().cpu().squeeze().numpy()
# lower, upper = posterior.mvn.confidence_region()
# .sqrt().view(x_i_grid.shape).detach().cpu().numpy()
std = posterior.variance.sqrt().detach().cpu().squeeze().numpy()

upper = mean + 1.645 * std
lower = mean - 1.645 * std

df_grid["mean"] = mean
df_grid["upper"] = upper
df_grid["lower"] = lower
df_grid["std"] = std
df_grid

# %%
bounds
# %%

# === Plotly描画 ===
x_i_col_name = "ケニア_ml"
x_j_col_name = "コスタリカ_ml"
df_plot = df_grid.groupby(
    [x_i_col_name, x_j_col_name], as_index=False
).mean()

# meshgridを生成（描画用）
x_vals = np.sort(df_plot[x_i_col_name].unique())
y_vals = np.sort(df_plot[x_j_col_name].unique())
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing='ij')


def reshape_surface(x_i_col_name, x_j_col_name, z_col_name):
    z = df_plot.pivot(
        index=x_i_col_name,
        columns=x_j_col_name,
        values=z_col_name
    ).values
    return z


reshape_surface(x_i_col_name, x_j_col_name, "mean")

# %%
df_plot

# %%

fig = go.Figure()

# GP平均
fig.add_trace(go.Surface(
    z=reshape_surface(x_i_col_name, x_j_col_name, "mean"),
    x=x_mesh,  # x_i_grid.cpu().numpy(),
    y=y_mesh,  # x_j_grid.cpu().numpy(),
    colorscale='Rdbu',
    reversescale=True,
    opacity=0.3,
    name='予測平均',
    # colorbar=dict(title="美味しさ")
))

# 上下限
fig.add_trace(go.Surface(
    z=reshape_surface(x_i_col_name, x_j_col_name, "upper"),
    x=x_mesh,  # x_i_grid.cpu().numpy(),
    y=y_mesh,  # x_j_grid.cpu().numpy(),
    opacity=0.6,
    colorscale='Reds',
    autocolorscale=False,
    name='Mean + 1.645σ',
    showscale=False
))

fig.add_trace(go.Surface(
    z=reshape_surface(x_i_col_name, x_j_col_name, "lower"),
    x=x_mesh,  # x_i_grid.cpu().numpy(),
    y=y_mesh,  # x_j_grid.cpu().numpy(),
    opacity=0.6,
    colorscale='Blues',
    autocolorscale=False,
    reversescale=True,
    name='Mean - 1.645',
    showscale=False
))


# === ユーザー試行点を追加 ===
fig.add_trace(go.Scatter3d(
    # x=X_train[:, x_i_idx].cpu().numpy(),
    # y=X_train[:, x_j_idx].cpu().numpy(),
    # z=Y_train.squeeze().cpu().numpy(),
    z=df[y_col],
    x=df[x_i_col_name],
    y=df[x_j_col_name],
    mode='markers',
    marker=dict(
        size=5,
        color='black',
        symbol='circle',
    ),
    name='試行点'
))

# === レイアウト ===
title_name = f"GP予測（平均 ± 1.645σ）と試行点：{x_i_col_name} × {x_j_col_name}"
fig.update_layout(
    title=title_name,
    scene=dict(
        xaxis_title=x_i_col_name,
        yaxis_title=x_j_col_name,
        zaxis_title=y_col,
        xaxis=dict(
            range=[
                df[x_i_col_name].min(),
                df[x_i_col_name].max()
            ]
        ),
        yaxis=dict(
            range=[
                df[x_j_col_name].min(),
                df[x_j_col_name].max()
            ]
        ),
        zaxis=dict(
            range=[
                np.floor(lower.min()),
                np.ceil(upper.max())
            ]
        ),
    ),
    template="plotly_white"
)

fig.write_html(
    "../../data/GP_kenya_tanzania.html",
    include_plotlyjs="cdn",
    full_html=True,
    auto_open=True
)

# %%
bounds

# %%
df_grid

# %%
df_grid.to_csv(
    "../../data/grid_predict.csv",
    index=False,
)
