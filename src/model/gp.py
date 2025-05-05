# %%
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import pandas as pd
import plotly.graph_objects as go

# デバイス設定（必要に応じて）
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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
model = SingleTaskGP(
    X_train, Y_train, outcome_transform=Standardize(m=1)).to(device)
mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device)
fit_gpytorch_mll(mll)

# %%

# === 可視化軸設定 ===
x_i_idx = 0  # ケニア_ml
x_j_idx = 2  # コスタリカ_ml

x_i_vals = torch.linspace(0, 50, 21)
x_j_vals = torch.linspace(0, 50, 21)
x_i_grid, x_j_grid = torch.meshgrid(x_i_vals, x_j_vals, indexing='ij')

# ベース入力（他の変数は平均固定）
base_input = X_train.median(dim=0).values.cpu()
inputs = base_input.repeat(x_i_grid.numel(), 1)
inputs[:, x_i_idx] = x_i_grid.flatten()
inputs[:, x_j_idx] = x_j_grid.flatten()

# %%
# === 予測 ===
model.eval()
posterior = model.posterior(inputs.to(device))
mean = posterior.mean.view(x_i_grid.shape).detach().cpu().numpy()
# lower, upper = posterior.mvn.confidence_region()
std = posterior.variance.sqrt().view(x_i_grid.shape).detach().cpu().numpy()

upper = mean + 1.645 * std
lower = mean - 1.645 * std

# %%

# === Plotly描画 ===
fig = go.Figure()

# GP平均
fig.add_trace(go.Surface(
    z=mean,
    x=x_i_grid.cpu().numpy(),
    y=x_j_grid.cpu().numpy(),
    colorscale='Rdbu',
    reversescale=True,
    opacity=0.3,
    name='予測平均',
    # colorbar=dict(title="美味しさ")
))

# 上下限
fig.add_trace(go.Surface(
    z=upper,
    x=x_i_grid.cpu().numpy(),
    y=x_j_grid.cpu().numpy(),
    opacity=0.6,
    colorscale='Reds',
    autocolorscale=False,
    name='Mean + 1.645σ',
    showscale=False
))

fig.add_trace(go.Surface(
    z=lower,
    x=x_i_grid.cpu().numpy(),
    y=x_j_grid.cpu().numpy(),
    opacity=0.6,
    colorscale='Blues',
    autocolorscale=False,
    reversescale=True,
    name='Mean - 1.645',
    showscale=False
))


# === ユーザー試行点を追加 ===
fig.add_trace(go.Scatter3d(
    x=X_train[:, x_i_idx].cpu().numpy(),
    y=X_train[:, x_j_idx].cpu().numpy(),
    z=Y_train.squeeze().cpu().numpy(),
    mode='markers',
    marker=dict(
        size=5,
        color='black',
        symbol='circle',
    ),
    name='試行点'
))

# === レイアウト ===
fig.update_layout(
    title="GP予測（平均 ± 1.645σ）と試行点：ケニア_ml × コスタリカ_ml",
    scene=dict(
        xaxis_title='ケニア_ml',
        yaxis_title='コスタリカ_ml',
        zaxis_title='美味しさ',
        xaxis=dict(range=[0, 50]),
        yaxis=dict(range=[0, 50]),
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
X_train.median(dim=0).values
# %%
inputs

# %%
