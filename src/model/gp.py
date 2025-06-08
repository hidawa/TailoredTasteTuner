# %%
import pandas as pd
import numpy as np
from itertools import product
import torch
from botorch.utils.transforms import normalize
from src.config import DATA_DIR
from src.model.return_candidates import CreateCandidates, CandidatesRequest
# %%
# デバイス設定（必要に応じて）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

# テスト用のリクエストデータ
df = pd.read_csv(
    DATA_DIR / "input_test.csv"
)
list_target_col = ["美味しさ"]
bounds = [
    [
        0.0 for _ in range(
            df.drop(columns=list_target_col).shape[1]
        )
    ],
    [
        1.0 for _ in range(
            df.drop(columns=list_target_col).shape[1]
        )
    ]
]
request_data = CandidatesRequest(
    X_train=df.drop(columns=list_target_col),
    Y_train=df[list_target_col],
    num_candidates=2,
    bounds=bounds
)

create_candidates = CreateCandidates()
response = create_candidates(request_data)

# %%
bounds_tensor = torch.tensor(
    request_data.bounds, dtype=torch.float64, device=device
)
x_col = response.candidates.columns.tolist()
# %%
levels = [0.0, 0.25, 0.5, 0.75, 1.0]
grid_points = [
    list(p) for p in product(levels, repeat=response.candidates.shape[1])
    if np.isclose(sum(p), 1.0)
]
# grid_points = torch.tensor(
#     candidate_list, dtype=torch.float64, device=device
# )
# # 各次元の範囲を10刻みで生成
# grid_ranges = [
#     np.arange(start, end+0.1, 0.25)
#     for start, end in zip(bounds[0].cpu(), bounds[1].cpu())
# ]
# # すべての組み合わせを生成
# grid_points = list(itertools.product(*grid_ranges))
# DataFrameに変換
df_grid = pd.DataFrame(grid_points, columns=x_col)
df_grid
X_grid = torch.tensor(df_grid[x_col].values, dtype=torch.float32).to(device)
X_grid_norm = normalize(X_grid, bounds_tensor)
df_grid

# %%
# === 予測 ===
response.gp_model.eval()
posterior = response.gp_model.posterior(X_grid_norm.to(device))
mean = posterior.mean.detach().cpu().squeeze().numpy()
std = posterior.variance.sqrt().detach().cpu().squeeze().numpy()

upper = mean + 1.645 * std
lower = mean - 1.645 * std

df_grid["mean"] = mean
df_grid["upper"] = upper
df_grid["lower"] = lower
df_grid["std"] = std
df_grid

# %%
df_grid.to_csv(
    DATA_DIR / "grid_predict.csv",
    index=False,
)
# %%
# %%
