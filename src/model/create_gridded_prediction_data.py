import pandas as pd
from pydantic import BaseModel, ConfigDict
import numpy as np
from itertools import product
from typing import List, Optional
import torch
from botorch.models import SingleTaskGP
from botorch.utils.transforms import normalize
from src.config import DATA_DIR
from src.model.return_candidates import CreateCandidates, CandidatesRequest


class GriddedPredictionDataResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    df_gridded: pd.DataFrame    # shape: [q, d]
    gp_model: SingleTaskGP        # 学習済みのGPモデル


class GriddedPredictionDataRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    X_train: pd.DataFrame
    Y_train: pd.DataFrame       # shape: [N]
    bounds: Optional[List[List[float]]]   # shape: [2, d], 0〜1などの制約
    # グリッドのレベル（例: [0.0, 0.25, 0.5, 0.75, 1.0]）
    gridded_levels: Optional[List[float]]


class CreateGriddedPredictionData:
    def __init__(self, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, request: GriddedPredictionDataRequest) -> GriddedPredictionDataResponse:
        X_train = request.X_train
        Y_train = request.Y_train
        bounds = request.bounds

        request_data = CandidatesRequest(
            X_train=X_train,
            Y_train=Y_train,
            num_candidates=1,
            bounds=bounds
        )

        create_candidates = CreateCandidates()
        response = create_candidates(request_data)

        bounds_tensor = torch.tensor(
            request_data.bounds, dtype=torch.float64, device=self.device
        )
        x_col = response.candidates.columns.tolist()

        # グリッドポイントの生成
        if request.gridded_levels is not None:
            levels = request.gridded_levels
        else:
            levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        grid_points = [
            list(p) for p in product(levels, repeat=response.candidates.shape[1])
            if np.isclose(sum(p), 1.0)
        ]
        df_gridded = pd.DataFrame(grid_points, columns=x_col)

        X_grid = torch.tensor(
            df_gridded[x_col].values, dtype=torch.float32).to(self.device)
        X_grid_norm = normalize(X_grid, bounds_tensor)

        response.gp_model.eval()
        posterior = response.gp_model.posterior(X_grid_norm.to(self.device))
        mean = posterior.mean.detach().cpu().squeeze().numpy()
        std = posterior.variance.sqrt().detach().cpu().squeeze().numpy()

        upper = mean + 1.645 * std
        lower = mean - 1.645 * std

        df_gridded["mean"] = mean
        df_gridded["upper"] = upper
        df_gridded["lower"] = lower
        df_gridded["std"] = std

        return GriddedPredictionDataResponse(
            df_gridded=df_gridded,
            gp_model=response.gp_model
        )


if __name__ == "__main__":
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
    request_data = GriddedPredictionDataRequest(
        X_train=df.drop(columns=list_target_col),
        Y_train=df[list_target_col],
        bounds=bounds,
        gridded_levels=[-0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25]
    )

    create_candidates = CreateGriddedPredictionData()
    response = create_candidates(request_data)
    print(response)
