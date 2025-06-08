import pandas as pd
from pydantic import BaseModel, ConfigDict
import numpy as np
from itertools import product
from typing import List, Optional
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.transforms import normalize
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from src.config import DATA_DIR


class CandidatesResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    candidates: pd.DataFrame    # shape: [q, d]
    predictions: pd.DataFrame  # shape: [q, 1]
    gp_model: SingleTaskGP        # 学習済みのGPモデル
    mean: List[float]                # shape: [q]
    variance: List[float]            # shape: [q]


class CandidatesRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    X_train: pd.DataFrame
    Y_train: pd.DataFrame       # shape: [N]
    num_candidates: int = 5     # q
    bounds: Optional[List[List[float]]]   # shape: [2, d], 0〜1などの制約


class CreateCandidates:
    def __init__(self, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, request: CandidatesRequest) -> CandidatesResponse:
        X_train = torch.tensor(
            request.X_train.values, dtype=torch.float64, device=self.device)
        Y_train = torch.tensor(
            request.Y_train.values, dtype=torch.float64, device=self.device)
        # === 1. boundsの決定 ===
        if request.bounds is None:
            bounds = torch.stack(
                [
                    X_train.min(dim=0).values,
                    X_train.max(dim=0).values
                ]
            ).to(self.device)
        else:
            bounds = torch.tensor(
                request.bounds, dtype=torch.float64, device=self.device)

        # === 2. 標準化（normalize to [0, 1]） ===
        X_train_normalized = normalize(
            X_train, bounds=bounds
        ).to(self.device)

        # === 3. GPモデル学習 ===
        model = SingleTaskGP(
            X_train_normalized,
            Y_train,
            outcome_transform=Standardize(m=1)
        ).to(self.device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # === 4. 離散候補点（0.25刻み, 合計=1.0）を生成 ===
        levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        candidate_list = [
            list(p) for p in product(levels, repeat=X_train.shape[1])
            if np.isclose(sum(p), 1.0)
        ]
        X_candidates_raw = torch.tensor(
            candidate_list, dtype=torch.float64, device=self.device)

        # === 4.5. 過去に試した点と一致する候補を除外 ===
        use_existing = False
        if use_existing:
            pass
        else:
            existing = X_train.cpu().numpy()
            mask = []
            for x in X_candidates_raw.cpu().numpy():
                is_duplicate = np.any([np.allclose(x, x0, atol=1e-6)
                                       for x0 in existing])
                mask.append(not is_duplicate)
            X_candidates_raw = X_candidates_raw[mask]

        # === 5. normalize candidates to [0,1] using same bounds ===
        X_candidates_normalized = normalize(X_candidates_raw, bounds=bounds)

        # === 6. qEI 計算 ===
        best_f = model.outcome_transform(Y_train)[0].max()

        use_noise = True
        if use_noise is False:
            acq = qExpectedImprovement(
                model=model,
                best_f=best_f,
                objective=IdentityMCObjective(),
            )
        else:
            # qNoisyExpectedImprovementは、ノイズを考慮したqEI
            acq = qNoisyExpectedImprovement(
                model=model,
                X_baseline=X_train_normalized,
            )

        with torch.no_grad():
            ei_values = acq(X_candidates_normalized.unsqueeze(1)).squeeze(-1)

        # === 7. 上位候補を選出 ===
        topk = torch.topk(ei_values, k=request.num_candidates)
        best_indices = topk.indices
        best_raws = X_candidates_raw[best_indices]  # 非標準化
        best_normalized = X_candidates_normalized[best_indices]

        # 予測値を取得
        posterior = model.posterior(best_normalized)
        mean = posterior.mean.squeeze(-1)
        variance = posterior.variance.squeeze(-1)

        # === 9. 出力用 DataFrame ===
        candidates_df = pd.DataFrame(
            best_raws.cpu().numpy(),
            columns=request.X_train.columns
        )
        predictions_df = pd.DataFrame(
            mean.cpu().detach().numpy().reshape(-1, 1),
            columns=["美味しさ"]
        )

        return CandidatesResponse(
            candidates=candidates_df,
            predictions=predictions_df,
            gp_model=model,
            mean=mean.cpu().tolist(),
            variance=variance.cpu().tolist()
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
    request_data = CandidatesRequest(
        X_train=df.drop(columns=list_target_col),
        Y_train=df[list_target_col],
        num_candidates=2,
        bounds=bounds
    )

    create_candidates = CreateCandidates()
    response = create_candidates(request_data)
    print(response)
