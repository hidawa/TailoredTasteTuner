import pandas as pd
from pydantic import BaseModel, ConfigDict

from typing import List, Optional
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll


class CandidatesResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    candidates: pd.DataFrame    # shape: [q, d]
    predictions: pd.DataFrame  # shape: [q, 1]
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
        # 入力データをテンソルに変換
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

        X_train_normalized = normalize(
            X_train, bounds=bounds
        ).to(self.device)

        standard_bounds = torch.zeros_like(bounds).to(self.device)
        standard_bounds[1] = 1

        # ガウス過程モデルの学習
        model = SingleTaskGP(
            X_train_normalized,
            Y_train,
            outcome_transform=Standardize(m=1)
        ).to(self.device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # 最良値（ベイズ最適化に必要）
        best_f = model.outcome_transform(Y_train)[0].max()

        # Acquisition function
        qEI = qExpectedImprovement(model=model, best_f=best_f)

        # 最適化
        candidates, _ = optimize_acqf(
            acq_function=qEI,
            bounds=standard_bounds,
            q=request.num_candidates,
            num_restarts=10,
            raw_samples=64,
        )

        # 予測値を取得
        posterior = model.posterior(candidates)
        if isinstance(posterior, tuple):
            posterior = posterior[0]
        mean = posterior.mean.squeeze(-1)
        variance = posterior.variance.squeeze(-1)

        candidates = unnormalize(candidates.detach(), bounds=bounds)

        candidates_df = pd.DataFrame(
            candidates.cpu().numpy(),
            columns=request.X_train.columns
        )
        predictions_df = pd.DataFrame(
            mean.cpu().detach().numpy(),
            columns=["美味しさ"]
        )

        return CandidatesResponse(
            candidates=candidates_df,
            predictions=predictions_df,
            mean=mean.cpu().tolist(),
            variance=variance.cpu().tolist()
        )


if __name__ == "__main__":
    # テスト用のリクエストデータ
    df = pd.read_csv(
        "./data/input_test.csv"
    )
    request_data = CandidatesRequest(
        X_train=df.drop(columns=["美味しさ"]),
        Y_train=df[["美味しさ"]],
        num_candidates=2,
        bounds=None
    )

    create_candidates = CreateCandidates()
    response = create_candidates(request_data)
    print(response)
