import optuna
from optuna.integration import BoTorchSampler
from botorch.utils.transforms import normalize
import torch
from src.model.optuna_model_config import ModelConfig, GPModelBuilder
from src.model.optuna_prediction import your_model_predict
from src.config import DATA_DIR


def run_optuna_optimization(X_train, Y_train, bounds, n_trials=50, step=0.25):
    """
    OptunaでBoTorchSamplerを使った最適化を実行する関数。

    Args:
        X_train (np.ndarray or pd.DataFrame): 訓練データの説明変数（4変数割合ベースなど）
        Y_train (np.ndarray or pd.DataFrame): 訓練データの目的変数（美味しさスコアなど）
        bounds (list of list): 入力の上下限 [[0,...], [1,...]]
        n_trials (int): 試行回数
        step (float): 離散刻み幅（例：0.25）

    Returns:
        study (optuna.study.Study): 最適化結果のStudyオブジェクト
    """

    config = ModelConfig(standardize=True, normalize=True)
    builder = GPModelBuilder(config)

    X_tensor = torch.tensor(X_train, dtype=torch.float64)
    Y_tensor = torch.tensor(Y_train, dtype=torch.float64)
    bounds_tensor = torch.tensor(bounds, dtype=torch.float64)

    if config.normalize:
        X_tensor = normalize(X_tensor, bounds_tensor)

    model = builder.build_model(X_tensor, Y_tensor)

    sampler = BoTorchSampler()

    def objective(trial):
        # 4変数の割合の和が1になるようにサンプル（離散ステップ付き）
        xs = []
        remain = 1.0
        for i in range(X_train.shape[1] - 1):
            xi = trial.suggest_float(f"x{i+1}", 0.0, remain, step=step)
            xs.append(xi)
            remain -= xi
        xi_last = remain
        if xi_last < 0 or xi_last > 1.0:
            raise optuna.exceptions.TrialPruned()
        xs.append(xi_last)
        # x1 = trial.suggest_float("x1", 0.0, 1.0, step=step)
        # x2 = trial.suggest_float("x2", 0.0, 1.0 - x1, step=step)
        # x3 = trial.suggest_float("x3", 0.0, 1.0 - x1 - x2, step=step)
        # x4 = 1.0 - x1 - x2 - x3
        # if x4 < 0 or x4 > 1.0:
        #     raise optuna.exceptions.TrialPruned()

        # x = [x1, x2, x3, x4]

        # ここはyour_model_predictを使う想定（自前予測関数）
        score = your_model_predict(xs, X_train, Y_train, bounds)

        return score

    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study


# 使用例
if __name__ == "__main__":
    import pandas as pd
    # データ読み込み例（適宜差し替え）
    df = pd.read_csv(DATA_DIR / "input_test.csv")
    X_train = df.drop(columns=["美味しさ"]).values
    Y_train = df[["美味しさ"]].values
    bounds = [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]

    study = run_optuna_optimization(
        X_train, Y_train, bounds, n_trials=30, step=0.25)
    print("Best trial:")
    print(study.best_trial.params)
    print("Best value:", study.best_value)
