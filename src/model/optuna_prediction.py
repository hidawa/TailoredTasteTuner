import torch
from botorch.utils.transforms import normalize, unnormalize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from src.model.optuna_model_config import ModelConfig, GPModelBuilder


def your_model_predict(x: list, X_train, Y_train, bounds):
    config = ModelConfig(standardize=True, normalize=True)
    builder = GPModelBuilder(config)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float64)

    if config.normalize:
        bounds_tensor = torch.tensor(bounds, dtype=torch.float64)
        X_train_tensor = normalize(X_train_tensor, bounds_tensor)
        x_tensor = normalize(
            torch.tensor(
                [x], dtype=torch.float64
            ),
            bounds_tensor)
    else:
        x_tensor = torch.tensor([x], dtype=torch.float64)

    model = builder.build_model(X_train_tensor, Y_train_tensor)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    model.eval()
    with torch.no_grad():
        posterior = model.posterior(x_tensor)
        mean = posterior.mean.item()
        var = posterior.variance.item()
    return mean
