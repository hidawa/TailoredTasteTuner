from botorch.models.transforms import Standardize
from botorch.models import SingleTaskGP
import torch


class ModelConfig:
    def __init__(self, standardize: bool = True, normalize: bool = True):
        self.standardize = standardize
        self.normalize = normalize


class GPModelBuilder:
    def __init__(self, config: ModelConfig):
        self.config = config

    def build_model(self, X_train: torch.Tensor, Y_train: torch.Tensor) -> SingleTaskGP:
        transforms = {}
        if self.config.standardize:
            transforms["outcome_transform"] = Standardize(m=1)
        model = SingleTaskGP(
            X_train,
            Y_train,
            **transforms
        )
        return model
