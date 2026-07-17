from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VBLLLinear(nn.Module):
    """Mean-field variational Bayesian linear layer.

    The layer analytically propagates diagonal weight uncertainty to diagonal
    output uncertainty for a fixed feature vector.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        prior_std: float = 1.0,
        init_rho: float = -5.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_std = prior_std
        self.weight_mu = nn.Parameter(torch.empty(output_dim, input_dim))
        self.weight_rho = nn.Parameter(torch.full((output_dim, input_dim), init_rho))
        if bias:
            self.bias_mu = nn.Parameter(torch.zeros(output_dim))
            self.bias_rho = nn.Parameter(torch.full((output_dim,), init_rho))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        if self.bias_mu is not None:
            fan_in = self.input_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        weight_var = F.softplus(self.weight_rho).pow(2)
        theta_mean = F.linear(features, self.weight_mu, self.bias_mu)
        theta_var = F.linear(features.pow(2), weight_var, None)
        if self.bias_rho is not None:
            theta_var = theta_var + F.softplus(self.bias_rho).pow(2)
        return {"theta_mean": theta_mean, "theta_var": theta_var.clamp_min(1e-12), "kl": self.kl()}

    def kl(self) -> Tensor:
        prior_var = self.prior_std**2
        weight_std = F.softplus(self.weight_rho)
        kl = _normal_kl(self.weight_mu, weight_std, prior_var).sum()
        if self.bias_mu is not None and self.bias_rho is not None:
            kl = kl + _normal_kl(self.bias_mu, F.softplus(self.bias_rho), prior_var).sum()
        return kl


def _normal_kl(mu: Tensor, std: Tensor, prior_var: float) -> Tensor:
    var = std.pow(2)
    return 0.5 * ((var + mu.pow(2)) / prior_var - 1.0 + math.log(prior_var) - var.log())
