"""Counterexample-guided (CEGIS) training of a neural Lyapunov candidate.

The result is *empirical*: a candidate with no counterexample found by the falsifier is NOT a
proof. Certification requires exporting V and checking the conditions formally (Lean)."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn


class NeuralLyapunovCandidate(nn.Module):
    """V(x) = φ(x) − φ(0) + ε‖x‖²  (so V(0) = 0 exactly)."""

    def __init__(self, state_dim: int, hidden_dim: int = 64, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, 1, bias=False))
        self.register_buffer("origin", torch.zeros(1, state_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.net(x) - self.net(self.origin) + self.epsilon * (x ** 2).sum(-1, keepdim=True)).squeeze(-1)


class CEGISLyapunovTrainer:
    def __init__(self, dynamics: Callable[[torch.Tensor], torch.Tensor], state_dim: int,
                 domain_radius: float = 1.0, hidden_dim: int = 64, lr: float = 1e-3,
                 batch_size: int = 512, margin: float = 1e-2, device: str = "cpu"):
        self.dynamics, self.state_dim, self.radius = dynamics, state_dim, domain_radius
        self.batch_size, self.margin, self.device = batch_size, margin, device
        self.model = NeuralLyapunovCandidate(state_dim, hidden_dim).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = torch.empty(0, state_dim, device=device)

    def _sample(self, n: int) -> torch.Tensor:
        d = torch.randn(n, self.state_dim, device=self.device)
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
        r = self.radius * torch.rand(n, 1, device=self.device) ** (1.0 / self.state_dim)
        return d * r

    def _lie(self, x: torch.Tensor) -> torch.Tensor:
        """⟨∇V(x), f(x)⟩ for a leaf tensor `x` with requires_grad=True (differentiable in x)."""
        v = self.model(x)
        grad = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        return (grad * self.dynamics(x)).sum(-1)

    def compute_lie_derivative(self, x: torch.Tensor):
        x = x.detach().requires_grad_(True)
        return self.model(x), self._lie(x)

    def train_step(self) -> Dict[str, float]:
        x = torch.cat([self._sample(self.batch_size), self.buffer])
        v, lie = self.compute_lie_derivative(x)
        sq = (x.detach() ** 2).sum(-1)
        violation = torch.relu(lie + self.margin * sq)         # want  lie ≤ −margin‖x‖²
        loss = violation.mean() + torch.relu(1e-3 * sq - v).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {"total_loss": float(loss), "max_lie_violation": float(violation.max())}

    def find_counterexamples(self, num_searches: int = 500, steps: int = 100,
                             step_size: float = 0.05) -> torch.Tensor:
        """Gradient ascent on the Lie derivative from random starts; returns violating points (k, d)."""
        x = self._sample(num_searches).requires_grad_(True)
        for _ in range(steps):
            with torch.enable_grad():
                g = torch.autograd.grad(self._lie(x).sum(), x)[0]
            with torch.no_grad():
                x += step_size * self.radius * g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                scale = (self.radius / x.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
                x *= scale                                         # project back onto the ball
        with torch.enable_grad():
            lie = self._lie(x).detach()
        pts = x.detach()
        bad = (lie > -self.margin * (pts ** 2).sum(-1)) & (pts.norm(dim=1) > 1e-2)
        return pts[bad]

    def run_cegis_loop(self, max_outer_loops: int = 10, epochs_per_loop: int = 500) -> nn.Module:
        for _ in range(max_outer_loops):
            for _ in range(epochs_per_loop):
                self.train_step()
            cex = self.find_counterexamples()
            if cex.shape[0] == 0:
                break
            self.buffer = torch.cat([self.buffer, cex])[-10_000:]
        return self.model
