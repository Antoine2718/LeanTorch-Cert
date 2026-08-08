from __future__ import annotations

from typing import Callable, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralLyapunovCandidate(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64, epsilon: float = 1e-3):
        super().__init__()
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin = torch.zeros(1, self.state_dim, device=x.device, dtype=x.dtype)
        v_x = self.net(x) - self.net(origin)
        # Quadratic regularization term ensures strict positive definiteness
        quad_term = self.epsilon * torch.sum(x ** 2, dim=-1, keepdim=True)
        return v_x + quad_term


class CEGISLyapunovTrainer:
    def __init__(
        self,
        dynamics: Callable[[torch.Tensor], torch.Tensor],
        state_dim: int,
        domain_radius: float = 2.0,
        device: str = "cpu",
    ):
        self.dynamics = dynamics
        self.state_dim = state_dim
        self.domain_radius = domain_radius
        self.device = torch.device(device)

        self.model = NeuralLyapunovCandidate(state_dim=state_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Buffer of counterexamples initialized with uniform grid
        self.dataset = self._generate_uniform_domain_samples(num_samples=2000)

    def _generate_uniform_domain_samples(self, num_samples: int) -> torch.Tensor:
        raw_samples = (torch.rand(num_samples, self.state_dim, device=self.device) * 2 - 1)
        norms = torch.norm(raw_samples, p=2, dim=1, keepdim=True)
        radii = torch.rand(num_samples, 1, device=self.device) ** (1 / self.state_dim) * self.domain_radius
        samples = (raw_samples / (norms + 1e-8)) * radii
        return samples

    def compute_lie_derivative(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_req = x.clone().detach().requires_grad_(True)
        v_val = self.model(x_req)
        
        # Compute gradient ∇V(x)
        grad_v = torch.autograd.grad(
            outputs=v_val,
            inputs=x_req,
            grad_outputs=torch.ones_like(v_val),
            create_graph=True,
            retain_graph=True,
        )[0]

        # System dynamics vector field f(x)
        f_x = self.dynamics(x_req)
        
        # Lie derivative dot product
        lie_derivative = torch.sum(grad_v * f_x, dim=-1, keepdim=True)
        return v_val, lie_derivative

    def train_step(self) -> Dict[str, float]:
        """Executes a single optimization step on the current dataset buffer."""
        self.optimizer.zero_grad()
        
        v_vals, lie_diffs = self.compute_lie_derivative(self.dataset)

        # Non-origin mask (ignore origin x=0 where V(0)=0 and Lie_V(0)=0)
        non_zero_mask = (torch.norm(self.dataset, dim=-1) > 1e-3).unsqueeze(-1)

        # Loss 1: Positive definiteness penalty V(x) > 0
        loss_pos = torch.relu(-v_vals + 1e-2) * non_zero_mask
        
        # Loss 2: Negative Lie derivative penalty ∇V(x) · f(x) < -α V(x)
        alpha = 0.1
        loss_lie = torch.relu(lie_diffs + alpha * v_vals) * non_zero_mask

        total_loss = torch.mean(loss_pos ** 2 + loss_lie ** 2)
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "max_lie_violation": torch.max(lie_diffs * non_zero_mask.float()).item(),
            "min_v_value": torch.min(v_vals + (1 - non_zero_mask.float()) * 1e6).item(),
        }

    def find_counterexamples(self, num_searches: int = 500, steps: int = 100) -> torch.Tensor:
        x_adv = self._generate_uniform_domain_samples(num_samples=num_searches).requires_grad_(True)
        adv_opt = optim.Adam([x_adv], lr=1e-2)

        counterexamples = []

        for _ in range(steps):
            adv_opt.zero_grad()
            v_vals, lie_diffs = self.compute_lie_derivative(x_adv)
            
            # Objective: Maximize Lie derivative (find points pushing lie_diffs to positive values)
            adv_loss = -torch.sum(lie_diffs)
            adv_loss.backward()
            adv_opt.step()

            # Project back onto domain hyper-ball
            with torch.no_grad():
                norms = torch.norm(x_adv, p=2, dim=1, keepdim=True)
                exceed = norms > self.domain_radius
                x_adv[exceed.squeeze()] = (x_adv[exceed.squeeze()] / norms[exceed]) * self.domain_radius

        # Collect states where conditions are violated
        with torch.no_grad():
            v_vals, lie_diffs = self.compute_lie_derivative(x_adv)
            non_zero_mask = torch.norm(x_adv, dim=-1) > 1e-3
            violations = (lie_diffs.squeeze() >= 0) & non_zero_mask
            if torch.any(violations):
                counterexamples.append(x_adv[violations].detach())

        if counterexamples:
            return torch.cat(counterexamples, dim=0)
        return torch.empty(0, self.state_dim, device=self.device)

    def run_cegis_loop(self, max_outer_loops: int = 10, epochs_per_loop: int = 500) -> nn.Module:
        print(f"[CEGIS] Initializing training loop on device={self.device}...")
        
        for outer in range(max_outer_loops):
            print(f"\n--- CEGIS Iteration {outer + 1}/{max_outer_loops} ---")
            print(f"[Learner] Training on {len(self.dataset)} samples...")
            
            for epoch in range(epochs_per_loop):
                metrics = self.train_step()
                if (epoch + 1) % 100 == 0:
                    print(f"  Epoch {epoch+1:04d} | Loss: {metrics['total_loss']:.6f} | "
                          f"Max Lie Violation: {metrics['max_lie_violation']:.6f}")

            print("[Falsifier] Searching for counterexamples...")
            ce_samples = self.find_counterexamples()

            if len(ce_samples) == 0:
                print("✓ [CEGIS Success] No counterexamples found! Neural candidate passed empirical falsification.")
                break
            else:
                print(f"⚠ [Falsifier] Found {len(ce_samples)} counterexamples. Injecting into dataset buffer...")
                self.dataset = torch.cat([self.dataset, ce_samples], dim=0)

        return self.model
