"""Low-rank observable/ambiguous decomposition on sampled CQR queries."""

from __future__ import annotations

import torch
import torch.nn as nn


class CQRObservabilityProjector(nn.Module):
    def __init__(self, mu_relative: float = 1e-3, jitter: float = 1e-6):
        super().__init__()
        self.mu_relative = float(mu_relative)
        self.jitter = float(jitter)

    @staticmethod
    def _batched(a_query: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if a_query.ndim == 2:
            return a_query.unsqueeze(0), True
        if a_query.ndim != 3:
            raise ValueError(f"a_query must be [R,N] or [B,R,N], got {a_query.shape}")
        return a_query, False

    def _solve(self, rhs: torch.Tensor, a_query: torch.Tensor) -> torch.Tensor:
        a_query, squeezed = self._batched(a_query)
        if rhs.ndim == 1:
            rhs = rhs.unsqueeze(0)
        device_type = a_query.device.type
        with torch.amp.autocast(device_type, enabled=False):
            a = a_query.float()
            b = rhs.float()
            gram = a @ a.transpose(-1, -2)
            rank = gram.shape[-1]
            mu = self.regularization_from_gram(gram)
            regularization = (mu + self.jitter).view(-1, 1, 1)
            eye = torch.eye(rank, device=a.device, dtype=a.dtype).expand_as(gram)
            solution = torch.linalg.solve(gram + regularization * eye, b.unsqueeze(-1)).squeeze(-1)
        return solution.squeeze(0) if squeezed else solution

    def regularization_from_gram(self, gram: torch.Tensor) -> torch.Tensor:
        """Return the relative Tikhonov scale used by the rank-space solve."""

        rank = gram.shape[-1]
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        return self.mu_relative * trace / max(rank, 1)

    @torch.no_grad()
    def effective_degrees_of_freedom(self, a_query: torch.Tensor) -> torch.Tensor:
        """Compute ``sum sigma^2 / (sigma^2 + mu)`` for diagnostics."""

        a, squeezed = self._batched(a_query)
        with torch.amp.autocast(a.device.type, enabled=False):
            gram = a.float() @ a.float().transpose(-1, -2)
            eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
            mu = self.regularization_from_gram(gram) + self.jitter
            result = (eigenvalues / (eigenvalues + mu.unsqueeze(-1))).sum(dim=-1)
        return result.squeeze(0) if squeezed else result

    def project_observable(self, z: torch.Tensor, a_query: torch.Tensor) -> torch.Tensor:
        a, squeezed = self._batched(a_query)
        z_batch = z.unsqueeze(0) if z.ndim == 1 else z
        with torch.amp.autocast(a.device.type, enabled=False):
            a32 = a.float()
            z32 = z_batch.float()
            modes = (a32 @ z32.unsqueeze(-1)).squeeze(-1)
            solved = self._solve(modes, a32)
            if solved.ndim == 1:
                solved = solved.unsqueeze(0)
            projected = (a32.transpose(-1, -2) @ solved.unsqueeze(-1)).squeeze(-1)
        projected = projected.to(z.dtype)
        return projected.squeeze(0) if squeezed else projected

    def project_ambiguous(self, z: torch.Tensor, a_query: torch.Tensor) -> torch.Tensor:
        return z - self.project_observable(z, a_query)

    def adjoint_evidence(
        self, residual_modes: torch.Tensor, a_query: torch.Tensor
    ) -> torch.Tensor:
        a, squeezed = self._batched(a_query)
        residual = residual_modes.unsqueeze(0) if residual_modes.ndim == 1 else residual_modes
        with torch.amp.autocast(a.device.type, enabled=False):
            solved = self._solve(residual.float(), a.float())
            if solved.ndim == 1:
                solved = solved.unsqueeze(0)
            evidence = (
                a.float().transpose(-1, -2) @ solved.unsqueeze(-1)
            ).squeeze(-1)
        evidence = evidence.to(residual_modes.dtype)
        return evidence.squeeze(0) if squeezed else evidence
