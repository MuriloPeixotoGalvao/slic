# slic_app/supcon.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

TORCH_OK = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    torch = None
    nn = None
    F = None


# -----------------------------
# Modelo: ContrastiveHead
# -----------------------------
if TORCH_OK:
    class ContrastiveHead(nn.Module):
        """
        Cabeça MLP simples para gerar embeddings normalizados (para SupCon).
        """
        def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32, p_drop: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p_drop),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x: "Tensor") -> "Tensor":
            z = self.net(x)
            z = F.normalize(z, dim=1, eps=1e-8)
            return z
else:
    class ContrastiveHead:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch não disponível. Instale torch para usar SupCon.")


# -----------------------------
# Loss: Supervised Contrastive
# -----------------------------
def _supcon_loss(z: "Tensor", y: "Tensor", temperature: float = 0.1) -> "Tensor":
    """
    z: (B, D) embeddings L2-normalizados
    y: (B,) labels
    Implementação supervisionada (Khosla et al., 2020) simplificada.
    """
    # Similaridade (cosine, pois z normalizado)
    sim = (z @ z.T) / float(temperature)  # (B,B)

    # Estabilidade numérica
    sim = sim - sim.max(dim=1, keepdim=True).values

    B = sim.shape[0]
    # máscara de positivos: mesmo rótulo, exclui self
    y = y.view(-1, 1)
    pos = (y == y.T).float()
    eye = torch.eye(B, device=sim.device, dtype=sim.dtype)
    pos = pos * (1.0 - eye)

    # denominador: todos exceto self
    exp_sim = torch.exp(sim) * (1.0 - eye)
    denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)

    # log prob
    log_prob = sim - torch.log(denom)

    # média apenas em âncoras que possuem pelo menos 1 positivo
    pos_count = pos.sum(dim=1).clamp_min(1.0)
    loss_i = -(pos * log_prob).sum(dim=1) / pos_count
    loss = loss_i.mean()
    return loss


# -----------------------------
# Treino SupCon em features (NumPy -> Torch)
# -----------------------------
def train_supcon_on_feats(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 200,
    batch: int = 128,
    lr: float = 1e-3,
    temperature: float = 0.1,
    device: Optional[str] = None,
    log_every: int = 25,
    progress_cb: Optional[Callable[[int, int, Optional[float]], None]] = None,
    hidden: int = 64,
    out_dim: int = 32,
    p_drop: float = 0.1,
) -> Tuple["ContrastiveHead", List[Tuple[int, float]]]:
    """
    Retorna (head_treinado, loss_log)
    """
    if not TORCH_OK:
        raise RuntimeError("PyTorch não disponível. Instale torch para usar SupCon.")

    assert X.ndim == 2, "X deve ser 2D (N, D)"
    assert y.ndim == 1, "y deve ser 1D (N,)"

    dev = device
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.int64))

    head = ContrastiveHead(in_dim=X.shape[1], hidden=hidden, out_dim=out_dim, p_drop=p_drop).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=float(lr))

    N = X.shape[0]
    idx = np.arange(N)

    loss_log: List[Tuple[int, float]] = []

    head.train()
    for ep in range(1, int(epochs) + 1):
        np.random.shuffle(idx)

        ep_losses = []
        for s in range(0, N, int(batch)):
            bidx = idx[s : s + int(batch)]
            xb = X_t[bidx].to(dev, non_blocking=True)
            yb = y_t[bidx].to(dev, non_blocking=True)

            z = head(xb)
            loss = _supcon_loss(z, yb, temperature=float(temperature))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            ep_losses.append(float(loss.detach().cpu().item()))

        ep_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")

        if (ep == 1) or (ep % int(log_every) == 0) or (ep == int(epochs)):
            loss_log.append((ep, ep_loss))
            if progress_cb is not None:
                progress_cb(ep, int(epochs), ep_loss)

    head.eval()
    if progress_cb is not None:
        progress_cb(int(epochs), int(epochs), loss_log[-1][1] if loss_log else None)

    return head, loss_log


# -----------------------------
# Embedding de features (batch)
# -----------------------------
@torch.no_grad() if TORCH_OK else (lambda f: f)  # type: ignore
def embed_all_feats(
    head: "ContrastiveHead",
    X: np.ndarray,
    batch: int = 2048,
    device: Optional[str] = None,
) -> np.ndarray:
    if not TORCH_OK:
        raise RuntimeError("PyTorch não disponível. Instale torch para usar embed_all_feats.")

    dev = device
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    head = head.to(dev)
    head.eval()

    X_t = torch.from_numpy(X.astype(np.float32))
    out_list = []

    N = X.shape[0]
    for s in range(0, N, int(batch)):
        xb = X_t[s : s + int(batch)].to(dev, non_blocking=True)
        zb = head(xb).detach().cpu().numpy()
        out_list.append(zb)

    return np.vstack(out_list).astype(np.float32)


__all__ = ["TORCH_OK", "ContrastiveHead", "train_supcon_on_feats", "embed_all_feats"]
