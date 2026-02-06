from __future__ import annotations
import numpy as np

def knn_predict_numpy_balanced(
    Z_ref: np.ndarray,
    y_ref: np.ndarray,
    Z_query: np.ndarray,
    *,
    k: int,
    balance: bool,
    leave_one_out_idx: np.ndarray | None = None,
) -> np.ndarray:
    Z_ref = Z_ref.astype(np.float32, copy=False)
    Z_query = Z_query.astype(np.float32, copy=False)
    y_ref = y_ref.astype(np.int32, copy=False)

    n_ref = Z_ref.shape[0]
    k = int(max(1, min(k, n_ref)))

    # pesos por classe (balanceado)
    if balance:
        cls, cnt = np.unique(y_ref, return_counts=True)
        w = {int(c): 1.0 / float(n) for c, n in zip(cls, cnt)}
    else:
        w = None

    y_out = np.zeros((Z_query.shape[0],), np.int32)

    # distâncias por bloco para não explodir RAM
    block = 512
    for i0 in range(0, Z_query.shape[0], block):
        i1 = min(Z_query.shape[0], i0 + block)
        Q = Z_query[i0:i1]

        # (q - r)^2 = q^2 + r^2 - 2 q.r
        q2 = (Q * Q).sum(axis=1, keepdims=True)
        r2 = (Z_ref * Z_ref).sum(axis=1, keepdims=True).T
        dots = Q @ Z_ref.T
        D = q2 + r2 - 2.0 * dots

        if leave_one_out_idx is not None:
            # para LOO, invalida o "mesmo" índice
            for ii, ridx in enumerate(leave_one_out_idx[i0:i1]):
                if 0 <= ridx < n_ref:
                    D[ii, ridx] = np.inf

        nn = np.argpartition(D, kth=k-1, axis=1)[:, :k]
        for ii in range(nn.shape[0]):
            neigh = y_ref[nn[ii]]
            if w is None:
                # voto simples
                vals, cnts = np.unique(neigh, return_counts=True)
                y_out[i0 + ii] = int(vals[np.argmax(cnts)])
            else:
                # voto ponderado
                score = {}
                for c in neigh:
                    score[int(c)] = score.get(int(c), 0.0) + w[int(c)]
                y_out[i0 + ii] = int(max(score.items(), key=lambda x: x[1])[0])

    return y_out
