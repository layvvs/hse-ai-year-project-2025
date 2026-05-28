import numpy as np


def _user_row_ravel(user_item_matrix, u):
    row = user_item_matrix[u]
    if hasattr(row, "toarray"):
        return row.toarray().ravel()
    return np.asarray(row).ravel()


class ELSA:
    def __init__(self, lam=250.0, top_k=5000):
        self.lam = lam
        self.top_k = top_k
        self._B = None
        self._ease_item_cols = None
        self.user_item_matrix = None

    def predict(self, user_id, user2id, id2item, k=10):
        u = user2id[user_id]
        row = _user_row_ravel(self.user_item_matrix, u)
        ease_cols = self._ease_item_cols
        x = row[ease_cols].astype(np.float64)
        x_bin = (x > 0).astype(np.float64)
        scores = x_bin @ self._B
        seen = np.flatnonzero(x_bin)
        scores[seen] = -1.0
        n_take = min(k, scores.shape[0])
        top_local = np.argsort(scores)[-n_take:][::-1]
        return [id2item[int(ease_cols[j])] for j in top_local]
