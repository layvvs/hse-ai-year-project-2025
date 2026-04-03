import numpy as np


def recall_at_k(test_true, user_item_matrix, user2id, id2item, k=10, recommend_fn=None):
    hits = 0
    total = 0

    for uid, true_items in test_true.items():
        recs = recommend_fn(uid, user_item_matrix, user2id, id2item, k)
        recs_set = set(recs)
        hits += len(recs_set & true_items)
        total += len(true_items)

    return hits / total if total > 0 else 0


def precision_at_k(test_true, user_item_matrix, user2id, id2item, k=10, recommend_fn=None):
    hits = 0
    users = 0

    for uid, true_items in test_true.items():
        recs = recommend_fn(uid, user_item_matrix, user2id, id2item, k)
        if len(recs) == 0:
            continue
        hits += len(set(recs) & true_items)
        users += 1

    return hits / (users * k) if users > 0 else 0


def dcg_at_k(recs, true_items, k):
    dcg = 0.0

    for i, item in enumerate(recs[:k]):
        if item in true_items:
            dcg += 1 / np.log2(i + 2)

    return dcg


def idcg_at_k(true_items, k):
    ideal_hits = min(len(true_items), k)

    idcg = 0.0
    for i in range(ideal_hits):
        idcg += 1 / np.log2(i + 2)

    return idcg


def ndcg_at_k(test_true, user_item_matrix, user2id, id2item, k=10, recommend_fn=None):
    ndcgs = []

    for uid, true_items in test_true.items():
        recs = recommend_fn(uid, user_item_matrix, user2id, id2item, k)
        dcg = dcg_at_k(recs, true_items, k)
        idcg = idcg_at_k(true_items, k)
        if idcg == 0:
            continue
        ndcgs.append(dcg / idcg)

    return np.mean(ndcgs) if ndcgs else 0
