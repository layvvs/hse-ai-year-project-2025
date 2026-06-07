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


def average_precision_at_k(recs, true_items, k):
    if not true_items:
        return 0.0
    hits = 0
    prec_sum = 0.0
    for i, item in enumerate(recs[:k]):
        if item in true_items:
            hits += 1
            prec_sum += hits / (i + 1)
    denom = min(len(true_items), k)
    return prec_sum / denom if denom > 0 else 0.0


def map_at_k(test_true, user_item_matrix, user2id, id2item, k=10, recommend_fn=None):
    aps = []
    for uid, true_items in test_true.items():
        if not true_items:
            continue
        recs = recommend_fn(uid, user_item_matrix, user2id, id2item, k)
        aps.append(average_precision_at_k(recs, true_items, k))
    return np.mean(aps) if aps else 0.0


def hit_rate_at_k(test_true, user_item_matrix, user2id, id2item, k=10, recommend_fn=None):
    hits = 0
    users = 0
    for uid, true_items in test_true.items():
        if not true_items:
            continue
        recs = recommend_fn(uid, user_item_matrix, user2id, id2item, k)
        users += 1
        if set(recs) & true_items:
            hits += 1
    return hits / users if users > 0 else 0.0
