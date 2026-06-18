import numpy as np
 
 
def precision_curve(recommended, relevant):
    relevant = set(relevant)
    hits, precisions = 0, []
    for i, item in enumerate(recommended, start=1):
        if item in relevant:
            hits += 1
        precisions.append(hits / i)
    return precisions
 
 
def recall_curve(recommended, relevant):
    relevant = set(relevant)
    total = len(relevant)
    hits, recalls = 0, []
    for item in recommended:
        if item in relevant:
            hits += 1
        recalls.append(hits / total if total > 0 else 0.0)
    return recalls
 
 
def f1_curve(recommended, relevant):
    relevant = set(relevant)
    total = len(relevant)
    hits, f1s = 0, []
    for i, item in enumerate(recommended, start=1):
        if item in relevant:
            hits += 1
        p = hits / i
        r = hits / total if total > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return f1s
 
 
def mrr_curve(recommended, relevant):
    relevant = set(relevant)
    mrr_values, found_rank = [], None
    for k in range(1, len(recommended) + 1):
        for rank, item in enumerate(recommended[:k], start=1):
            if item in relevant:
                found_rank = rank
                break
        mrr_values.append(1.0 / found_rank if found_rank else 0.0)
    return mrr_values
 
 
def ndcg_at_k(recommended, relevant, k):
    relevant = set(relevant)
    hits = [1 if r in relevant else 0 for r in recommended[:k]]
    dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0
 
 
def precision_at_k(recommended, relevant, k):
    relevant = set(relevant)
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / k if k > 0 else 0.0
 
 
def recall_at_k(recommended, relevant, k):
    relevant = set(relevant)
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / len(relevant) if len(relevant) > 0 else 0.0
 
 
def hit_rate_at_k(recommended, relevant, k):
    relevant = set(relevant)
    return 1 if any(r in relevant for r in recommended[:k]) else 0