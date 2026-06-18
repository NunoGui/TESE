


def precision_at_k(test_items, recommended_items, k):

    if not recommended_items or k == 0:
        return 0.0
    
    recommended_at_k = recommended_items[:k]
    relevant_items = set(test_items)
    
  
    num_relevant = sum([1 for item in recommended_at_k if item in relevant_items])
    
    precision = num_relevant / k
    return precision



def precision_curve(recommended, relevant):

    relevant = set(relevant)  
    precisions = []
    hits = 0

    for i, item in enumerate(recommended, start=1):
        if item in relevant:
            hits += 1
        precisions.append(hits / i)

    return precisions



import matplotlib.pyplot as plt

def hit_ratio_curve_single(recommended_items, relevant_items, plot=False):

    ks = range(1, len(recommended_items) + 1)
    hr_values = []

    hits = 0
    total_relevant = len(relevant_items)

    for k in ks:
        if recommended_items[k - 1] in relevant_items:
            hits += 1
        hr = hits / total_relevant  
        hr_values.append(hr)

    if plot:
        plt.figure(figsize=(7, 4))
        plt.plot(ks, hr_values, marker='o')
        plt.title("Hit Ratio Curve (HR@k)")
        plt.xlabel("k (Top-k recommendations)")
        plt.ylabel("Hit Ratio")
        plt.grid(True)
        plt.show()
    print(recommended_items)
    print(relevant_items)
    print(list(ks), hr_values)
    return hr_values

def mean_reciprocal_rank_at_k(recommended_items, relevant_items):

    mrr_values = []
    found_rank = None
    

    for k in range(1, len(recommended_items) + 1):

        for rank, item in enumerate(recommended_items[:k], start=1):
            if item in relevant_items:
                found_rank = rank
                break
        
        if found_rank is not None:
            mrr_values.append(1.0 / found_rank)
        else:
            mrr_values.append(0.0)
    
    return mrr_values



def recall_at_k(test_items, recommended_items, k):

    if not recommended_items or not test_items:
        return 0.0

    recommended_at_k = recommended_items[:k]
    relevant_items = set(test_items)

    num_relevant = sum(1 for item in recommended_at_k if item in relevant_items)
    total_relevant = len(relevant_items)

    return num_relevant / total_relevant if total_relevant > 0 else 0.0


def recall_curve(recommended, relevant):

    relevant = set(relevant)
    total_relevant = len(relevant)
    recalls = []
    hits = 0

    if total_relevant == 0:
        return [0.0] * len(recommended)

    for i, item in enumerate(recommended, start=1):
        if item in relevant:
            hits += 1
        recalls.append(hits / total_relevant)

    return recalls

## F1-Score
def f1_score_at_k(test_items, recommended_items, k):

    precision = precision_at_k(test_items, recommended_items, k)
    recall = recall_at_k(test_items, recommended_items, k)

    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def f1_curve(recommended, relevant):

    relevant = set(relevant)
    total_relevant = len(relevant)
    precisions = []
    recalls = []
    hits = 0
    f1s = []

    for i, item in enumerate(recommended, start=1):
        if item in relevant:
            hits += 1
        precision = hits / i
        recall = hits / total_relevant if total_relevant > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

        if precision + recall > 0:
            f1s.append(2 * precision * recall / (precision + recall))
        else:
            f1s.append(0.0)

    return f1s
