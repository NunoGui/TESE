

import pandas as pd
import numpy as np

import rs_cf_algorithms as cf
import rs_demographic_algorithms as demo
import similarities
import similarities
import evaluation



from sklearn.metrics.pairwise import cosine_similarity






def hybrid_recommendation(
    train_df,
    test_df,
    demo_df,
    n_neighbors=1,
    cf_weight=0.5,
    demo_weight=0.5,
    top_k=20
):
    import evaluation
    import numpy as np
    import pandas as pd

   
    cf_rankings = cf.CF_likes_rankings(train_df, test_df, n_neighbors)
    demo_rankings = demo.DEMOGRAPHIC_knn_rankings(train_df, test_df, demo_df, n_neighbors)

    precision_list, recall_list, f1_list, mrr_list = [], [], [], []

    for user in test_df['user'].unique():

        relevant_items = test_df.loc[
            (test_df['user'] == user) & (test_df['rating'] == 1),
            'item'
        ].tolist()

        if len(relevant_items) == 0:
            continue

        cf_list = cf_rankings.get(user, [])
        demo_list = demo_rankings.get(user, [])

        all_items = list(set(cf_list + demo_list))

        scores = {}

        for item in all_items:
            cf_rank = cf_list.index(item)+1 if item in cf_list else len(cf_list)+1
            demo_rank = demo_list.index(item)+1 if item in demo_list else len(demo_list)+1

            cf_score = 1.0 / cf_rank
            demo_score = 1.0 / demo_rank

            scores[item] = cf_weight * cf_score + demo_weight * demo_score

        ranked_items = sorted(scores, key=scores.get, reverse=True)

        
        precision_list.append(evaluation.precision_curve(ranked_items, relevant_items))
        recall_list.append(evaluation.recall_curve(ranked_items, relevant_items))
        f1_list.append(evaluation.f1_curve(ranked_items, relevant_items))
        mrr_list.append(evaluation.mean_reciprocal_rank_at_k(ranked_items, relevant_items))

    mean_precision = np.mean(precision_list, axis=0)
    mean_recall = np.mean(recall_list, axis=0)
    mean_f1 = np.mean(f1_list, axis=0)
    mean_mrr = np.mean(mrr_list, axis=0)

    return (
        pd.DataFrame([mean_precision[:top_k]], columns=[f'top{i}' for i in range(1, top_k+1)]),
        pd.DataFrame([mean_mrr[:top_k]], columns=[f'top{i}' for i in range(1, top_k+1)]),
        pd.DataFrame([mean_recall[:top_k]], columns=[f'top{i}' for i in range(1, top_k+1)]),
        pd.DataFrame([mean_f1[:top_k]], columns=[f'top{i}' for i in range(1, top_k+1)])
    )




def hybrid_cf_emotions_demo(
    train_df,
    test_df,
    demo_df,
    n_neighbors=1,
    cf_weight=0.7,
    demo_weight=0.3,
    top_k=20
):
    import rs_cf_algorithms as cf
    import rs_demographic_algorithms as demo

    
    cf_rankings = cf.CF_emotions_rankings(train_df, test_df, n_neighbors)
    demo_rankings = demo.DEMOGRAPHIC_knn_rankings(train_df, test_df, demo_df, n_neighbors)

    users = test_df['user'].unique()

    precision_list, recall_list, f1_list, mrr_list = [], [], [], []

    for user in users:

        if user not in cf_rankings or user not in demo_rankings:
            continue

        relevant_items = test_df.loc[
            (test_df['user'] == user) & (test_df['rating'] == 1),
            'item'
        ].tolist()

        if len(relevant_items) == 0:
            continue

        cf_items = cf_rankings[user]
        demo_items = demo_rankings[user]

        all_items = list(set(cf_items + demo_items))

        score_dict = {}

        for item in all_items:
            cf_rank = cf_items.index(item) + 1 if item in cf_items else len(cf_items) + 1
            demo_rank = demo_items.index(item) + 1 if item in demo_items else len(demo_items) + 1

            cf_score = 1.0 / cf_rank
            demo_score = 1.0 / demo_rank

            score_dict[item] = cf_weight * cf_score + demo_weight * demo_score

        ranked_items = sorted(score_dict, key=score_dict.get, reverse=True)

        precision_list.append(
            evaluation.precision_curve(ranked_items, relevant_items)[:top_k]
        )
        recall_list.append(
            evaluation.recall_curve(ranked_items, relevant_items)[:top_k]
        )
        f1_list.append(
            evaluation.f1_curve(ranked_items, relevant_items)[:top_k]
        )
        mrr_list.append(
            evaluation.mean_reciprocal_rank_at_k(ranked_items, relevant_items)[:top_k]
        )

    mean_precision = pd.DataFrame(
        [np.mean(precision_list, axis=0)],
        columns=[f'top{i}' for i in range(1, top_k + 1)]
    )

    mean_recall = pd.DataFrame(
        [np.mean(recall_list, axis=0)],
        columns=[f'top{i}' for i in range(1, top_k + 1)]
    )

    mean_f1 = pd.DataFrame(
        [np.mean(f1_list, axis=0)],
        columns=[f'top{i}' for i in range(1, top_k + 1)]
    )

    mean_mrr = pd.DataFrame(
        [np.mean(mrr_list, axis=0)],
        columns=[f'top{i}' for i in range(1, top_k + 1)]
    )

    return mean_precision, mean_mrr, mean_recall, mean_f1




def hybrid_cf_vad_demo(
    train_df,
    test_df,
    demo_df,
    n_neighbors=1,
    cf_weight=0.7,
    demo_weight=0.3,
    top_k=20
):
    import rs_cf_algorithms as cf
    import rs_demographic_algorithms as demo

    cf_rankings   = cf.CF_vad_rankings(train_df, test_df, n_neighbors)
    demo_rankings = demo.DEMOGRAPHIC_knn_rankings(train_df, test_df, demo_df, n_neighbors)

    precision_list, recall_list, f1_list, mrr_list = [], [], [], []

    for user in test_df['user'].unique():

        if user not in cf_rankings or user not in demo_rankings:
            continue

        relevant_items = test_df.loc[
            (test_df['user'] == user) & (test_df['rating'] == 1),
            'item'
        ].tolist()

        if len(relevant_items) == 0:
            continue

        cf_items   = cf_rankings[user]
        demo_items = demo_rankings[user]

        all_items = list(set(cf_items + demo_items))

        score_dict = {}
        for item in all_items:
            cf_rank   = cf_items.index(item)   + 1 if item in cf_items   else len(cf_items)   + 1
            demo_rank = demo_items.index(item) + 1 if item in demo_items else len(demo_items) + 1

            score_dict[item] = cf_weight * (1.0 / cf_rank) + demo_weight * (1.0 / demo_rank)

        ranked_items = sorted(score_dict, key=score_dict.get, reverse=True)

        precision_list.append(evaluation.precision_curve(ranked_items, relevant_items)[:top_k])
        recall_list.append(evaluation.recall_curve(ranked_items, relevant_items)[:top_k])
        f1_list.append(evaluation.f1_curve(ranked_items, relevant_items)[:top_k])
        mrr_list.append(evaluation.mean_reciprocal_rank_at_k(ranked_items, relevant_items)[:top_k])

    cols = [f'top{i}' for i in range(1, top_k + 1)]

    return (
        pd.DataFrame([np.mean(precision_list, axis=0)], columns=cols),
        pd.DataFrame([np.mean(mrr_list,       axis=0)], columns=cols),
        pd.DataFrame([np.mean(recall_list,    axis=0)], columns=cols),
        pd.DataFrame([np.mean(f1_list,        axis=0)], columns=cols),
    )


def hybrid_cf_all_demo(
    train_df,
    test_df,
    demo_df,
    n_neighbors=1,
    cf_weight=0.7,
    demo_weight=0.3,
    top_k=20
):
    import rs_cf_algorithms as cf
    import rs_demographic_algorithms as demo

    cf_rankings   = cf.CF_all_rankings(train_df, test_df, n_neighbors)
    demo_rankings = demo.DEMOGRAPHIC_knn_rankings(train_df, test_df, demo_df, n_neighbors)

    precision_list, recall_list, f1_list, mrr_list = [], [], [], []

    for user in test_df['user'].unique():

        if user not in cf_rankings or user not in demo_rankings:
            continue

        relevant_items = test_df.loc[
            (test_df['user'] == user) & (test_df['rating'] == 1),
            'item'
        ].tolist()

        if len(relevant_items) == 0:
            continue

        cf_items   = cf_rankings[user]
        demo_items = demo_rankings[user]

        all_items = list(set(cf_items + demo_items))

        score_dict = {}
        for item in all_items:
            cf_rank   = cf_items.index(item)   + 1 if item in cf_items   else len(cf_items)   + 1
            demo_rank = demo_items.index(item) + 1 if item in demo_items else len(demo_items) + 1

            score_dict[item] = cf_weight * (1.0 / cf_rank) + demo_weight * (1.0 / demo_rank)

        ranked_items = sorted(score_dict, key=score_dict.get, reverse=True)

        precision_list.append(evaluation.precision_curve(ranked_items, relevant_items)[:top_k])
        recall_list.append(evaluation.recall_curve(ranked_items, relevant_items)[:top_k])
        f1_list.append(evaluation.f1_curve(ranked_items, relevant_items)[:top_k])
        mrr_list.append(evaluation.mean_reciprocal_rank_at_k(ranked_items, relevant_items)[:top_k])

    cols = [f'top{i}' for i in range(1, top_k + 1)]

    return (
        pd.DataFrame([np.mean(precision_list, axis=0)], columns=cols),
        pd.DataFrame([np.mean(mrr_list,       axis=0)], columns=cols),
        pd.DataFrame([np.mean(recall_list,    axis=0)], columns=cols),
        pd.DataFrame([np.mean(f1_list,        axis=0)], columns=cols),
    )