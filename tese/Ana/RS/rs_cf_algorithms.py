import datasets
import split
import similarities
import evaluation
import numpy as np
import pandas as pd
import sys

from sklearn.decomposition import TruncatedSVD, NMF


def build_user_item_matrix(df):
    matrix = df.pivot_table(
        index='user',
        columns='item',
        values='rating',
        fill_value=0
    )
    return matrix, matrix.index, matrix.columns




def CF_likes(train_df, test_df, n):

    users_cos_sim = similarities.cosine_sim(train_df)

    precision_list, mrr_list, recall_list, f1s_list = [], [], [], []

    for user in test_df['user'].unique():

        relevant_items = test_df.loc[(test_df['user'] == user) & (test_df['rating'] == 1), 'item'].tolist()
        if len(relevant_items) == 0:
            continue
        if user not in users_cos_sim.index:
            continue

        k_most_sim = similarities.get_k_most_sim_users(users_cos_sim, user, n)
        scores = similarities.get_scores(test_df[test_df['user'] == user]['item'].tolist(), k_most_sim, train_df)

        precision_list.append(evaluation.precision_curve(scores.index.to_list(), relevant_items)[:20])
        mrr_list.append(evaluation.mean_reciprocal_rank_at_k(scores.index.to_list(), relevant_items)[:20])
        recall_list.append(evaluation.recall_curve(scores.index.to_list(), relevant_items)[:20])
        f1s_list.append(evaluation.f1_curve(scores.index.to_list(), relevant_items)[:20])

    cols = ['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20']

    precision_df = pd.DataFrame(precision_list, columns=cols)
    print('Precisio CF likes ', n, 'most similar user: ')
    print(pd.DataFrame([precision_df.mean()]))
    mean_precision_df = pd.DataFrame([precision_df.mean()])

    mrr_df = pd.DataFrame(mrr_list, columns=cols)
    print('MRR CF likes ', n, 'most similar user: ')
    print(pd.DataFrame([mrr_df.mean()]))
    mean_mrr_df = pd.DataFrame([mrr_df.mean()])

    recall_df = pd.DataFrame(recall_list, columns=cols)
    print('Recall CF likes ', n, 'most similar user: ')
    print(pd.DataFrame([recall_df.mean()]))
    mean_recall_df = pd.DataFrame([recall_df.mean()])

    f1s_df = pd.DataFrame(f1s_list, columns=cols)
    print('F1-Score CF likes ', n, 'most similar user: ')
    print(pd.DataFrame([f1s_df.mean()]))
    mean_f1s_df = pd.DataFrame([f1s_df.mean()])

    return (mean_precision_df, mean_mrr_df, mean_recall_df, mean_f1s_df)


################ RANKINGS LIKE ############################




def CF_likes_rankings(train_df, test_df, n_neighbors):
    import similarities

    users_cos_sim = similarities.cosine_sim(train_df)

    user_rankings = {}

    for user in test_df['user'].unique():

        if user not in users_cos_sim.index:
            continue


        k_most_sim = similarities.get_k_most_sim_users(
            users_cos_sim, user, n_neighbors
        )


        test_items = test_df[test_df['user'] == user]['item'].tolist()


        scores = similarities.get_scores(
            test_items, k_most_sim, train_df
        )


        user_rankings[user] = scores.sort_values(ascending=False).index.tolist()

    return user_rankings



def CF_emotions(train_df, test_df, n):
    users_cos_sim = similarities.cosine_sim_vector(train_df)

    precision_list, mrr_list, recall_list, f1s_list = [], [], [], []

    for user in test_df['user'].unique():

        relevant_items = test_df.loc[(test_df['user'] == user) & (test_df['rating'] == 1), 'item'].tolist()
        if len(relevant_items) == 0:
            continue
        if user not in users_cos_sim.index:
            continue

        k_most_sim = similarities.get_k_most_sim_users(users_cos_sim, user, n)
        scores = similarities.get_scores(test_df[test_df['user'] == user]['item'].tolist(), k_most_sim, train_df)

        precision_list.append(evaluation.precision_curve(scores.index.to_list(), relevant_items)[:20])
        mrr_list.append(evaluation.mean_reciprocal_rank_at_k(scores.index.to_list(), relevant_items)[:20])
        recall_list.append(evaluation.recall_curve(scores.index.to_list(), relevant_items)[:20])
        f1s_list.append(evaluation.f1_curve(scores.index.to_list(), relevant_items)[:20])

    cols = ['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20']

    precision_df = pd.DataFrame(precision_list, columns=cols)
    print('Precisio CF emotions ', n, 'most similar user: ')
    print(pd.DataFrame([precision_df.mean()]))
    mean_precision_df = pd.DataFrame([precision_df.mean()])

    mrr_df = pd.DataFrame(mrr_list, columns=cols)
    print('MRR CF emotions ', n, 'most similar user: ')
    print(pd.DataFrame([mrr_df.mean()]))
    mean_mrr_df = pd.DataFrame([mrr_df.mean()])

    recall_df = pd.DataFrame(recall_list, columns=cols)
    print('Recall CF emotions ', n, 'most similar user: ')
    print(pd.DataFrame([recall_df.mean()]))
    mean_recall_df = pd.DataFrame([recall_df.mean()])

    f1s_df = pd.DataFrame(f1s_list, columns=cols)
    print('F1-Score CF emotions ', n, 'most similar user: ')
    print(pd.DataFrame([f1s_df.mean()]))
    mean_f1s_df = pd.DataFrame([f1s_df.mean()])

    return (mean_precision_df, mean_mrr_df, mean_recall_df, mean_f1s_df)



############## Emotion RANKINGS #########################

def CF_emotions_rankings(train_df, test_df, n_neighbors):
    
    users_cos_sim = similarities.cosine_sim_vector(train_df[['user', 'item', 'rating', 'anger', 'fear', 'disgust', 'sadness','happiness', 'surprise', 'neutral']])
    user_rankings = {}

    for user in test_df['user'].unique():
        if user not in users_cos_sim.index:
            continue

        k_most_sim = similarities.get_k_most_sim_users(
            users_cos_sim, user, n_neighbors
        )

        test_items = test_df[test_df['user'] == user]['item'].tolist()

        scores = similarities.get_scores(
            test_items,
            k_most_sim,
            train_df
        )

        ranked_items = scores.sort_values(ascending=False).index.tolist()
        user_rankings[user] = ranked_items

    return user_rankings



def CF_vad(train_df, test_df, n):
    users_cos_sim = similarities.cosine_sim_vector(train_df)

    precision_list, mrr_list, recall_list, f1s_list = [], [], [], []

    for user in test_df['user'].unique():

        relevant_items = test_df.loc[(test_df['user'] == user) & (test_df['rating'] == 1), 'item'].tolist()
        if len(relevant_items) == 0:
            continue
        if user not in users_cos_sim.index:
            continue

        k_most_sim = similarities.get_k_most_sim_users(users_cos_sim, user, n)
        scores = similarities.get_scores(test_df[test_df['user'] == user]['item'].tolist(), k_most_sim, train_df)

        precision_list.append(evaluation.precision_curve(scores.index.to_list(), relevant_items)[:20])
        mrr_list.append(evaluation.mean_reciprocal_rank_at_k(scores.index.to_list(), relevant_items)[:20])
        recall_list.append(evaluation.recall_curve(scores.index.to_list(), relevant_items)[:20])
        f1s_list.append(evaluation.f1_curve(scores.index.to_list(), relevant_items)[:20])

    cols = ['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20']

    precision_df = pd.DataFrame(precision_list, columns=cols)
    print('Precisio CF VAD ', n, 'most similar user: ')
    print(pd.DataFrame([precision_df.mean()]))
    mean_precision_df = pd.DataFrame([precision_df.mean()])

    mrr_df = pd.DataFrame(mrr_list, columns=cols)
    print('MRR CF VAD ', n, 'most similar user: ')
    print(pd.DataFrame([mrr_df.mean()]))
    mean_mrr_df = pd.DataFrame([mrr_df.mean()])

    recall_df = pd.DataFrame(recall_list, columns=cols)
    print('Recall CF VAD ', n, 'most similar user: ')
    print(pd.DataFrame([recall_df.mean()]))
    mean_recall_df = pd.DataFrame([recall_df.mean()])

    f1s_df = pd.DataFrame(f1s_list, columns=cols)
    print('F1-Score CF VAD ', n, 'most similar user: ')
    print(pd.DataFrame([f1s_df.mean()]))
    mean_f1s_df = pd.DataFrame([f1s_df.mean()])

    return (mean_precision_df, mean_mrr_df, mean_recall_df, mean_f1s_df)



def CF_vad_rankings(train_df, test_df, n_neighbors):
    users_cos_sim = similarities.cosine_sim_vector(train_df[['user', 'item', 'rating', 'valence', 'arousal', 'dominance']])

    user_rankings = {}

    for user in test_df['user'].unique():
        if user not in users_cos_sim.index:
            continue

        k_most_sim = similarities.get_k_most_sim_users(users_cos_sim, user, n_neighbors)
        test_items = test_df[test_df['user'] == user]['item'].tolist()
        scores = similarities.get_scores(test_items, k_most_sim, train_df)
        user_rankings[user] = scores.sort_values(ascending=False).index.tolist()

    return user_rankings






def CF_all(train_df, test_df, n):
    users_cos_sim = similarities.cosine_sim_vector(train_df)

    precision_list, mrr_list, recall_list, f1s_list = [], [], [], []

    for user in test_df['user'].unique():

        relevant_items = test_df.loc[(test_df['user'] == user) & (test_df['rating'] == 1), 'item'].tolist()
        if len(relevant_items) == 0:
            continue
        if user not in users_cos_sim.index:
            continue

        k_most_sim = similarities.get_k_most_sim_users(users_cos_sim, user, n)
        scores = similarities.get_scores(test_df[test_df['user'] == user]['item'].tolist(), k_most_sim, train_df)

        precision_list.append(evaluation.precision_curve(scores.index.to_list(), relevant_items)[:20])
        mrr_list.append(evaluation.mean_reciprocal_rank_at_k(scores.index.to_list(), relevant_items)[:20])
        recall_list.append(evaluation.recall_curve(scores.index.to_list(), relevant_items)[:20])
        f1s_list.append(evaluation.f1_curve(scores.index.to_list(), relevant_items)[:20])

    cols = ['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20']

    precision_df = pd.DataFrame(precision_list, columns=cols)
    print('Precisio CF all ', n, 'most similar user: ')
    print(pd.DataFrame([precision_df.mean()]))
    mean_precision_df = pd.DataFrame([precision_df.mean()])

    mrr_df = pd.DataFrame(mrr_list, columns=cols)
    print('MRR CF all ', n, 'most similar user: ')
    print(pd.DataFrame([mrr_df.mean()]))
    mean_mrr_df = pd.DataFrame([mrr_df.mean()])

    recall_df = pd.DataFrame(recall_list, columns=cols)
    print('Recall CF all ', n, 'most similar user: ')
    print(pd.DataFrame([recall_df.mean()]))
    mean_recall_df = pd.DataFrame([recall_df.mean()])

    f1s_df = pd.DataFrame(f1s_list, columns=cols)
    print('F1-Score CF all ', n, 'most similar user: ')
    print(pd.DataFrame([f1s_df.mean()]))
    mean_f1s_df = pd.DataFrame([f1s_df.mean()])

    return (mean_precision_df, mean_mrr_df, mean_recall_df, mean_f1s_df)



def CF_all_rankings(train_df, test_df, n_neighbors):
   
    users_cos_sim = similarities.cosine_sim_vector(train_df[['user', 'item', 'rating', 'anger', 'fear', 'disgust', 'sadness','happiness', 'surprise', 'neutral','valence', 'arousal', 'dominance']])

    user_rankings = {}

    for user in test_df['user'].unique():
        if user not in users_cos_sim.index:
            continue

        k_most_sim = similarities.get_k_most_sim_users(users_cos_sim, user, n_neighbors)
        test_items = test_df[test_df['user'] == user]['item'].tolist()
        scores = similarities.get_scores(test_items, k_most_sim, train_df)
        user_rankings[user] = scores.sort_values(ascending=False).index.tolist()

    return user_rankings







def CF_SVD(train_df, test_df, n, n_factors=5):

    matrix, users, items = build_user_item_matrix(train_df)

    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_factors = svd.fit_transform(matrix)
    item_factors = svd.components_

    predictions = np.dot(user_factors, item_factors)
    pred_df = pd.DataFrame(predictions, index=users, columns=items)

    precision_list, mrr_list, recall_list, f1s_list = [], [], [], []

    for user in test_df['user'].unique():
        if user not in pred_df.index:
            continue


        test_items = test_df[test_df['user'] == user]['item'].tolist()

        if not test_items:
            continue


        relevant_items = test_df.loc[
            (test_df['user'] == user) & (test_df['rating'] == 1),
            'item'
        ].tolist()

        if not relevant_items:
            continue


        scores = {}
        for item in test_items:
            if item in pred_df.columns:
                scores[item] = pred_df.loc[user, item]
            else:
                scores[item] = 0.0  

        scores_series = pd.Series(scores).sort_values(ascending=False)
        ranked_items = scores_series.index.tolist()

        precision_list.append(evaluation.precision_curve(ranked_items, relevant_items)[:20])
        mrr_list.append(evaluation.mean_reciprocal_rank_at_k(ranked_items, relevant_items)[:20])
        recall_list.append(evaluation.recall_curve(ranked_items, relevant_items)[:20])
        f1s_list.append(evaluation.f1_curve(ranked_items, relevant_items)[:20])

    cols = ['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10',
            'top11','top12','top13','top14','top15','top16','top17','top18','top19','top20']

    precision_df = pd.DataFrame(precision_list, columns=cols)
    print('Precision CF SVD:')
    print(pd.DataFrame([precision_df.mean()]))
    mean_precision_df = pd.DataFrame([precision_df.mean()])

    mrr_df = pd.DataFrame(mrr_list, columns=cols)
    print('MRR CF SVD:')
    print(pd.DataFrame([mrr_df.mean()]))
    mean_mrr_df = pd.DataFrame([mrr_df.mean()])

    recall_df = pd.DataFrame(recall_list, columns=cols)
    print('Recall CF SVD:')
    print(pd.DataFrame([recall_df.mean()]))
    mean_recall_df = pd.DataFrame([recall_df.mean()])

    f1s_df = pd.DataFrame(f1s_list, columns=cols)
    print('F1-Score CF SVD:')
    print(pd.DataFrame([f1s_df.mean()]))
    mean_f1s_df = pd.DataFrame([f1s_df.mean()])

    return mean_precision_df, mean_mrr_df, mean_recall_df, mean_f1s_df








def CF_NMF(train_df, test_df, n, n_factors=5):

    matrix, users, items = build_user_item_matrix(train_df)

    nmf = NMF(n_components=n_factors, init='random', random_state=42, max_iter=300)
    user_factors = nmf.fit_transform(matrix)
    item_factors = nmf.components_

    predictions = np.dot(user_factors, item_factors)
    pred_df = pd.DataFrame(predictions, index=users, columns=items)

    precision_list, mrr_list, recall_list, f1s_list = [], [], [], []

    for user in test_df['user'].unique():
        if user not in pred_df.index:
            continue


        test_items = test_df[test_df['user'] == user]['item'].tolist()

        if not test_items:
            continue


        relevant_items = test_df.loc[
            (test_df['user'] == user) & (test_df['rating'] == 1),
            'item'
        ].tolist()

        if not relevant_items:
            continue


        scores = {}
        for item in test_items:
            if item in pred_df.columns:
                scores[item] = pred_df.loc[user, item]
            else:
                scores[item] = 0.0  

        scores_series = pd.Series(scores).sort_values(ascending=False)
        ranked_items = scores_series.index.tolist()

        precision_list.append(evaluation.precision_curve(ranked_items, relevant_items)[:20])
        mrr_list.append(evaluation.mean_reciprocal_rank_at_k(ranked_items, relevant_items)[:20])
        recall_list.append(evaluation.recall_curve(ranked_items, relevant_items)[:20])
        f1s_list.append(evaluation.f1_curve(ranked_items, relevant_items)[:20])

    cols = ['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10',
            'top11','top12','top13','top14','top15','top16','top17','top18','top19','top20']

    precision_df = pd.DataFrame(precision_list, columns=cols)
    print('Precision CF NMF:')
    print(pd.DataFrame([precision_df.mean()]))
    mean_precision_df = pd.DataFrame([precision_df.mean()])

    mrr_df = pd.DataFrame(mrr_list, columns=cols)
    print('MRR CF NMF:')
    print(pd.DataFrame([mrr_df.mean()]))
    mean_mrr_df = pd.DataFrame([mrr_df.mean()])

    recall_df = pd.DataFrame(recall_list, columns=cols)
    print('Recall CF NMF:')
    print(pd.DataFrame([recall_df.mean()]))
    mean_recall_df = pd.DataFrame([recall_df.mean()])

    f1s_df = pd.DataFrame(f1s_list, columns=cols)
    print('F1-Score CF NMF:')
    print(pd.DataFrame([f1s_df.mean()]))
    mean_f1s_df = pd.DataFrame([f1s_df.mean()])

    return mean_precision_df, mean_mrr_df, mean_recall_df, mean_f1s_df





def build_user_item_feature_matrix(df, feature_cols):

    df_pivot = df.pivot_table(
        index='user',
        columns='item',
        values=feature_cols,
        fill_value=0
    )
   
    df_pivot.columns = [f"{item}_{feat}" for feat, item in df_pivot.columns]
    df_pivot = df_pivot.fillna(0)
    return df_pivot


def _score_test_items_from_flat_matrix(user, test_items, pred_df, feature_cols):

    scores = {}
    for item in test_items:
        item_cols = [f"{item}_{feat}" for feat in feature_cols if f"{item}_{feat}" in pred_df.columns]
        if item_cols:
            scores[item] = pred_df.loc[user, item_cols].sum()
        else:
            scores[item] = 0.0  
    return pd.Series(scores).sort_values(ascending=False)


def _run_matrix_factorization(model, train_df, test_df, feature_cols, label):

    matrix = build_user_item_feature_matrix(train_df, feature_cols)
    users = matrix.index

    user_factors = model.fit_transform(matrix)
    item_factors = model.components_

    predictions = np.dot(user_factors, item_factors)
    pred_df = pd.DataFrame(predictions, index=users, columns=matrix.columns)

    precision_list, mrr_list, recall_list, f1s_list = [], [], [], []

    for user in test_df['user'].unique():
        if user not in pred_df.index:
            continue

        test_items = test_df[test_df['user'] == user]['item'].tolist()
        if not test_items:
            continue

        relevant_items = test_df.loc[
            (test_df['user'] == user) & (test_df['rating'] == 1), 'item'
        ].tolist()
        if not relevant_items:
            continue

        scores = _score_test_items_from_flat_matrix(user, test_items, pred_df, feature_cols)
        ranked_items = scores.index.tolist()

        precision_list.append(evaluation.precision_curve(ranked_items, relevant_items)[:20])
        mrr_list.append(evaluation.mean_reciprocal_rank_at_k(ranked_items, relevant_items)[:20])
        recall_list.append(evaluation.recall_curve(ranked_items, relevant_items)[:20])
        f1s_list.append(evaluation.f1_curve(ranked_items, relevant_items)[:20])

    cols = ['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10',
            'top11','top12','top13','top14','top15','top16','top17','top18','top19','top20']

    precision_df = pd.DataFrame(precision_list, columns=cols)
    mrr_df       = pd.DataFrame(mrr_list,       columns=cols)
    recall_df    = pd.DataFrame(recall_list,    columns=cols)
    f1s_df       = pd.DataFrame(f1s_list,       columns=cols)

    print(f'Precision {label}:') ; print(pd.DataFrame([precision_df.mean()]))
    print(f'MRR {label}:')       ; print(pd.DataFrame([mrr_df.mean()]))
    print(f'Recall {label}:')    ; print(pd.DataFrame([recall_df.mean()]))
    print(f'F1-Score {label}:')  ; print(pd.DataFrame([f1s_df.mean()]))

    return (pd.DataFrame([precision_df.mean()]),
            pd.DataFrame([mrr_df.mean()]),
            pd.DataFrame([recall_df.mean()]),
            pd.DataFrame([f1s_df.mean()]))



# EMOTION COLS / VAD COLS / ALL COLS


EMOTION_COLS = ['anger', 'fear', 'disgust', 'sadness', 'happiness', 'surprise', 'neutral']
VAD_COLS     = ['valence', 'arousal', 'dominance']
ALL_COLS     = ['rating', 'anger', 'fear', 'disgust', 'sadness',
                'happiness', 'surprise', 'neutral', 'valence', 'arousal', 'dominance']



# SVD VARIANTS


def CF_SVD_emotions(train_df, test_df, n, n_factors=5):
  
    model = TruncatedSVD(n_components=n_factors, random_state=42)
    return _run_matrix_factorization(model, train_df, test_df, EMOTION_COLS, f'CF SVD Emotions (n={n})')


def CF_SVD_vad(train_df, test_df, n, n_factors=5):
    
    model = TruncatedSVD(n_components=n_factors, random_state=42)
    return _run_matrix_factorization(model, train_df, test_df, VAD_COLS, f'CF SVD VAD (n={n})')


def CF_SVD_all(train_df, test_df, n, n_factors=5):
    
    model = TruncatedSVD(n_components=n_factors, random_state=42)
    return _run_matrix_factorization(model, train_df, test_df, ALL_COLS, f'CF SVD ALL (n={n})')



# NMF VARIANTS


def CF_NMF_emotions(train_df, test_df, n, n_factors=5):

    model = NMF(n_components=n_factors, init='random', random_state=42, max_iter=500)
    return _run_matrix_factorization(model, train_df, test_df, EMOTION_COLS, f'CF NMF Emotions (n={n})')


def CF_NMF_vad(train_df, test_df, n, n_factors=5):

    model = NMF(n_components=n_factors, init='random', random_state=42, max_iter=500)
    return _run_matrix_factorization(model, train_df, test_df, VAD_COLS, f'CF NMF VAD (n={n})')


def CF_NMF_all(train_df, test_df, n, n_factors=5):

    model = NMF(n_components=n_factors, init='random', random_state=42, max_iter=500)
    return _run_matrix_factorization(model, train_df, test_df, ALL_COLS, f'CF NMF ALL (n={n})')



def CF_random(train_df, test_df, n):
    precision_list, mrr_list, recall_list, f1s_list = [], [], [], []

    for user in test_df['user'].unique():

        relevant_items = test_df.loc[(test_df['user'] == user) & (test_df['rating'] == 1), 'item'].tolist()
        if len(relevant_items) == 0:
            continue

        random_recommended_list = test_df[test_df['user'] == user]['item'].sample(frac=1).reset_index(drop=True).to_list()

        precision_list.append(evaluation.precision_curve(random_recommended_list, relevant_items)[:20])
        mrr_list.append(evaluation.mean_reciprocal_rank_at_k(random_recommended_list, relevant_items)[:20])
        recall_list.append(evaluation.recall_curve(random_recommended_list, relevant_items)[:20])
        f1s_list.append(evaluation.f1_curve(random_recommended_list, relevant_items)[:20])

    cols = ['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20']

    precision_df = pd.DataFrame(precision_list, columns=cols)
    print('Precisio CF random ', n, 'most similar user: ')
    print(pd.DataFrame([precision_df.mean()]))
    mean_precision_df = pd.DataFrame([precision_df.mean()])

    mrr_df = pd.DataFrame(mrr_list, columns=cols)
    print('MRR CF random ', n, 'most similar user: ')
    print(pd.DataFrame([mrr_df.mean()]))
    mean_mrr_df = pd.DataFrame([mrr_df.mean()])

    recall_df = pd.DataFrame(recall_list, columns=cols)
    print('Recall CF random ', n, 'most similar user: ')
    print(pd.DataFrame([recall_df.mean()]))
    mean_recall_df = pd.DataFrame([recall_df.mean()])

    f1s_df = pd.DataFrame(f1s_list, columns=cols)
    print('F1-Score CF random ', n, 'most similar user: ')
    print(pd.DataFrame([f1s_df.mean()]))
    mean_f1s_df = pd.DataFrame([f1s_df.mean()])

    return (mean_precision_df, mean_mrr_df, mean_recall_df, mean_f1s_df)



