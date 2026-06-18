
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

import similarities
import evaluation










def preprocess_demographics(demo_df, n_components=2):


    demo_df = demo_df.copy()

    user_ids = demo_df['id']


    features = demo_df.drop(columns=['id'])


    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(features)


    pca = PCA(n_components=min(n_components, encoded.shape[1]), random_state=42)
    reduced = pca.fit_transform(encoded)


    print("Explained variance per component:")
    print(pca.explained_variance_ratio_)

    print("\nTotal explained variance:")
    print(pca.explained_variance_ratio_.sum())

    demo_vectors = pd.DataFrame(reduced, index=user_ids)

    return demo_vectors






def DEMOGRAPHIC_knn(train_df, test_df, demo_df, n_neighbors):


    demo_vectors = preprocess_demographics(demo_df)


    rating_users = set(train_df['user']).union(set(test_df['user']))
    demo_vectors = demo_vectors.loc[demo_vectors.index.isin(rating_users)]

  
    sim_matrix = cosine_similarity(demo_vectors)

    similarity_df = pd.DataFrame(
        sim_matrix,
        index=demo_vectors.index,
        columns=demo_vectors.index
    )

    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_mrr = []

    users = test_df['user'].unique()

    for user in users:

        if user not in similarity_df.index:
            continue


        relevant_items = test_df[
            (test_df['user'] == user) &
            (test_df['rating'] == 1)
        ]['item'].tolist()

        if len(relevant_items) == 0:
            continue


        similar_users = similarities.get_k_most_sim_users(
            similarity_df,
            user,
            n_neighbors
        )


        test_items = test_df[test_df['user'] == user]['item'].tolist()
         


        scores = similarities.get_scores(
            test_items,
            similar_users,
            train_df
        )

        recommended_items = scores.index.tolist()


        precision = evaluation.precision_curve(
            recommended_items,
            relevant_items
        )

        recall = evaluation.recall_curve(
            recommended_items,
            relevant_items
        )

        f1 = evaluation.f1_curve(
            recommended_items,
            relevant_items
        )

        mrr = evaluation.mean_reciprocal_rank_at_k(
            recommended_items,
            relevant_items
        )

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_mrr.append(mrr)


    mean_precision = np.mean(all_precisions, axis=0)
    mean_recall = np.mean(all_recalls, axis=0)
    mean_f1 = np.mean(all_f1s, axis=0)
    mean_mrr = np.mean(all_mrr, axis=0)

    mean_precision_df = pd.DataFrame(
        [mean_precision[:20]],
        columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20']
    )

    mean_recall_df = pd.DataFrame(
        [mean_recall[:20]],
        columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20']
    )

    mean_f1_df = pd.DataFrame(
        [mean_f1[:20]],
        columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20']
    )

    mean_mrr_df = pd.DataFrame(
        [mean_mrr[:20]],
        columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20']
    )

    return (
        mean_precision_df,
        mean_mrr_df,
        mean_recall_df,
        mean_f1_df
    )





################### RANKINGS KNN ###############################

def DEMOGRAPHIC_knn_rankings(train_df, test_df, demo_df, n_neighbors):
    import similarities
    from sklearn.metrics.pairwise import cosine_similarity

    demo_vectors = preprocess_demographics(demo_df)

    rating_users = set(train_df['user']).union(set(test_df['user']))
    demo_vectors = demo_vectors.loc[demo_vectors.index.isin(rating_users)]

    sim_matrix = cosine_similarity(demo_vectors)

    similarity_df = pd.DataFrame(
        sim_matrix,
        index=demo_vectors.index,
        columns=demo_vectors.index
    )

    user_rankings = {}

    for user in test_df['user'].unique():

        if user not in similarity_df.index:
            continue

        similar_users = similarities.get_k_most_sim_users(
            similarity_df,
            user,
            n_neighbors
        )

        test_items = test_df[test_df['user'] == user]['item'].tolist()

        scores = similarities.get_scores(
            test_items,
            similar_users,
            train_df
        )

        user_rankings[user] = scores.sort_values(ascending=False).index.tolist()

    return user_rankings




def DEMOGRAPHIC_kmeans(train_df, test_df, demo_df, n_clusters=3):

    demo_vectors = preprocess_demographics(demo_df)

    rating_users = set(train_df['user']).union(set(test_df['user']))
    demo_vectors = demo_vectors.loc[demo_vectors.index.isin(rating_users)]

    # --- clustering ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(demo_vectors)

    cluster_map = pd.Series(clusters, index=demo_vectors.index)

    all_precisions, all_recalls, all_f1s, all_mrr = [], [], [], []

    users = test_df['user'].unique()

    for user in users:

        if user not in cluster_map.index:
            continue

        user_cluster = cluster_map[user]


        similar_users = cluster_map[cluster_map == user_cluster].index

        relevant_items = test_df.loc[
            (test_df['user'] == user) & (test_df['rating'] == 1),
            'item'
        ].tolist()

        if len(relevant_items) == 0:
            continue

        test_items = test_df[test_df['user'] == user]['item'].tolist()


        user_item_matrix = train_df.pivot_table(
            index="user", columns="item", values="rating", fill_value=0
        )

        scores = {}
        for item in test_items:
            if item in user_item_matrix.columns:
                scores[item] = user_item_matrix.loc[similar_users, item].mean()
            else:
                scores[item] = 0.0

        recommended_items = sorted(scores, key=scores.get, reverse=True)

        all_precisions.append(evaluation.precision_curve(recommended_items, relevant_items))
        all_recalls.append(evaluation.recall_curve(recommended_items, relevant_items))
        all_f1s.append(evaluation.f1_curve(recommended_items, relevant_items))
        all_mrr.append(evaluation.mean_reciprocal_rank_at_k(recommended_items, relevant_items))

    mean_precision = np.mean(all_precisions, axis=0)
    mean_recall = np.mean(all_recalls, axis=0)
    mean_f1 = np.mean(all_f1s, axis=0)
    mean_mrr = np.mean(all_mrr, axis=0)

    cols = [f"top{i}" for i in range(1, 21)]

    return (
        pd.DataFrame([mean_precision[:20]], columns=cols),
        pd.DataFrame([mean_mrr[:20]], columns=cols),
        pd.DataFrame([mean_recall[:20]], columns=cols),
        pd.DataFrame([mean_f1[:20]], columns=cols),
    )





def DEMOGRAPHIC_dbscan(train_df, test_df, demo_df, eps=0.5, min_samples=2):

    demo_vectors = preprocess_demographics(demo_df)

    rating_users = set(train_df['user']).union(set(test_df['user']))
    demo_vectors = demo_vectors.loc[demo_vectors.index.isin(rating_users)]

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    clusters = db.fit_predict(demo_vectors)

    cluster_map = pd.Series(clusters, index=demo_vectors.index)

    user_item_matrix = train_df.pivot_table(
        index="user", columns="item", values="rating", fill_value=0
    )

    all_precisions, all_recalls, all_f1s, all_mrr = [], [], [], []

    for user in test_df['user'].unique():

        if user not in cluster_map.index:
            continue

        label = cluster_map[user]


        if label == -1:
            similar_users = user_item_matrix.index  
        else:
            similar_users = cluster_map[cluster_map == label].index

        relevant_items = test_df.loc[
            (test_df['user'] == user) & (test_df['rating'] == 1),
            'item'
        ].tolist()

        if len(relevant_items) == 0:
            continue

        test_items = test_df[test_df['user'] == user]['item'].tolist()

        scores = {}
        for item in test_items:
            if item in user_item_matrix.columns:
                scores[item] = user_item_matrix.loc[similar_users, item].mean()
            else:
                scores[item] = 0.0

        recommended_items = sorted(scores, key=scores.get, reverse=True)

        all_precisions.append(evaluation.precision_curve(recommended_items, relevant_items))
        all_recalls.append(evaluation.recall_curve(recommended_items, relevant_items))
        all_f1s.append(evaluation.f1_curve(recommended_items, relevant_items))
        all_mrr.append(evaluation.mean_reciprocal_rank_at_k(recommended_items, relevant_items))

    mean_precision = np.mean(all_precisions, axis=0)
    mean_recall = np.mean(all_recalls, axis=0)
    mean_f1 = np.mean(all_f1s, axis=0)
    mean_mrr = np.mean(all_mrr, axis=0)

    cols = [f"top{i}" for i in range(1, 21)]

    return (
        pd.DataFrame([mean_precision[:20]], columns=cols),
        pd.DataFrame([mean_mrr[:20]], columns=cols),
        pd.DataFrame([mean_recall[:20]], columns=cols),
        pd.DataFrame([mean_f1[:20]], columns=cols),
    )






def DEMOGRAPHIC_spectral(train_df, test_df, demo_df, n_clusters=3):

    demo_vectors = preprocess_demographics(demo_df)

    rating_users = set(train_df['user']).union(set(test_df['user']))
    demo_vectors = demo_vectors.loc[demo_vectors.index.isin(rating_users)]

    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        random_state=42
    )

    clusters = spectral.fit_predict(demo_vectors)

    cluster_map = pd.Series(clusters, index=demo_vectors.index)

    user_item_matrix = train_df.pivot_table(
        index="user", columns="item", values="rating", fill_value=0
    )

    all_precisions, all_recalls, all_f1s, all_mrr = [], [], [], []

    for user in test_df['user'].unique():

        if user not in cluster_map.index:
            continue

        user_cluster = cluster_map[user]
        similar_users = cluster_map[cluster_map == user_cluster].index

        relevant_items = test_df.loc[
            (test_df['user'] == user) & (test_df['rating'] == 1),
            'item'
        ].tolist()

        if len(relevant_items) == 0:
            continue

        test_items = test_df[test_df['user'] == user]['item'].tolist()

        scores = {}
        for item in test_items:
            if item in user_item_matrix.columns:
                scores[item] = user_item_matrix.loc[similar_users, item].mean()
            else:
                scores[item] = 0.0

        recommended_items = sorted(scores, key=scores.get, reverse=True)

        all_precisions.append(evaluation.precision_curve(recommended_items, relevant_items))
        all_recalls.append(evaluation.recall_curve(recommended_items, relevant_items))
        all_f1s.append(evaluation.f1_curve(recommended_items, relevant_items))
        all_mrr.append(evaluation.mean_reciprocal_rank_at_k(recommended_items, relevant_items))

    mean_precision = np.mean(all_precisions, axis=0)
    mean_recall = np.mean(all_recalls, axis=0)
    mean_f1 = np.mean(all_f1s, axis=0)
    mean_mrr = np.mean(all_mrr, axis=0)

    cols = [f"top{i}" for i in range(1, 21)]

    return (
        pd.DataFrame([mean_precision[:20]], columns=cols),
        pd.DataFrame([mean_mrr[:20]], columns=cols),
        pd.DataFrame([mean_recall[:20]], columns=cols),
        pd.DataFrame([mean_f1[:20]], columns=cols),
    )

def DEMOGRAPHIC_random_forest(train_df, test_df, demo_df, n_neighbors=5, n_estimators=100):

    demo_vectors = preprocess_demographics(demo_df)
    rating_users = set(train_df['user']).union(set(test_df['user']))
    demo_vectors = demo_vectors.loc[demo_vectors.index.isin(rating_users)]


    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    X = demo_vectors.values
    y = np.zeros(len(X)) 
    rf.fit(X, y)


    leaf_indices = rf.apply(X)
    similarity_matrix = cosine_similarity(leaf_indices)
    similarity_df = pd.DataFrame(similarity_matrix, index=demo_vectors.index, columns=demo_vectors.index)

    all_precisions, all_recalls, all_f1s, all_mrr = [], [], [], []

    for user in test_df['user'].unique():
        if user not in similarity_df.index:
            continue

        relevant_items = test_df.loc[
            (test_df['user'] == user) & (test_df['rating'] == 1),
            'item'
        ].tolist()

        if len(relevant_items) == 0:
            continue

        similar_users = similarities.get_k_most_sim_users(similarity_df, user, n_neighbors)
        test_items = test_df[test_df['user'] == user]['item'].tolist()
        scores = similarities.get_scores(test_items, similar_users, train_df)
        recommended_items = scores.index.tolist()

        all_precisions.append(evaluation.precision_curve(recommended_items, relevant_items))
        all_recalls.append(evaluation.recall_curve(recommended_items, relevant_items))
        all_f1s.append(evaluation.f1_curve(recommended_items, relevant_items))
        all_mrr.append(evaluation.mean_reciprocal_rank_at_k(recommended_items, relevant_items))

    mean_precision = np.mean(all_precisions, axis=0)
    mean_recall = np.mean(all_recalls, axis=0)
    mean_f1 = np.mean(all_f1s, axis=0)
    mean_mrr = np.mean(all_mrr, axis=0)

    mean_precision_df = pd.DataFrame([mean_precision[:20]], columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mean_recall_df = pd.DataFrame([mean_recall[:20]], columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mean_f1_df = pd.DataFrame([mean_f1[:20]], columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mean_mrr_df = pd.DataFrame([mean_mrr[:20]], columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])

    return mean_precision_df, mean_mrr_df, mean_recall_df, mean_f1_df







