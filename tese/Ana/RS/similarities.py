import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim(df):

    user_item_matrix = df.pivot_table(index="user", 
                                    columns="item", 
                                    values="rating", 
                                    fill_value=0)


    cos_sim = cosine_similarity(user_item_matrix)
    cosine_df = pd.DataFrame(cos_sim, 
                            index=user_item_matrix.index, 
                            columns=user_item_matrix.index)

    return cosine_df


def pearson_similarity(df):

    user_item_matrix = df.pivot_table(index="user",
                                      columns="item",
                                      values="rating",
                                      fill_value=0)


    pearson_corr = user_item_matrix.T.corr(method='pearson')
    sim_df = pearson_corr.fillna(0) 

    return sim_df


def cosine_sim_vector(df):
    item_cols = df.columns[3:]
    df_pivot = df.pivot(index='user', columns='item', values=item_cols.tolist())


    df_pivot.columns = [f"{item}_{feat}" for item, feat in df_pivot.columns]


    df_pivot = df_pivot.fillna(0)


    similarity_matrix = cosine_similarity(df_pivot)
    similarity_df = pd.DataFrame(similarity_matrix, index=df_pivot.index, columns=df_pivot.index)


    return similarity_df



def get_k_most_sim_users(similarity_df,target_user, k ):

    if target_user not in similarity_df.index:
        raise ValueError(f"User {target_user} not found in dataset.")

    sim_scores = similarity_df.loc[target_user].drop(target_user)


    return sim_scores.nlargest(k)


def get_scores(test_items, similar_users, df_train):

    user_item_matrix = df_train.pivot_table(index="user", 
                                      columns="item", 
                                      values="rating", 
                                      fill_value=0)
    

    neighbor_ids = similar_users.index
    scores = {}
    for item in test_items:
        if item in user_item_matrix.columns:
            neighbor_ratings = user_item_matrix.loc[neighbor_ids, item]
            scores[item] = neighbor_ratings.mean() 
        else:
            scores[item] = 0.0 

    return pd.Series(scores).sort_values(ascending=False)
