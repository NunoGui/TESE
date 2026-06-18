import pandas as pd
import numpy as np
 
 
def remove_degenerate_users(df):
    #remove users com todos os ratings iguais (todos 0 ou todos 1)
    users_all_zeros = df.groupby('user')['rating'].transform(lambda x: (x == 0).all())
    users_all_ones  = df.groupby('user')['rating'].transform(lambda x: (x == 1).all())
    return df[~users_all_zeros & ~users_all_ones].copy()
 
 
def split_user_data(ratings, feature_cols, n_train=12, min_test=20, seed=42):
    """
    12 imagens para treino, 20 para teste
    17/20 negativos aleatórios até atingir min_test itens
    """
    rng = np.random.default_rng(seed)
    all_items = set(ratings['item'].unique())
    train_list, test_list = [], []
 
    for user_id, group in ratings.groupby('user'):
        if len(group) < n_train + 1:
            train_list.append(group)
            continue
 
        shuffled = group.sample(frac=1, random_state=seed)
        train = shuffled.iloc[:n_train]
        test  = shuffled.iloc[n_train:]
 
        seen_items   = set(group['item'])
        unseen       = list(all_items - seen_items)
        n_neg_needed = max(0, min_test - len(test))
 
        if n_neg_needed > 0 and len(unseen) >= n_neg_needed:
            neg_sample = rng.choice(unseen, size=n_neg_needed, replace=False)
            neg_df = pd.DataFrame({'user': user_id, 'item': neg_sample, 'rating': 0})
            for col in feature_cols:
                if col not in neg_df.columns:
                    neg_df[col] = 0
            test = pd.concat([test, neg_df], ignore_index=True)
 
        train_list.append(train)
        test_list.append(test)
 
    train_data = pd.concat(train_list).reset_index(drop=True)
    test_data  = pd.concat(test_list).reset_index(drop=True)
    return train_data, test_data