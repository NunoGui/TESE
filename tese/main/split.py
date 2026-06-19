import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
 
 
def remove_degenerate_users(df):
    """Remove users com todos os ratings iguais (todos 0 ou todos 1)"""
    users_all_zeros = df.groupby('user')['rating'].transform(lambda x: (x == 0).all())
    users_all_ones  = df.groupby('user')['rating'].transform(lambda x: (x == 1).all())
    return df[~users_all_zeros & ~users_all_ones].copy()
 
 
def split_user_data_balanced(data, total_items=3084, min_test_items=20, random_state=42):
    """
    Split baseado na metodologia da Ana (split2.py):
    - 80/20 split por user
    - Completa o teste com negativos aleatórios até atingir min_test_items
    - Negativos têm rating=0
    """
    data = remove_degenerate_users(data)
 
    train_list = []
    test_list  = []
 
    rng = np.random.default_rng(seed=random_state)
    all_items = set(range(1, total_items + 1))
 
    for user_id, user_data in data.groupby('user'):
        seen_items = set(user_data['item'])
 
        if len(user_data) == 1:
            train_list.append(user_data)
            continue
 
        train, test = train_test_split(
            user_data, test_size=0.2, random_state=random_state
        )
 
        # Completar teste com negativos até min_test_items
        n_negatives_needed = max(0, min_test_items - len(test))
        unseen_items = list(all_items - seen_items)
 
        if n_negatives_needed > 0 and len(unseen_items) >= n_negatives_needed:
            negative_samples = rng.choice(unseen_items, size=n_negatives_needed, replace=False)
            negative_df = pd.DataFrame({
                'user':   user_id,
                'item':   negative_samples,
                'rating': 0
            })
            # Preencher restantes colunas com 0
            for col in user_data.columns:
                if col not in negative_df.columns:
                    negative_df[col] = 0
            test = pd.concat([test, negative_df], ignore_index=True)
        elif n_negatives_needed > 0:
            negative_samples = unseen_items
            negative_df = pd.DataFrame({
                'user':   user_id,
                'item':   negative_samples,
                'rating': 0
            })
            for col in user_data.columns:
                if col not in negative_df.columns:
                    negative_df[col] = 0
            test = pd.concat([test, negative_df], ignore_index=True)
 
        train_list.append(train)
        test_list.append(test)
 
    train_data = pd.concat(train_list).reset_index(drop=True)
    test_data  = pd.concat(test_list).reset_index(drop=True)
 
    return train_data, test_data
 
 
def kfold_split(data, k_folds=5, total_items=3084, min_test_items=20, base_seed=42):
    """
    Gerador de k folds seguindo a metodologia da Ana:
    random_state = base_seed + k para cada fold
    """
    for k in range(k_folds):
        train_data, test_data = split_user_data_balanced(
            data,
            total_items=total_items,
            min_test_items=min_test_items,
            random_state=base_seed + k
        )
        yield k, train_data, test_data