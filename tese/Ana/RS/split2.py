import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split


def split_user_data(data):
    train_list = []
    test_list = []

    for user_id, user_data in data.groupby('user'):
        if len(user_data) == 1:  
            train_list.append(user_data)
        else:
            train, test = train_test_split(user_data, test_size=0.2, random_state=42)
            

            train_list.append(train)
            test_list.append(test)

    train_data = pd.concat(train_list).reset_index(drop=True)
    test_data = pd.concat(test_list).reset_index(drop=True)

    return train_data, test_data


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def remove_users_all_zero_ratings(df):

    
    users_all_zeros = df.groupby('user')['rating'].transform(lambda x: (x == 0).all())
    users_all_ones = df.groupby('user')['rating'].transform(lambda x: (x == 1).all())

   
    df_filtered = df[~users_all_zeros].copy()
    df_filtered2 = df_filtered[~users_all_ones].copy()

    return df_filtered2




def split_user_data_balanced(
    data, total_items=3084, min_test_items=20, random_state=42
):

    data = remove_users_all_zero_ratings(data)

    train_list = []
    test_list = []

    rng = np.random.default_rng(seed=random_state)

    for user_id, user_data in data.groupby('user'):
        seen_items = set(user_data['item'])
        user_items_count = len(user_data)


        if user_items_count == 1:
            train = user_data
            test = pd.DataFrame(columns=user_data.columns)
        else:
            train, test = train_test_split(
                user_data, test_size=0.2, random_state=random_state
            )


        n_negatives_needed = max(0, min_test_items - len(test))


        all_items = set(range(1, total_items + 1))
        unseen_items = list(all_items - seen_items)

        if len(unseen_items) >= n_negatives_needed and n_negatives_needed > 0:
            negative_samples = rng.choice(unseen_items, size=n_negatives_needed, replace=False)
            negative_df = pd.DataFrame({
                'user': user_id,
                'item': negative_samples,
                'rating': 0  
            })
            test = pd.concat([test, negative_df], ignore_index=True)
        elif n_negatives_needed > 0:

            negative_samples = unseen_items
            negative_df = pd.DataFrame({
                'user': user_id,
                'item': negative_samples,
                'rating': 0
            })
            test = pd.concat([test, negative_df], ignore_index=True)

        train_list.append(train)
        test_list.append(test)

    train_data = pd.concat(train_list).reset_index(drop=True)
    test_data = pd.concat(test_list).reset_index(drop=True)

    return train_data, test_data