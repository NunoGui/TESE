from sklearn.model_selection import train_test_split
import pandas as pd

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

def split_user_data_balanced(data, test_size=0.2, random_state=42, max_test_items=3):
    
    data = remove_users_all_zero_ratings(data)
   
    train_list = []
    test_list = []

    for user_id, user_data in data.groupby('user'):
        if len(user_data) == 1:  
            train_list.append(user_data)
        else:
            train, test = train_test_split(user_data, test_size=0.2, random_state=42)

            if (test['rating'] == 0).all():
         

                candidate = train[train['rating'] == 0].sample(1,random_state=42)
                
                replace_idx = np.random.choice(test.index)
                
                
                test.loc[replace_idx] = candidate.iloc[0]

            elif (test['rating'] == 1).all():
     

                candidate = train[train['rating'] == 0].sample(1,random_state=42)
                replace_idx = np.random.choice(test.index)
                test.loc[replace_idx] = candidate.iloc[0]


            train_list.append(train)
            test_list.append(test)

    train_data = pd.concat(train_list).reset_index(drop=True)
    test_data = pd.concat(test_list).reset_index(drop=True)


    return train_data, test_data