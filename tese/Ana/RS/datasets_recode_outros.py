import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#

def load_csv(file_path):
    try:
        
        df = pd.read_csv(file_path, delimiter=';')
        
        
        print("Columns found in the CSV:")
        for col in df.columns:
            print(f"- {col}")

        return df
    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def transform_ratings(dataset):
    
    dataset = dataset[['id_survey', 'id_photo', 'like_bool']]
    dataset = dataset.rename(columns={'id_survey': 'user', 'id_photo': 'item','like_bool': 'rating' })

    return dataset



def transform_ratings_emotions(dataset):
    
    dataset = dataset[['id_survey', 'id_photo', 'like_bool', 'anger', 'fear', 'disgust', 'sadness', 'happiness', 'surprise', 'neutral'  ]]
    dataset = dataset.rename(columns={'id_survey': 'user', 'id_photo': 'item','like_bool': 'rating' })

    dataset = stadardize(dataset, ['anger', 'fear', 'disgust', 'sadness', 'happiness', 'surprise', 'neutral' ])
    return dataset

def transform_ratings_vad(dataset):
    
    dataset = dataset[['id_survey', 'id_photo', 'like_bool', 'valence', 'arousal', 'dominance' ]]
    dataset = dataset.rename(columns={'id_survey': 'user', 'id_photo': 'item','like_bool': 'rating' })
    dataset = stadardize(dataset, ['valence', 'arousal', 'dominance'])
    return dataset

def transform_ratings_all(dataset):
    
    #dataset = dataset[['id_survey', 'id_photo', 'like_bool', 'valence', 'arousal', 'dominance' ]]
    dataset = dataset.rename(columns={'id_survey': 'user', 'id_photo': 'item','like_bool': 'rating' })
    #dataset = stadardize(dataset, ['anger', 'fear', 'disgust', 'sadness', 'happiness', 'surprise', 'neutral', 'valence', 'arousal', 'dominance' ])
    return dataset


def transform_demographics(dataset):
    
    dataset = dataset[['id', 'age', 'populational_aff', 'gender', 'education', 'city', 'country_residence', 'date_survey', 'consented', 'hobby_other', 
                                      'Board games', 'Cinema', 'Cooking', 'Crafts', 'Cycling', 'Dancing', 'Drawing', 'Gardening', 
                                      'Hiking', 'Music', 'Painting', 'Photography', 'Reading', 'Running', 'Sculpture', 'Swimming', 'Theater',
                                      'Video games', 'Age-Related Macular Degeneration', 'Amblyopia', 'Astigmatism', 'Cataract', 
                                      'Color Blindness', 'Dry Eye Syndrome', 'Glaucoma', 'Hyperopia', 'Keratoconus', 'Myopia', 'Presbyopia', 
                                      'Strabismus', 'Blind or have serious difficulty seeing, even when wearing glasses', 
                                      'Chronic illness that is neurological, physical, or a mental health diagnosis (e.g., dementia)',
                                      'Deaf or have serious difficulty hearing', 'I do not have a disability or impairment', 'Learning disability (e.g., dyslexia, dyscalculia)' , 
                                      'Mobility limitation, including serious difficulty walking or climbing stairs', 'Motor limitation, including manual dexterity' , 
                                      'Neurodiverse (e.g., autism, Attention Deficit and Hyperactivity Disorder)' , 'Prefer not to disclose', 'Speech or language impairment' , 'Temporary impairment']]



    age_mapping = {
        '51 - 60': '51 - 80',
        '61 - 70': '51 - 80',
        '71 - 80': '51 - 80'
    }
    dataset['age'] = dataset['age'].replace(age_mapping)


    education_mapping = {
        'Primary School or Less': 'Other',
        'Middle School': 'Other'
        
    }
    dataset['education'] = dataset['education'].replace(education_mapping)

    dataset['country_residence'] = np.where(dataset['country_residence'] == 'Portugal', 'Portugal', 'Other')

    dataset['city'] = np.where(dataset['city'] == 'Lisboa', 'Lisboa', 'Other')

    dataset['populational_aff'] = np.where(dataset['populational_aff'] == 'White/Caucasian', 'White/Caucasian', 'OTher')

    dataset['Other diseases'] = dataset[['Deaf or have serious difficulty hearing', 'Learning disability (e.g., dyslexia, dyscalculia)', 'Mobility limitation, including serious difficulty walking or climbing stairs', 'Motor limitation, including manual dexterity', 'Prefer not to disclose', 'Speech or language impairment', 'Temporary impairment']].sum(axis=1)

    
    dataset = dataset.drop(columns=['Deaf or have serious difficulty hearing', 'Learning disability (e.g., dyslexia, dyscalculia)', 'Mobility limitation, including serious difficulty walking or climbing stairs', 'Motor limitation, including manual dexterity', 'Prefer not to disclose', 'Speech or language impairment', 'Temporary impairment'])

    
    dataset['Other acuity issues'] = dataset[['Age-Related Macular Degeneration', 'Amblyopia', 'Cataract','Color Blindness','Dry Eye Syndrome', 'Glaucoma',  'Keratoconus', 'Presbyopia', 'Strabismus', 'Blind or have serious difficulty seeing, even when wearing glasses']].sum(axis=1)

    
    dataset = dataset.drop(columns=['Age-Related Macular Degeneration', 'Amblyopia', 'Cataract','Color Blindness','Dry Eye Syndrome', 'Glaucoma',  'Keratoconus', 'Presbyopia', 'Strabismus', 'Blind or have serious difficulty seeing, even when wearing glasses'])


    dataset = dataset.rename(columns={'id': 'id', 'age': 'age','populational_aff': 'populational_aff', 
                                      'gender':'gender', 'education':'education', 'city':'city', 'country_residence':'country_residence', 'date_survey':'date_survey', 'consented':'consented', 'hobby_other':'hobby_other', 
                                      'Board games':'Board_games', 'Cinema':'Cinema', 'Cooking':'Cooking', 'Crafts':'Crafts', 'Cycling':'Cycling', 'Dancing':'Dancing', 'Drawing':'Drawing', 'Gardening':'Gardening', 
                                      'Hiking':'Hiking', 'Music':'Music', 'Painting':'Painting', 'Photography':'Photography', 'Reading':'Reading', 'Running':'Running', 'Sculpture':'Sculpture', 'Swimming':'Swimming', 'Theater':'Theater',
                                      'Video games':'Video_games','Astigmatism':'Astigmatism', 'Hyperopia':'Hypermetropia','Myopia':'Myopia', 'Other acuity issues':'Other acuity issues','Chronic illness that is neurological, physical, or a mental health diagnosis (e.g., dementia)':'Chronic illness',
                                      'I do not have a disability or impairment':'No_disability_impairment',  
                                      'Neurodiverse (e.g., autism, Attention Deficit and Hyperactivity Disorder)':'Neurodiverse', 'Other diseases': 'Other diseases'})
    
	 												

    return dataset
    







	 												


def generate_dummy_similarity_data(ids, similarity_range=(0.1, 1.0)):
   
    item_pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            item_pairs.append((ids[i], ids[j]))
    
    
    similarities = np.random.uniform(similarity_range[0], similarity_range[1], len(item_pairs))
    
   
    similarity_data = pd.DataFrame(item_pairs, columns=['item1', 'item2'])
    similarity_data['sim'] = similarities
    
    
    return similarity_data

## Usar MinMaxScaler
def stadardize(df, list_of_columns):
    scaler = StandardScaler()

    cols_to_standardize = list_of_columns

    
    df[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])

    return df