import pandas as pd
import matplotlib.pyplot as plt
#

import pandas as pd
import matplotlib.pyplot as plt


csv_files = [
    r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/precision_all_n1_kfold5.csv"#,
    #r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/precision_all_n5_kfold5.csv",
    #r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/precision_all_n10_kfold5.csv",
    #r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/precision_all_n20_kfold5.csv"
]


titles = [
    "n = 1",
    "n = 5",
    "n = 10",
    "n = 20"
]




name_mapping = {
    "CF_likes":"CF-KNN-Likes",
    "CF_emotions":"CF-KNN-Emo",
    "CF_vad":"CF-KNN-VAD",
    "CF_all":"CF-KNN-All",
    "SVD_likes":"CF-SVD-Likes",
    "SVD_emotions":"CF-SVD-Emo",
    "SVD_vad":"CF-SVD-VAD",
    "SVD_all":"CF-SVD-All",
    "NMF_likes":"CF-NMF-Likes",
    "NMF_emotions":"CF-NMF-Emo",
    "NMF_vad":"CF-NMF-VAD",
    "NMF_all":"CF-NMF-All",
    "Demo_KNN":"D-KNN",
    "Demo_KMeans":"D-K-means",
    "Demo_DBSCAN":"D-DBScan",
    "Demo_Spectral":"D-SP",
    "Demo_RandomForest": "D-RForest",
    "CF_random":"Random",
    "Hybrid_CF_Demo":"CF-Like-Demo",
    "Hybrid_CF_Emotions_Demo":"CF-Emo-Demo",
    "Hybrid_CF_VAD_Demo":"CF-VAD-Demo",
    "Hybrid_CF_ALL_Demo":"CF-ALL-Demo"
}


hybrid_models = {
    "Hybrid_CF_Demo",
    "Hybrid_CF_Emotions_Demo",
    "Hybrid_CF_VAD_Demo",
    "Hybrid_CF_ALL_Demo"
}

demo_models = {
    "Demo_KNN",
    "Demo_KMeans",
    "Demo_DBSCAN",
    "Demo_Spectral",
    "Demo_RandomForest",
}


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()


for i, (csv, ax) in enumerate(zip(csv_files, axes)):

    df = pd.read_csv(csv)

    
    model_col = df.columns[0]

    
    value_cols = df.columns[1:]
    for _, row in df.iterrows():

        original_label = str(row[model_col]).strip()

        
        label = name_mapping.get(original_label, original_label)

        
        if original_label == "NMF_emotions":
            color = "darkblue"
        elif original_label == "NMF_vad":
            color = "deeppink"
        elif original_label == "NMF_all":
            color = "#800020"
        elif original_label == "Hybrid_CF_Emotions_Demo":
            color = "yellow"
        elif original_label == "Hybrid_CF_VAD_Demo":
            color = "black"
        else:
            color = None

    
        if original_label in hybrid_models:
            marker = "^"      
        elif original_label in demo_models:
            marker="s"
        else:
            marker = "o"      

        y_values = row[value_cols].astype(float).values

        ax.plot(
            range(1, len(value_cols) + 1),
            y_values,
            marker=marker,
            label=label,
            color=color,
            markersize=7
    )
    
    ax.set_title(titles[i], fontsize=16)
    ax.set_xlabel("Top@k", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)

    ax.set_xticks(range(1, len(value_cols) + 1))

    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)

    
    
    ax.legend(
        fontsize=7,          
        loc='upper right',   
        frameon=True,        
        framealpha=0.8,      
        borderpad=0.3,
        labelspacing=0.3,
        handletextpad=0.4

    )

for ax in axes[len(csv_files):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()





csv_files = [
    r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/recall_all_n1_kfold5.csv"#,
    #r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/recall_all_n5_kfold5.csv",
    #r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/recall_all_n10_kfold5.csv",
    #r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/recall_all_n20_kfold5.csv"
]

# Titles for each subplot
titles = [
    "n = 1",
    "n = 5",
    "n = 10",
    "n = 20"
]




name_mapping = {
    "CF_likes":"CF-KNN-Likes",
    "CF_emotions":"CF-KNN-Emo",
    "CF_vad":"CF-KNN-VAD",
    "CF_all":"CF-KNN-All",
    "SVD_likes":"CF-SVD-Likes",
    "SVD_emotions":"CF-SVD-Emo",
    "SVD_vad":"CF-SVD-VAD",
    "SVD_all":"CF-SVD-All",
    "NMF_likes":"CF-NMF-Likes",
    "NMF_emotions":"CF-NMF-Emo",
    "NMF_vad":"CF-NMF-VAD",
    "NMF_all":"CF-NMF-All",
    "Demo_KNN":"D-KNN",
    "Demo_KMeans":"D-K-means",
    "Demo_DBSCAN":"D-DBScan",
    "Demo_Spectral":"D-SP",
    "Demo_RandomForest": "D-RForest",
    "CF_random":"Random",
    "Hybrid_CF_Demo":"CF-Like-Demo",
    "Hybrid_CF_Emotions_Demo":"CF-Emo-Demo",
    "Hybrid_CF_VAD_Demo":"CF-VAD-Demo",
    "Hybrid_CF_ALL_Demo":"CF-ALL-Demo"
}


hybrid_models = {
    "Hybrid_CF_Demo",
    "Hybrid_CF_Emotions_Demo",
    "Hybrid_CF_VAD_Demo",
    "Hybrid_CF_ALL_Demo"
}

demo_models = {
    "Demo_KNN",
    "Demo_KMeans",
    "Demo_DBSCAN",
    "Demo_Spectral",
    "Demo_RandomForest",
}


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()


for i, (csv, ax) in enumerate(zip(csv_files, axes)):

    df = pd.read_csv(csv)

    
    model_col = df.columns[0]

    
    value_cols = df.columns[1:]
    for _, row in df.iterrows():

        original_label = str(row[model_col]).strip()

        
        label = name_mapping.get(original_label, original_label)

        
        if original_label == "NMF_emotions":
            color = "darkblue"
        elif original_label == "NMF_vad":
            color = "deeppink"
        elif original_label == "NMF_all":
            color = "#800020"
        elif original_label == "Hybrid_CF_Emotions_Demo":
            color = "yellow"
        elif original_label == "Hybrid_CF_VAD_Demo":
            color = "black"
        else:
            color = None

    
        if original_label in hybrid_models:
            marker = "^"      # triangle
        elif original_label in demo_models:
            marker="s"
        else:
            marker = "o"      # circle

        y_values = row[value_cols].astype(float).values

        ax.plot(
            range(1, len(value_cols) + 1),
            y_values,
            marker=marker,
            label=label,
            color=color,
            markersize=7
    )
    
    ax.set_title(titles[i], fontsize=16)
    ax.set_xlabel("Top@k", fontsize=14)
    ax.set_ylabel("Recall", fontsize=14)

    ax.set_xticks(range(1, len(value_cols) + 1))

    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)
    '''
    ax.legend(
        fontsize=8,
        loc='lower right'
    '''
    
    
    ax.legend(
        fontsize=7,          
        loc='lower right',   
        frameon=True,        
        framealpha=0.8,      
        borderpad=0.3,
        labelspacing=0.3,
        handletextpad=0.4

    )


for ax in axes[len(csv_files):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()



csv_files = [
    r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/mrr_all_n1_kfold5.csv"#,
    #r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/mrr_all_n5_kfold5.csv",
    #r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/mrr_all_n10_kfold5.csv",
    #r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/mrr_all_n20_kfold5.csv"
]


titles = [
    "n = 1",
    "n = 5",
    "n = 10",
    "n = 20"
]




name_mapping = {
    "CF_likes":"CF-KNN-Likes",
    "CF_emotions":"CF-KNN-Emo",
    "CF_vad":"CF-KNN-VAD",
    "CF_all":"CF-KNN-All",
    "SVD_likes":"CF-SVD-Likes",
    "SVD_emotions":"CF-SVD-Emo",
    "SVD_vad":"CF-SVD-VAD",
    "SVD_all":"CF-SVD-All",
    "NMF_likes":"CF-NMF-Likes",
    "NMF_emotions":"CF-NMF-Emo",
    "NMF_vad":"CF-NMF-VAD",
    "NMF_all":"CF-NMF-All",
    "Demo_KNN":"D-KNN",
    "Demo_KMeans":"D-K-means",
    "Demo_DBSCAN":"D-DBScan",
    "Demo_Spectral":"D-SP",
    "Demo_RandomForest": "D-RForest",
    "CF_random":"Random",
    "Hybrid_CF_Demo":"CF-Like-Demo",
    "Hybrid_CF_Emotions_Demo":"CF-Emo-Demo",
    "Hybrid_CF_VAD_Demo":"CF-VAD-Demo",
    "Hybrid_CF_ALL_Demo":"CF-ALL-Demo"
}


hybrid_models = {
    "Hybrid_CF_Demo",
    "Hybrid_CF_Emotions_Demo",
    "Hybrid_CF_VAD_Demo",
    "Hybrid_CF_ALL_Demo"
}

demo_models = {
    "Demo_KNN",
    "Demo_KMeans",
    "Demo_DBSCAN",
    "Demo_Spectral",
    "Demo_RandomForest",
}


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()


for i, (csv, ax) in enumerate(zip(csv_files, axes)):

    df = pd.read_csv(csv)


    model_col = df.columns[0]


    value_cols = df.columns[1:]
    for _, row in df.iterrows():

        original_label = str(row[model_col]).strip()


        label = name_mapping.get(original_label, original_label)


        if original_label == "NMF_emotions":
            color = "darkblue"
        elif original_label == "NMF_vad":
            color = "deeppink"
        elif original_label == "NMF_all":
            color = "#800020"
        elif original_label == "Hybrid_CF_Emotions_Demo":
            color = "yellow"
        elif original_label == "Hybrid_CF_VAD_Demo":
            color = "black"
        else:
            color = None


        if original_label in hybrid_models:
            marker = "^"      
        elif original_label in demo_models:
            marker="s"
        else:
            marker = "o"      

        y_values = row[value_cols].astype(float).values

        ax.plot(
            range(1, len(value_cols) + 1),
            y_values,
            marker=marker,
            label=label,
            color=color,
            markersize=7
    )
    
    ax.set_title(titles[i], fontsize=16)
    ax.set_xlabel("Top@k", fontsize=14)
    ax.set_ylabel("MRR", fontsize=14)

    ax.set_xticks(range(1, len(value_cols) + 1))

    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)
    '''
    ax.legend(
        fontsize=8,
        loc='lower right'
    '''
    
    
    ax.legend(
        fontsize=7,          
        loc='lower right',   
        frameon=True,        
        framealpha=0.8,      
        borderpad=0.3,
        labelspacing=0.3,
        handletextpad=0.4

    )


for ax in axes[len(csv_files):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()



csv_files = [
    r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/f1s_all_n1_kfold5.csv"#,
    #r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/f1s_all_n5_kfold5.csv",
    #r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/f1s_all_n10_kfold5.csv",
    #r"C:/Users/anado/Documents/FORMACAO IT/FCUL/Dissertacao/04 - Scripts/code_mb-20251019 - alterado Ana/code_ad_mb_sma_v38/f1s_all_n20_kfold5.csv"
]


titles = [
    "n = 1",
    "n = 5",
    "n = 10",
    "n = 20"
]




name_mapping = {
    "CF_likes":"CF-KNN-Likes",
    "CF_emotions":"CF-KNN-Emo",
    "CF_vad":"CF-KNN-VAD",
    "CF_all":"CF-KNN-All",
    "SVD_likes":"CF-SVD-Likes",
    "SVD_emotions":"CF-SVD-Emo",
    "SVD_vad":"CF-SVD-VAD",
    "SVD_all":"CF-SVD-All",
    "NMF_likes":"CF-NMF-Likes",
    "NMF_emotions":"CF-NMF-Emo",
    "NMF_vad":"CF-NMF-VAD",
    "NMF_all":"CF-NMF-All",
    "Demo_KNN":"D-KNN",
    "Demo_KMeans":"D-K-means",
    "Demo_DBSCAN":"D-DBScan",
    "Demo_Spectral":"D-SP",
    "Demo_RandomForest": "D-RForest",
    "CF_random":"Random",
    "Hybrid_CF_Demo":"CF-Like-Demo",
    "Hybrid_CF_Emotions_Demo":"CF-Emo-Demo",
    "Hybrid_CF_VAD_Demo":"CF-VAD-Demo",
    "Hybrid_CF_ALL_Demo":"CF-ALL-Demo"
}


hybrid_models = {
    "Hybrid_CF_Demo",
    "Hybrid_CF_Emotions_Demo",
    "Hybrid_CF_VAD_Demo",
    "Hybrid_CF_ALL_Demo"
}

demo_models = {
    "Demo_KNN",
    "Demo_KMeans",
    "Demo_DBSCAN",
    "Demo_Spectral",
    "Demo_RandomForest",
}


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()


for i, (csv, ax) in enumerate(zip(csv_files, axes)):

    df = pd.read_csv(csv)

    
    model_col = df.columns[0]

    
    value_cols = df.columns[1:]
    for _, row in df.iterrows():

        original_label = str(row[model_col]).strip()

        
        label = name_mapping.get(original_label, original_label)

        
        if original_label == "NMF_emotions":
            color = "darkblue"
        elif original_label == "NMF_vad":
            color = "deeppink"
        elif original_label == "NMF_all":
            color = "#800020"
        elif original_label == "Hybrid_CF_Emotions_Demo":
            color = "yellow"
        elif original_label == "Hybrid_CF_VAD_Demo":
            color = "black"
        else:
            color = None


        if original_label in hybrid_models:
            marker = "^"      
        elif original_label in demo_models:
            marker="s"
        else:
            marker = "o"      

        y_values = row[value_cols].astype(float).values

        ax.plot(
            range(1, len(value_cols) + 1),
            y_values,
            marker=marker,
            label=label,
            color=color,
            markersize=7
    )
    
    ax.set_title(titles[i], fontsize=16)
    ax.set_xlabel("Top@k", fontsize=14)
    ax.set_ylabel("F1 Score", fontsize=14)

    ax.set_xticks(range(1, len(value_cols) + 1))

    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)

    
    
    ax.legend(
        fontsize=7,          
        loc='upper right',   
        frameon=True,       
        framealpha=0.8,      
        borderpad=0.3,
        labelspacing=0.3,
        handletextpad=0.4

    )


for ax in axes[len(csv_files):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()


