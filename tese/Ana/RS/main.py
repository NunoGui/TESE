
import datasets_recode_outros
import numpy as np
import sys
import rs_cf_algorithms
import pandas as pd
#import split
import split2
import rs_demographic_algorithms
import rs_cf_demo_algorithms

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random

np.random.seed(42)
random.seed(42)

def main():

   

    ratings = datasets_recode_outros.load_csv("data/ratings_1025.csv")
    demographics0 = pd.read_csv("data/demographics_test.csv")

 

    ratings = datasets_recode_outros.transform_ratings_all(ratings)


    demographics = datasets_recode_outros.transform_demographics(demographics0)


    

    precision_cf_likes = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20' ])
    mrr_cf_likes = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_cf_likes = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_cf_likes = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])

    precision_cf_emotions = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mrr_cf_emotions = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_cf_emotions = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_cf_emotions = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])

    precision_cf_vad = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mrr_cf_vad = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_cf_vad = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_cf_vad = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])

    precision_cf_all = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mrr_cf_all = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_cf_all = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_cf_all = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])

    precision_cf_random = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mrr_cf_random = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_cf_random = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_cf_random = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])

    
    precision_cf_svd = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mrr_cf_svd = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_cf_svd = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_cf_svd = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])

    precision_cf_nmf = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mrr_cf_nmf = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_cf_nmf = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_cf_nmf = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    




    precision_cf_svd_emotions = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    mrr_cf_svd_emotions       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    recall_cf_svd_emotions    = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    f1s_cf_svd_emotions       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])

    precision_cf_svd_vad = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    mrr_cf_svd_vad       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    recall_cf_svd_vad    = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    f1s_cf_svd_vad       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])

    precision_cf_svd_all = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    mrr_cf_svd_all       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    recall_cf_svd_all    = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    f1s_cf_svd_all       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])

    precision_cf_nmf_emotions = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    mrr_cf_nmf_emotions       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    recall_cf_nmf_emotions    = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    f1s_cf_nmf_emotions       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])

    precision_cf_nmf_vad = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    mrr_cf_nmf_vad       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    recall_cf_nmf_vad    = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    f1s_cf_nmf_vad       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])

    precision_cf_nmf_all = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    mrr_cf_nmf_all       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    recall_cf_nmf_all    = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    f1s_cf_nmf_all       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])

    precision_demo_knn = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mrr_demo_knn = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_demo_knn = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_demo_knn = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])

    precision_demo_kmeans = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mrr_demo_kmeans = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_demo_kmeans = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_demo_kmeans = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])


    precision_demo_dbscan = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mrr_demo_dbscan = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_demo_dbscan = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_demo_dbscan = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])


    precision_demo_spec = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mrr_demo_spec = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_demo_spec = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_demo_spec = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])

    precision_demo_rforest = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mrr_demo_rforest = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_demo_rforest = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_demo_rforest = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])


    precision_cf_demo = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mrr_cf_demo = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_cf_demo = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_cf_demo = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])

    precision_cf_emo_demo = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    mrr_cf_emo_demo = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    recall_cf_emo_demo = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])
    f1s_cf_emo_demo = pd.DataFrame(0, index=range(1), columns=['top1', 'top2', 'top3', 'top4', 'top5', 'top6','top7', 'top8', 'top9','top10', 'top11', 'top12', 'top13', 'top14', 'top15', 'top16','top17', 'top18', 'top19','top20'])


    precision_cf_vad_demo = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    mrr_cf_vad_demo       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    recall_cf_vad_demo    = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    f1s_cf_vad_demo       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])

    precision_cf_all_demo = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    mrr_cf_all_demo       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    recall_cf_all_demo    = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])
    f1s_cf_all_demo       = pd.DataFrame(0, index=range(1), columns=['top1','top2','top3','top4','top5','top6','top7','top8','top9','top10','top11','top12','top13','top14','top15','top16','top17','top18','top19','top20'])


    # Parameters
    n_neighbors = 3       # for kNN demo
    n_clusters = 3        # for kMeans, Agglomerative, Spectral
    eps = 0.5             # for DBSCAN
    min_samples = 2       # for DBSCAN
    n_trees = 100         # for Random Forest




    k_folds = 5
    n = 1 # most similar users
    for k in range(k_folds):
        #train_data, test_data = split.split_user_data_balanced(ratings) # o original
        #train_data, test_data = split2.split_user_data_balanced(ratings) # o anterior
        train_data, test_data = split2.split_user_data_balanced(ratings, random_state=42+k)
        print('k = ', k)

        
        precision_likes, mrr_likes, recall_likes, f1s_likes = rs_cf_algorithms.CF_likes(train_data[['user', 'item', 'rating']], test_data[['user', 'item', 'rating']], n)
        precision_cf_likes += precision_likes
        mrr_cf_likes += mrr_likes
        recall_cf_likes += recall_likes
        f1s_cf_likes += f1s_likes

        
        precision_emotions, mrr_emotions, recall_emotions, f1s_emotions  = rs_cf_algorithms.CF_emotions(train_data[['user', 'item', 'rating','anger', 'fear', 'disgust', 'sadness', 'happiness', 'surprise', 'neutral']], test_data[['user', 'item', 'rating','anger', 'fear', 'disgust', 'sadness', 'happiness', 'surprise', 'neutral']], n)
        precision_cf_emotions += precision_emotions
        mrr_cf_emotions += mrr_emotions
        recall_cf_emotions += recall_emotions
        f1s_cf_emotions += f1s_emotions

        
        precision_vad, mrr_vad, recall_vad, f1s_vad  = rs_cf_algorithms.CF_vad(train_data[['user', 'item', 'rating','valence', 'arousal', 'dominance']], test_data[['user', 'item', 'rating','valence', 'arousal', 'dominance']], n)
        precision_cf_vad += precision_vad
        mrr_cf_vad += mrr_vad
        recall_cf_vad += recall_vad
        f1s_cf_vad += f1s_vad

        precision_all, mrr_all, recall_all, f1s_all  =rs_cf_algorithms.CF_all(train_data, test_data, n)
        precision_cf_all +=  precision_all
        mrr_cf_all +=  mrr_all
        recall_cf_all += recall_all
        f1s_cf_all += f1s_all
        


        
        precision_svd, mrr_svd, recall_svd, f1s_svd = rs_cf_algorithms.CF_SVD(train_data, test_data, n)
        precision_cf_svd +=  precision_svd
        mrr_cf_svd +=  mrr_svd
        recall_cf_svd += recall_svd
        f1s_cf_svd += f1s_svd

        
        precision_nmf, mrr_nmf, recall_nmf, f1s_nmf = rs_cf_algorithms.CF_NMF(train_data, test_data, n)
        precision_cf_nmf +=  precision_nmf
        mrr_cf_nmf +=  mrr_nmf
        recall_cf_nmf += recall_nmf
        f1s_cf_nmf += f1s_nmf


        #  SVD variants
        precision_svd_emo, mrr_svd_emo, recall_svd_emo, f1s_svd_emo = rs_cf_algorithms.CF_SVD_emotions(train_data, test_data, n)
        precision_cf_svd_emotions += precision_svd_emo
        mrr_cf_svd_emotions       += mrr_svd_emo
        recall_cf_svd_emotions    += recall_svd_emo
        f1s_cf_svd_emotions       += f1s_svd_emo

        precision_svd_vad, mrr_svd_vad, recall_svd_vad, f1s_svd_vad = rs_cf_algorithms.CF_SVD_vad(train_data, test_data, n)
        precision_cf_svd_vad += precision_svd_vad
        mrr_cf_svd_vad       += mrr_svd_vad
        recall_cf_svd_vad    += recall_svd_vad
        f1s_cf_svd_vad       += f1s_svd_vad

        precision_svd_all, mrr_svd_all, recall_svd_all, f1s_svd_all = rs_cf_algorithms.CF_SVD_all(train_data, test_data, n)
        precision_cf_svd_all += precision_svd_all
        mrr_cf_svd_all       += mrr_svd_all
        recall_cf_svd_all    += recall_svd_all
        f1s_cf_svd_all       += f1s_svd_all

        # NMF variants
        precision_nmf_emo, mrr_nmf_emo, recall_nmf_emo, f1s_nmf_emo = rs_cf_algorithms.CF_NMF_emotions(train_data, test_data, n)
        precision_cf_nmf_emotions += precision_nmf_emo
        mrr_cf_nmf_emotions       += mrr_nmf_emo
        recall_cf_nmf_emotions    += recall_nmf_emo
        f1s_cf_nmf_emotions       += f1s_nmf_emo

        precision_nmf_vad, mrr_nmf_vad, recall_nmf_vad, f1s_nmf_vad = rs_cf_algorithms.CF_NMF_vad(train_data, test_data, n)
        precision_cf_nmf_vad += precision_nmf_vad
        mrr_cf_nmf_vad       += mrr_nmf_vad
        recall_cf_nmf_vad    += recall_nmf_vad
        f1s_cf_nmf_vad       += f1s_nmf_vad

        precision_nmf_all, mrr_nmf_all, recall_nmf_all, f1s_nmf_all = rs_cf_algorithms.CF_NMF_all(train_data, test_data, n)
        precision_cf_nmf_all += precision_nmf_all
        mrr_cf_nmf_all       += mrr_nmf_all
        recall_cf_nmf_all    += recall_nmf_all
        f1s_cf_nmf_all       += f1s_nmf_all
        
             
        precision_demo_knn0, mrr_demo_knn0, recall_demo_knn0, f1s_demo_knn0 = rs_demographic_algorithms.DEMOGRAPHIC_knn( train_data, test_data, demographics, n)
        precision_demo_knn +=  precision_demo_knn0
        mrr_demo_knn +=  mrr_demo_knn0
        recall_demo_knn += recall_demo_knn0
        f1s_demo_knn += f1s_demo_knn0

        
        precision_demo_kmeans0, mrr_demo_kmeans0, recall_demo_kmeans0, f1s_demo_kmeans0 = rs_demographic_algorithms.DEMOGRAPHIC_kmeans( train_data, test_data, demographics, n)
        precision_demo_kmeans +=  precision_demo_kmeans0
        mrr_demo_kmeans +=  mrr_demo_kmeans0
        recall_demo_kmeans += recall_demo_kmeans0
        f1s_demo_kmeans += f1s_demo_kmeans0

        
        precision_demo_dbscan0, mrr_demo_dbscan0, recall_demo_dbscan0, f1s_demo_dbscan0 = rs_demographic_algorithms.DEMOGRAPHIC_dbscan  (train_data, test_data, demographics, eps=eps, min_samples=min_samples)  #( train_data, test_data, demographics, n)
        precision_demo_dbscan +=  precision_demo_dbscan0
        mrr_demo_dbscan +=  mrr_demo_dbscan0
        recall_demo_dbscan += recall_demo_dbscan0
        f1s_demo_dbscan += f1s_demo_dbscan0



        
        precision_demo_spec0, mrr_demo_spec0, recall_demo_spec0, f1s_demo_spec0 = rs_demographic_algorithms.DEMOGRAPHIC_spectral(train_data, test_data, demographics, n_clusters=n_clusters)
        precision_demo_spec +=  precision_demo_spec0
        mrr_demo_spec +=  mrr_demo_spec0
        recall_demo_spec += recall_demo_spec0
        f1s_demo_spec += f1s_demo_spec0

        
        precision_demo_rforest0, mrr_demo_rforest0, recall_demo_rforest0, f1s_demo_rforest0 = rs_demographic_algorithms.DEMOGRAPHIC_random_forest(train_data, test_data, demographics, n_estimators=n_trees)
        precision_demo_rforest +=  precision_demo_rforest0
        mrr_demo_rforest +=  mrr_demo_rforest0
        recall_demo_rforest += recall_demo_rforest0
        f1s_demo_rforest += f1s_demo_rforest0
        
        
        
        precision_random, mrr_random, recall_random, f1s_random  =rs_cf_algorithms.CF_random(train_data, test_data, n)
        precision_cf_random +=  precision_random
        mrr_cf_random +=  mrr_random
        recall_cf_random += recall_random
        f1s_cf_random += f1s_random
        
        
        
        




        precision_cf_demo0, mrr_cf_demo0, recall_cf_demo0, f1s_cf_demo0 = rs_cf_demo_algorithms.hybrid_recommendation(
        train_data,
        test_data,
        demographics,
        n_neighbors=1)

        precision_cf_demo +=  precision_cf_demo0
        mrr_cf_demo +=  mrr_cf_demo0
        recall_cf_demo += recall_cf_demo0
        f1s_cf_demo += f1s_cf_demo0 


        
        precision_cf_emo_demo0, mrr_cf_emo_demo0, recall_cf_emo_demo0, f1s_cf_emo_demo0 = rs_cf_demo_algorithms.hybrid_cf_emotions_demo(
        train_data,
        test_data,
        demographics,
        n_neighbors=1)

        precision_cf_emo_demo +=  precision_cf_emo_demo0
        mrr_cf_emo_demo +=  mrr_cf_emo_demo0
        recall_cf_emo_demo += recall_cf_emo_demo0
        f1s_cf_emo_demo += f1s_cf_emo_demo0 

        
        
        
        precision_cf_vad_demo0, mrr_cf_vad_demo0, recall_cf_vad_demo0, f1s_cf_vad_demo0 = rs_cf_demo_algorithms.hybrid_cf_vad_demo(train_data, test_data, demographics, n_neighbors=1)
        precision_cf_vad_demo += precision_cf_vad_demo0
        mrr_cf_vad_demo       += mrr_cf_vad_demo0
        recall_cf_vad_demo    += recall_cf_vad_demo0
        f1s_cf_vad_demo       += f1s_cf_vad_demo0

        precision_cf_all_demo0, mrr_cf_all_demo0, recall_cf_all_demo0, f1s_cf_all_demo0 = rs_cf_demo_algorithms.hybrid_cf_all_demo(train_data, test_data, demographics, n_neighbors=1)
        precision_cf_all_demo += precision_cf_all_demo0
        mrr_cf_all_demo       += mrr_cf_all_demo0
        recall_cf_all_demo    += recall_cf_all_demo0
        f1s_cf_all_demo       += f1s_cf_all_demo0

        
        






    print('### ---- ###')


   
    def tag(df, name):
        df = df.copy()
        df.insert(0, 'algorithm', name)
        return df
    




    final_precision_df = pd.concat([
        # CF

        tag(precision_cf_likes/k_folds,          'CF_likes'),
        tag(precision_cf_emotions/k_folds,        'CF_emotions'),
        tag(precision_cf_vad/k_folds,             'CF_vad'),
        tag(precision_cf_all/k_folds,             'CF_all'),
        tag(precision_cf_random/k_folds,          'CF_random'),
        # SVD
        tag(precision_cf_svd/k_folds,             'SVD_likes'),
        tag(precision_cf_svd_emotions/k_folds,    'SVD_emotions'),
        tag(precision_cf_svd_vad/k_folds,         'SVD_vad'),
        tag(precision_cf_svd_all/k_folds,         'SVD_all'),




        # NMF
        tag(precision_cf_nmf/k_folds,             'NMF_likes'),
        tag(precision_cf_nmf_emotions/k_folds,    'NMF_emotions'),
        tag(precision_cf_nmf_vad/k_folds,         'NMF_vad'),
        tag(precision_cf_nmf_all/k_folds,         'NMF_all'),
        # Demographic
        tag(precision_demo_knn/k_folds,           'Demo_KNN'),
        tag(precision_demo_kmeans/k_folds,        'Demo_KMeans'),
        tag(precision_demo_dbscan/k_folds,        'Demo_DBSCAN'),
        tag(precision_demo_spec/k_folds,          'Demo_Spectral'),
        tag(precision_demo_rforest/k_folds,       'Demo_RandomForest'),
        # Hybrid
        tag(precision_cf_demo/k_folds,            'Hybrid_CF_Demo'),
        tag(precision_cf_emo_demo/k_folds,        'Hybrid_CF_Emotions_Demo'),
        tag(precision_cf_vad_demo/k_folds,        'Hybrid_CF_VAD_Demo'),
        tag(precision_cf_all_demo/k_folds,        'Hybrid_CF_ALL_Demo'),
    ], ignore_index=True)
    print(final_precision_df)

    final_mrr_df = pd.concat([
        # CF
        tag(mrr_cf_likes/k_folds,                'CF_likes'),
        tag(mrr_cf_emotions/k_folds,              'CF_emotions'),
        tag(mrr_cf_vad/k_folds,                   'CF_vad'),
        tag(mrr_cf_all/k_folds,                   'CF_all'),
        tag(mrr_cf_random/k_folds,                'CF_random'),
        # SVD
        tag(mrr_cf_svd/k_folds,                   'SVD_likes'),
        tag(mrr_cf_svd_emotions/k_folds,          'SVD_emotions'),
        tag(mrr_cf_svd_vad/k_folds,               'SVD_vad'),
        tag(mrr_cf_svd_all/k_folds,               'SVD_all'),
        # NMF
        tag(mrr_cf_nmf/k_folds,                   'NMF_likes'),
        tag(mrr_cf_nmf_emotions/k_folds,          'NMF_emotions'),
        tag(mrr_cf_nmf_vad/k_folds,               'NMF_vad'),
        tag(mrr_cf_nmf_all/k_folds,               'NMF_all'),
        # Demographic
        tag(mrr_demo_knn/k_folds,                 'Demo_KNN'),
        tag(mrr_demo_kmeans/k_folds,              'Demo_KMeans'),
        tag(mrr_demo_dbscan/k_folds,              'Demo_DBSCAN'),
        tag(mrr_demo_spec/k_folds,                'Demo_Spectral'),
        tag(mrr_demo_rforest/k_folds,             'Demo_RandomForest'),
        # Hybrid
        tag(mrr_cf_demo/k_folds,                  'Hybrid_CF_Demo'),
        tag(mrr_cf_emo_demo/k_folds,              'Hybrid_CF_Emotions_Demo'),
        tag(mrr_cf_vad_demo/k_folds,              'Hybrid_CF_VAD_Demo'),
        tag(mrr_cf_all_demo/k_folds,              'Hybrid_CF_ALL_Demo'),
    ], ignore_index=True)
    print(final_mrr_df)

    final_recall_df = pd.concat([
        # CF
        tag(recall_cf_likes/k_folds,              'CF_likes'),
        tag(recall_cf_emotions/k_folds,           'CF_emotions'),
        tag(recall_cf_vad/k_folds,                'CF_vad'),
        tag(recall_cf_all/k_folds,                'CF_all'),
        tag(recall_cf_random/k_folds,             'CF_random'),
        # SVD
        tag(recall_cf_svd/k_folds,                'SVD_likes'),
        tag(recall_cf_svd_emotions/k_folds,       'SVD_emotions'),
        tag(recall_cf_svd_vad/k_folds,            'SVD_vad'),
        tag(recall_cf_svd_all/k_folds,            'SVD_all'),
        # NMF
        tag(recall_cf_nmf/k_folds,                'NMF_likes'),
        tag(recall_cf_nmf_emotions/k_folds,       'NMF_emotions'),
        tag(recall_cf_nmf_vad/k_folds,            'NMF_vad'),
        tag(recall_cf_nmf_all/k_folds,            'NMF_all'),
        # Demographic
        tag(recall_demo_knn/k_folds,              'Demo_KNN'),
        tag(recall_demo_kmeans/k_folds,           'Demo_KMeans'),
        tag(recall_demo_dbscan/k_folds,           'Demo_DBSCAN'),
        tag(recall_demo_spec/k_folds,             'Demo_Spectral'),
        tag(recall_demo_rforest/k_folds,          'Demo_RandomForest'),
        # Hybrid
        tag(recall_cf_demo/k_folds,               'Hybrid_CF_Demo'),
        tag(recall_cf_emo_demo/k_folds,           'Hybrid_CF_Emotions_Demo'),
        tag(recall_cf_vad_demo/k_folds,           'Hybrid_CF_VAD_Demo'),
        tag(recall_cf_all_demo/k_folds,           'Hybrid_CF_ALL_Demo'),
    ], ignore_index=True)
    print(final_recall_df)

    final_f1s_df = pd.concat([
        # CF
        tag(f1s_cf_likes/k_folds,                 'CF_likes'),
        tag(f1s_cf_emotions/k_folds,              'CF_emotions'),
        tag(f1s_cf_vad/k_folds,                   'CF_vad'),
        tag(f1s_cf_all/k_folds,                   'CF_all'),
        tag(f1s_cf_random/k_folds,                'CF_random'),
        # SVD
        tag(f1s_cf_svd/k_folds,                   'SVD_likes'),
        tag(f1s_cf_svd_emotions/k_folds,          'SVD_emotions'),
        tag(f1s_cf_svd_vad/k_folds,               'SVD_vad'),
        tag(f1s_cf_svd_all/k_folds,               'SVD_all'),
        # NMF
        tag(f1s_cf_nmf/k_folds,                   'NMF_likes'),
        tag(f1s_cf_nmf_emotions/k_folds,          'NMF_emotions'),
        tag(f1s_cf_nmf_vad/k_folds,               'NMF_vad'),
        tag(f1s_cf_nmf_all/k_folds,               'NMF_all'),
        # Demographic
        tag(f1s_demo_knn/k_folds,                 'Demo_KNN'),
        tag(f1s_demo_kmeans/k_folds,              'Demo_KMeans'),
        tag(f1s_demo_dbscan/k_folds,              'Demo_DBSCAN'),
        tag(f1s_demo_spec/k_folds,                'Demo_Spectral'),
        tag(f1s_demo_rforest/k_folds,             'Demo_RandomForest'),
        # Hybrid
        tag(f1s_cf_demo/k_folds,                  'Hybrid_CF_Demo'),
        tag(f1s_cf_emo_demo/k_folds,              'Hybrid_CF_Emotions_Demo'),
        tag(f1s_cf_vad_demo/k_folds,              'Hybrid_CF_VAD_Demo'),
        tag(f1s_cf_all_demo/k_folds,              'Hybrid_CF_ALL_Demo'),
    ], ignore_index=True)
    print(final_f1s_df)

    final_precision_df.to_csv('precision_all_n1_kfold5.csv', index=False)
    final_mrr_df.to_csv('mrr_all_n1_kfold5.csv', index=False)
    final_recall_df.to_csv('recall_all_n1_kfold5.csv', index=False)
    final_f1s_df.to_csv('f1s_all_n1_kfold5.csv', index=False)


    

if __name__ == "__main__":
    main()



