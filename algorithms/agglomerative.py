import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, v_measure_score, completeness_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


def agglomerative_clustering(csv_name, cluster_count=3):
    '''
    Unsupervised Learning
    Clustering Hierarchical Aglomerating
    Recursively merges pair of clusters of sample data based on the euclidean distance
    '''

    dataset = pd.read_csv(f'{csv_name}')
    x = dataset.iloc[:, 2:].values
    y = dataset.iloc[:, 1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(x)

    aglomerative = AgglomerativeClustering(
        n_clusters=cluster_count, affinity='euclidean')
    y_pred = aglomerative.fit_predict(X)

    silhouette = silhouette_score(X, y_pred).round(2)
    ari = adjusted_rand_score(y, y_pred)
    homogenity = homogeneity_score(y, y_pred)
    completness = completeness_score(y, y_pred)
    vmeasure = v_measure_score(y, y_pred)

    results = dict()
    results['name'] = "Aglomerative"
    results['supervised'] = False
    results['score'] = silhouette
    results['ari'] = ari
    results['homogenity'] = homogenity
    results['completness'] = completness
    results['v_measure'] = vmeasure

    return y.tolist(), y_pred, results, cluster_count
