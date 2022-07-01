import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, v_measure_score, completeness_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def agglomerative_clustering_test(csv_name: str, sample_x: list[list], cluster_count: int = 3):
    '''
    Unsupervised Learning
    Clustering Hierarchical Aglomerating
    Recursively merges pair of clusters of sample data based on the euclidean distance
    '''

    dataset = pd.read_csv(f'{csv_name}')
    x = dataset.iloc[:, 2:].values
    y = dataset.iloc[:, 1].values.tolist()
    y=[int(l) for l in y]

    sample_x = pd.DataFrame(sample_x)
    x_sample = sample_x.iloc[:, 2:].values
    y_sample = (sample_x.iloc[:, 1].values).tolist()
    y_sample = [int(l) for l in y_sample]

    X_real=np.concatenate((x,x_sample))
    y_real=y+y_sample

    scaler = StandardScaler()
    X = scaler.fit_transform(X_real)

    aglomerative = AgglomerativeClustering(
        n_clusters=cluster_count, affinity='euclidean')
    y_pred = aglomerative.fit_predict(X)

    silhouette = silhouette_score(X, y_pred).round(2)

    ari = adjusted_rand_score(y_real, y_pred)
    homogenity = homogeneity_score(y_real, y_pred)
    completness = completeness_score(y_real,y_pred)
    vmeasure = v_measure_score(y_real, y_pred)

    results = dict()
    results['name'] = "Aglomerative"
    results['supervised'] = False
    results['score'] = silhouette
    results['ari'] = ari
    results['homogenity'] = homogenity
    results['completness'] = completness
    results['v_measure'] = vmeasure

    return y_real, y_pred, results, cluster_count
