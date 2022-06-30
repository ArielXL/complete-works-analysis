import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, v_measure_score, completeness_score
from sklearn.preprocessing import StandardScaler


def kmeans(csv_name: str, sample_x: list[list] | None = None, writer: int = 1):
    '''
    Unsupervised Learning
    Clustering using Kmeans algorith
    Based on closest clouster elements
    '''

    dataset = pd.read_csv(f'{csv_name}')
    x = dataset.iloc[:, 2:].values
    y = dataset.iloc[:, 1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(x)

    kmeans_start = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    # Selecting the best clousters count
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_start)
        kmeans.fit(X)
        # Sum of squared distances of samples to their closest cluster center, weighted by the sample weights
        sse.append(kmeans.inertia_)

    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    cluster_count = kl.elbow

    # Fitting K-Means to the dataset
    kmeans = KMeans(n_clusters=cluster_count,
                    init='k-means++', random_state=42)
    y_pred = kmeans.fit_predict(X)

    # Getting the clousters
    if(sample_x):
        # Determine the totals of cluster and writer documents
        clusters_ = [0]*cluster_count
        cluster_writer = [0]*cluster_count
        for index in range(len(y_pred)):
            c = y_pred[index]
            clusters_[c] = clusters_[c]+1

            if y[index] == writer:
                cluster_writer[c] = cluster_writer[c]+1

        sample_x=pd.DataFrame(sample_x)
        x_sample = sample_x.iloc[:, 2:].values
        y_sample = (sample_x.iloc[:, 1].values).tolist()
        y_sample=[int(x) for x in y_sample]
        

        all_v=np.concatenate((x,x_sample))
        v=len(all_v)-len(x_sample)

        sample_transf=scaler.fit_transform(all_v).tolist()
        sample_transf = sample_transf[v:]
        y_predict = kmeans.predict(sample_transf)

        y_new_pred = []
        # I will accept the articuls with 75% or more the marti
        for index in range(len(y_predict)):
            value = y_predict[index]
            probab = cluster_writer[value]/clusters_[value]
            if probab >= value:
                y_new_pred.append(writer)
            else:
                y_new_pred.append(writer+1)

        y_reals=y.tolist()+y_sample
        y_p = y_pred.tolist()+y_new_pred
        ari = adjusted_rand_score(y_reals, y_p)
        homogenity = homogeneity_score(y_reals, y_p)
        completness = completeness_score(y_reals, y_p)
        vmeasure = v_measure_score(y_reals, y_p)

    else:
        y_new_pred = None
        y_sample = None
        ari = adjusted_rand_score(y, kmeans.labels_)
        homogenity = homogeneity_score(y, kmeans.labels_)
        completness = completeness_score(y, kmeans.labels_)
        vmeasure = v_measure_score(y, kmeans.labels_)

    silhouette = silhouette_score(X, kmeans.labels_).round(2)
    results = dict()
    results['name'] = "KMEANS"
    results['supervised'] = False
    results['score'] = silhouette
    results['ari'] = ari
    results['homogenity'] = homogenity
    results['completness'] = completness
    results['v_measure'] = vmeasure

    return y_sample, y_new_pred, results, cluster_count
