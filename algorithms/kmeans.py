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
    if sample_x:
        # Determine the totals of cluster and writer documents
        clusters_ = [0]*cluster_count
        cluster_writer = [0]*cluster_count
        for index in range(len(y_pred)):
            c = y_pred[index]
            clusters_[c] = clusters_[c]+1

            if y[index] == writer:
                cluster_writer[c] = cluster_writer[c]+1

        sample_x = pd.DataFrame(sample_x)
        x_sample = sample_x.iloc[:, 2:].values
        y_sample = (sample_x.iloc[:, 1].values).tolist()
        y_sample = [int(l) for l in y_sample]

        y_predict=[]
        for i in range(len(y_sample)):
            temp_x=x_sample[i]
            temp = np.concatenate((x,[ temp_x]))
            sample_transf = scaler.fit_transform(temp).tolist()
            sample_transf = sample_transf[-1]
            y_temp = kmeans.predict([sample_transf]).tolist()
            y_predict.append(y_temp[0])
        
        y_new_pred = []
        # I will accept the articuls with 75% or more the marti
        for index in range(len(y_predict)):
            value = y_predict[index]
            probab = cluster_writer[value]/clusters_[value]
            if probab >= 0.75:
                y_new_pred.append(writer)
            else:
                y_new_pred.append(writer+1)

        silhouette = silhouette_score(X, y_pred).round(2)

        temp_y = y.tolist()+y_sample
        temp_y_pred = y_pred.tolist()+y_new_pred

        ari = adjusted_rand_score(temp_y, temp_y_pred)
        homogenity = homogeneity_score(temp_y, temp_y_pred)
        completness = completeness_score(temp_y, temp_y_pred)
        vmeasure = v_measure_score(temp_y, temp_y_pred)

        y=y_sample
        y_pred=y_new_pred

    else:
        y=y.tolist()
        silhouette = silhouette_score(X, y_pred).round(2)
        ari = adjusted_rand_score(y, y_pred)
        homogenity = homogeneity_score(y, y_pred)
        completness = completeness_score(y, y_pred)
        vmeasure = v_measure_score(y, y_pred)

    results = dict()
    results['name'] = "KMEANS"
    results['supervised'] = False
    results['score'] = silhouette
    results['ari'] = ari
    results['homogenity'] = homogenity
    results['completness'] = completness
    results['v_measure'] = vmeasure

    return y, y_pred, results, cluster_count


def kmeans_test(csv_name: str, sample_x: list[list]):
    '''
    Unsupervised Learning
    Clustering using Kmeans algorith
    Based on closest clouster elements
    '''

    dataset = pd.read_csv(f'{csv_name}')
    x = dataset.iloc[:, 2:].values
    y = dataset.iloc[:, 1].values.tolist()
    y = [int(l) for l in y]

    sample_x = pd.DataFrame(sample_x)
    x_sample = sample_x.iloc[:, 2:].values
    y_sample = (sample_x.iloc[:, 1].values).tolist()
    y_sample = [int(l) for l in y_sample]

    X_real=np.concatenate((x,x_sample))
    y_real=y+y_sample
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X_real)
    
    
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

    silhouette = silhouette_score(X, y_pred).round(2)

    ari = adjusted_rand_score(y_real, y_pred)
    homogenity = homogeneity_score(y_real, y_pred)
    completness = completeness_score(y_real,y_pred)
    vmeasure = v_measure_score(y_real, y_pred)

    results = dict()
    results['name'] = "KMEANS"
    results['supervised'] = False
    results['score'] = silhouette
    results['ari'] = ari
    results['homogenity'] = homogenity
    results['completness'] = completness
    results['v_measure'] = vmeasure

    return y_real, y_pred, results, cluster_count
