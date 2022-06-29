from algorithms.kmeansClustering import kmeans
from algorithms.aglomerativeClustering import aglomerative_clustering
from dataVector.docVector import DocumentVector
from utils import csv_lines_count, print_no_supervised_metrics, build_vectors_no_append

'''
    Compare results KMEANS and Aglomerative Clustering
'''


def compare_no_supervised(csv_name: str = 'vector/set.csv'):
    print('Comparando metricas de Algoritmos de Aprendizaje No Supervisado:\n')
    _, y_pred_1, r1, cluster_count = kmeans(csv_name)
    _, y_pred_2, r2, _ = aglomerative_clustering(csv_name, cluster_count)
    print_no_supervised_metrics(y_pred_1, r1)
    print_no_supervised_metrics(y_pred_2, r2)


'''
    Applies KMEANS to the training data set 
'''


def using_training_data(csv_name: str = 'vector/set.csv'):
    start = csv_lines_count(csv_name)-1
    index = start

    index, names1, vector1 = build_vectors_no_append(
        'testData/marti', 1, index)
    _, names2, vector2 = build_vectors_no_append(
        'testData/otros', 2, index)
    vectors = vector1+vector2
    names = names1+names2

    y_real, y_pred, measures, _ = kmeans(csv_name, vectors, 1)

    return names, y_real, y_pred, measures


'''
    Applies KMEANS to an input text
'''


def using_input(input_str: str, csv_name: str = 'vector/set.csv'):
    index = csv_lines_count(csv_name)-1
    vector = DocumentVector(document=input_str, id=index,
                            writer=2).construct_vector()

    _, y_pred, measures, _ = kmeans(csv_name, [vector], 1)

    return y_pred, measures
