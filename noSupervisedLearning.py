from algorithms.kmeans import kmeans, kmeans_test
from algorithms.agglomerative import agglomerative_clustering_test
from utils import csv_lines_count, print_no_supervised_metrics, build_vectors_no_append


def compare_no_supervised(csv_name: str = 'vector/set.csv'):
    '''
        Compare results KMEANS and Aglomerative Clustering
    '''

    print('Comparando metricas de Algoritmos de Aprendizaje No Supervisado:\n')

    index = csv_lines_count(csv_name)-1
    
    index, _, vector1 = build_vectors_no_append(
        'testData/marti', 1, index)
    _, _, vector2 = build_vectors_no_append(
        'testData/otros', 2, index)
    vectors = vector1+vector2

    y_r1, y_pred_1, r1, cluster_count = kmeans_test(csv_name=csv_name,sample_x=vectors)
    _, y_pred_2, r2, _ = agglomerative_clustering_test(csv_name=csv_name,cluster_count=cluster_count,sample_x=vectors)
    
    c_1=0
    c_2=0

    for i in range(len(y_r1)):
        if y_pred_1[i] == y_r1[i]:
            print(f'KMEANS calculo correctamente el documento {i+1}.')
            c_1+=1
        else:
            print(f'KMEANS no calculo correctamente el documento {i+1}.')

        if y_pred_2[i] == y_r1[i]:
            print(f'Cluster Aglomerativo calculo correctamente el documento {i+1}\n.')
            c_2+=1
        else:
            print(f'Cluster Aglomerativo no calculo correctamente el documento {i+1}\n.')
    
    c_1=(c_1/len(y_pred_1))*100
    c_2=(c_2/len(y_pred_1))*100
    
    print(f'Porciento Calculado correctamente por KMEANS: {c_1}.')
    print(f'Porciento Calculado correctamente por AGLOMERATIVO: {c_2}\n.')

    print_no_supervised_metrics(y_pred_1, r1)
    print_no_supervised_metrics(y_pred_2, r2)


def using_training_data(csv_name: str = 'vector/set.csv'):
    '''
        Applies KMEANS to the training data set 
    '''

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

def using_input(input_str: str, csv_name: str = 'vector/set.csv'):
    '''
        Applies KMEANS to an input text
    '''

    start = csv_lines_count(csv_name)-1
    index = start

    index, names, vector = build_vectors_no_append(
        input_str, 2, index)

    _, y_pred, measures, _ = kmeans(csv_name, vector, 1)

    return names, None, y_pred, measures
