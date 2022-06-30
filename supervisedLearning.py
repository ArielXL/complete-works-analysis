from algorithms.naivebayes import naive_bayes
from docProcess import documents_vectorized
from utils import csv_delete_n_last_row, csv_lines_count


def naive_bayes_practice(csv_name: str = 'vector/set.csv'):
    '''
        Applies Naive-Bayes to calculate measures for the data start 
    '''
    _, _, measures = naive_bayes(csv_name)

    print(f'\nMedidas')
    for x in measures:
        print(f'{x[0]}')
        print(f'{x[1]}\n')
    print()


def using_training_data(csv_name: str = 'vector/set.csv'):
    '''
        Applies Naive-Bayes to the training data set 
    '''

    start = csv_lines_count(csv_name)-1
    index = start

    # Training with Marti
    index, names1 = documents_vectorized('testData/marti', csv_name, index, 1)
    index, names2 = documents_vectorized('testData/otros', csv_name, index, 2)

    pos = index-start

    y_real, y_pred, measures = naive_bayes(csv_name, start)
    names = names1+names2
    csv_delete_n_last_row(csv_name, pos)

    return names, y_real, y_pred, measures


def using_input(input_str: str, csv_name: str = 'vector/set.csv'):
    '''
        Applies Naive-Bayes to an input text
    '''

    start = csv_lines_count(csv_name)-1
    index, names = documents_vectorized(input_str, csv_name, start, 1)

    pos = index-start

    _, y_pred, measures = naive_bayes(csv_name, start)
    csv_delete_n_last_row(csv_name, pos)

    return names, None, y_pred, measures
