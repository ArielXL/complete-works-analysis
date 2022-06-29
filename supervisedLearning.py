from algorithms.naivebayes import naive_bayes
from dataVector.docVector import DocumentVector
from docProcess import documents_vectorized
from utils import csv_delete_n_last_row, csv_lines_count

'''
    Applies Naive-Bayes to calculate measures for the data start 
'''


def naive_bayes_practice(csv_name: str = 'vector/set.csv'):
    _, _, measures = naive_bayes(csv_name)

    print(f'\nMedidas')
    for x in measures:
        print(f'{x[0]}')
        print(f'{x[1]}\n')
    print()


'''
    Applies Naive-Bayes to the training data set 
'''


def using_training_data(csv_name: str = 'vector/set.csv'):

    start = csv_lines_count(csv_name)-1
    index = start

    '''
        Training with Marti 
    '''
    index,names1 = documents_vectorized('testData/marti', csv_name, index, 1)
    index,names2 = documents_vectorized('testData/otros', csv_name, index, 2)

    pos = index-start

    y_real, y_pred, measures = naive_bayes(csv_name, start)
    names=names1+names2
    csv_delete_n_last_row(csv_name, pos)
    
    return names,y_real,y_pred,measures

'''
    Applies Naive-Bayes to an input text
'''


def using_input(input_str: str, csv_name: str = 'vector/set.csv'):
    index = csv_lines_count(csv_name)-1

    doc = DocumentVector(document=input_str, id=index, writer=2)
    doc.append_vector(csv_name=csv_name)

    _, y_pred, measures= naive_bayes(csv_name, index)
    y_pred = y_pred[0]
    
    csv_delete_n_last_row(csv_name, 1)

    return y_pred,measures
