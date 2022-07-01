from utils import *
from algorithms.naivebayes import naive_bayes
from algorithms.kmeans import kmeans

'''
    Console Work
'''

def data_testing(csv_name: str = 'vector/set.csv'):

    index = csv_lines_count(csv_name)-1

    index, names1, vector1 = build_vectors_no_append(
        'testData/marti', 1, index)
    _, names2, vector2 = build_vectors_no_append(
        'testData/otros', 2, index)
    vectors = vector1+vector2
    names = names1+names2

    return vectors,names

def data_input(rute:str,csv_name: str = 'vector/set.csv'):
    
    index= csv_lines_count(csv_name)-1

    index, names, vectors = build_vectors_no_append(
        rute, 2, index)

    return vectors,names   


def main():
    print('\nProyecto Final de Inteligencia Artificial.')
    print('Integrantes:\n  Thalia Blanco Figueras C512.\n  Ariel Plasencia Diaz C512.\n  Eziel Ramos Piñon C512.\n')

    while True:
        while True:
            print('Calcular Similitud con Estilo Martiano:')
            data = int(input(
                '  1. Con Datos de Entrenamiento.\n  2. Con Ruta de Carpeta con Textos a Analizar.\n'))
            if data != 1 and data != 2:
                print('\n!!! ENTRADA INCORRECTA !!!\n')
            else:
                break

        while True:
            print('\nAlgoritmo de Aprendizaje a Emplear:')
            alg = int(input('  1. Naive Bayes.\n  2. KMEANS.\n  3. Ambos.\n'))
            if alg != 1 and alg != 2 and alg != 3:
                print('\nEntrada incorrecta\n')
            else:
                print()
                break

        if data == 1:

            vectors,names=data_testing()
            
            if alg == 1:
                y_real, y_pred, measures=naive_bayes(sample_x=vectors)
                print_results_with_info(names, y_real, y_pred, measures)

            elif alg == 2:
                y_real, y_pred, measures=kmeans(sample_x=vectors)
                print_results_with_info(names, y_real, y_pred, measures)

            else:
                y_real, y_pred1, measures1=naive_bayes(sample_x=vectors)
                _, y_pred2, measures2=kmeans(sample_x=vectors)
                print_results_with_info(names, y_real, y_pred1, measures1, y_pred2, measures2)

        else:
            while True:
                rute=input('Introducir Ruta de Textos a Analizar:\n')
                if rute:
                    print()
                    break

            vectors,names=data_input(rute=rute)

            if alg == 1:
                _, y_pred, measures=naive_bayes(sample_x=vectors)
                print_result_with_input(names,y_pred, measures)

            elif alg == 2:
                _, y_pred, measures=kmeans(sample_x=vectors)
                print_result_with_input(names,y_pred, measures)

            else:
                _, y_pred1, measures1=naive_bayes(sample_x=vectors)
                _, y_pred2, measures2=kmeans(sample_x=vectors)
                print_result_with_input(names,y_pred1, measures1, y_pred2, measures2)


if __name__ == "__main__":
    main()
