from utils import *
from supervisedLearning import using_input as slui
from noSupervisedLearning import using_input as nslui

from supervisedLearning import using_training_data as slutd
from noSupervisedLearning import using_training_data as nslutd

'''
    Console Work
'''


def main():
    print('\nProyecto Final de Inteligencia Artificial.')
    print('Integrantes:\n  Thalia Blanco Figueras C512\n  Ariel Plasencia Diaz C512\n  Eziel Ramos Piñon C512\n')

    while True:
        while True:
            print('Calcular Similitud con Estilo Martiano:')
            data = int(input(
                '  1. Con Datos de Entrenamiento.\n  2. Con Texto Introducido.\n'))
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
            if alg == 1:
                names, y_real, y_pred, measures = slutd()
                print_results_with_info(names, y_real, y_pred, measures)

            elif alg == 2:
                names, y_real, y_pred, measures = nslutd()
                print_results_with_info(names, y_real, y_pred, measures)

            else:
                names1, y_real, y_pred1, measures1 = slutd()
                _, _, y_pred2, measures2 = nslutd()
                print_results_with_info(
                    names1, y_real, y_pred1, measures1, y_pred2, measures2)

        else:
            while True:
                text=input('Introducir Texto a Comparar:\n')
                if text:
                    print()
                    break

            if alg == 1:
                y_pred, measures = slui(text)
                print_result_with_input(y_pred, measures)

            elif alg == 2:
                y_pred, measures = nslui(text)
                print_result_with_input(y_pred, measures)

            else:
                y_pred1, measures1 = slui(text)
                y_pred2, measures2 = nslui(text)
                print_result_with_input(y_pred1, measures1, y_pred2, measures2)


if __name__ == "__main__":
    main()
