import csv
import os
from datetime import datetime
from dataVector.docVector import DocumentVector

def csv_delete_n_last_row(csv_name,count):
    with open(csv_name, 'r') as f:
        lines=f.readlines()
        lines=lines[:-count]
    f.close()    
    with open(csv_name, 'w', newline='') as f:
        writer=csv.writer(f)
        for line in lines:
            line=line[:-1]
            line=line.split(',')
            writer.writerow(line)
    f.close()

def csv_lines_count(csv_name):
    with open(csv_name, 'r') as f:
        lines=f.readlines()
        result=len(lines)
    f.close()
    return result

def build_vectors_no_append(data_dir:str,writer:int,index):
    vectors=[]
    names=[]
    
    print('Convirtiendo documentos en vectores....')
    for filename in os.listdir(data_dir):
        names.append(filename)
        file = os.path.join(data_dir, filename)
        now = datetime.now()
        current = now.strftime("%H:%M:%S")

        print(f"Iniciando conversion de documento {file} a las {current}")

        with open(file, encoding='utf-8') as f:
            text_file = f.read()
        row = DocumentVector(document=text_file, id=index, writer=writer).construct_vector()
        vectors.append(row)
        f.close()

        now = datetime.now()
        current = now.strftime("%H:%M:%S")

        print(f"Culminando conversion de documento {file} a las {current}")

        index += 1
    print()    

    return index,names,vectors

def print_no_supervised_metrics(y_pred:list,result:dict):
    name=result['name']
    score=result['score']
    ari=result['ari']
    homogenity=result['homogenity']
    completness=result['completness']
    v_measure=result['v_measure']

    print(f'Metricas Obtenidas de Aplicar {name}:')
    # print(f'Y Predecida: {y_pred}')
    
    '''
        mean intra-cluster distance a and the mean nearest-cluster distance b. (b - a) / max(a,b)
        A higher silhouette coefficient suggests better clusters
        for supervised uses score function acc
    '''
    print(f'Puntuacion Silhouette: {score}.')
    print('Otras Metricas:')

    '''
        measure the similarity between true and predicted labels.
        The ARI output values range between -1 and 1. 
        A score close to 0.0 indicates random assignments.
        A score close to 1 indicates perfectly labeled clusters.
    '''
    print(f'ARI: {ari}.')

    '''
        each cluster contains only members of a single class.
        0.0 is as bad as it can be, 1.0 is a perfect score.
    '''
    print(f'Homogenidad: {homogenity}.')

    '''
    all members of a given class are assigned to the same cluster.
    opposite to homogenity
    0.0 is as bad as it can be, 1.0 is a perfect score.
    '''
    print(f'Completitud: {completness}.')

    '''
        0.0 is as bad as it can be, 1.0 is a perfect score.
    '''
    print(f'V-Medida: {v_measure}.\n')

def print_supervised_metrics(y_pred:list,result:dict):
    name=result['name']
    print(f'Metricas Obtenidas de Aplicar {name}.')
    vac=result['score']
    ac=result['accurency']
    print(f'Promedio de Precision de los Datos: {vac}.')
    print(f'Precision de los Resultados: {ac}.\n')

def print_result_with_input(y_pred_1:list,measures_1:dict,y_pred_2:list|None=None,measures_2:dict|None=None):
    
    n_1=measures_1['name']
    if y_pred_1==1:
        print(f'Segun el Algoritmo {n_1} el Texto es de Marti.')
    else:
        print(f'Segun el Algoritmo {n_1} el Texto no es de Marti.')

    if y_pred_2:
        n_2=measures_2['name']
        if y_pred_2==1:
            print(f'Segun el Algoritmo {n_2} el Texto es de Marti.')
        else:
            print(f'Segun el Algoritmo {n_2} el Texto no es de Marti.')
    print()


def print_results_with_info(names:list,y_real:list,y_pred_1:list,measures_1:dict,y_pred_2:list|None=None,measures_2:dict|None=None):
    c_1=0
    c_2=0

    for i in range(len(y_real)):
        # Printing Writer
        o=names[i]
        if y_real[i]==1:
            print(f'La obra {o} es de Marti.')
        else:
            print(f'La obra {o} no es de Marti.')
        
        n_1=measures_1['name']
        # Printing result for 1 algorithm
        if y_real[i]==y_pred_1[i]:

            print(f'El Algoritmo {n_1} lo Calculo Correctamente.')
            c_1+=1
        else:
            print(f'El Algoritmo {n_1} no lo Calculo Correctamente.')

        if y_pred_2:
            n_2=measures_2['name']
            # Printing result for 2 algorithm
            if y_real[i]==y_pred_2[i]:
                c_2+=1
                print(f'El Algoritmo {n_2} lo Calculo Correctamente.')
            else:
                print(f'El Algoritmo {n_2} no lo Calculo Correctamente.')    

        print()

    print(f'Grado de Precision al Aplicar {n_1}: {(c_1/len(y_real))*100}%.')
    if y_pred_2:
        print(f'Grado de Precision al Aplicar {n_2}: {(c_2/len(y_real))*100}%.\n')

    if measures_1['supervised']:
        print_supervised_metrics(y_pred_1,measures_1)
    else:
        print_no_supervised_metrics(y_pred_1,measures_1)

    if y_pred_2:
        if measures_2['supervised']:
            print_supervised_metrics(y_pred_2,measures_2)
        else:
            print_no_supervised_metrics(y_pred_2,measures_2)

    print()        