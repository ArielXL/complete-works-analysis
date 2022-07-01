import os
import csv
from datetime import datetime
from dataVector.docVector import DocumentVector





def building_vectors(data_dir: str, csv_name: str = 'vector/set.csv', my_writer: str = 1):
    ''' 
        Converting all data information into a vector

        Starting the csv file, Columns:
        ID, Writer(Int), Words Count,Words Length,Words Small,Words Different,Punctuation Marks
        Stopwords,Root Words,Nouns Phrases,Verbal Phrases,Noun Adj Phrases,Adj Noun Phrases
        Noun Verb Phrases
    '''

    def documents_vectorized(data_dir: str, csv_name: str, index: int, writer: int):
        '''    
        Build for every document in data_dir the vector
        '''
        names = []
        print('Convirtiendo documentos en vectores....')
        for filename in os.listdir(data_dir):
            names.append(filename)
            file = os.path.join(data_dir, filename)
            now = datetime.now()
            current = now.strftime("%H:%M:%S")
    
            print(f"Iniciando conversion de documento {file} a las {current}")
    
            with open(file, encoding='utf-8') as f:
                text_file = f.read()
            doc = DocumentVector(document=text_file, id=index, writer=writer)
            doc.append_vector(csv_name=csv_name)
            f.close()
    
            now = datetime.now()
            current = now.strftime("%H:%M:%S")
    
            print(f"Culminando conversion de documento {file} a las {current}")
    
            index += 1
    
        print()
    
        return index, names

    set_f1 = ['ID', 'Author', 'Words Count', 'Words Length',
              'Words Small', 'Words Different']

    set_f2 = set_f1+['Punctuation Marks', 'Stopwords', 'Root Words']

    set_f3 = set_f2+['Nouns Phrases', 'Verbal Phrases',
                     'Noun Adj Phrases', 'Adj Noun Phrases', 'Noun Verb Phrases']

    with open(csv_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(set_f3)
    f.close()

    index = 0
    for folder in os.listdir(data_dir):
        writ = 2
        folder = os.path.join(data_dir, folder)
        now = datetime.now()
        current = now.strftime("%H:%M:%S")

        print(f"Attending folder {folder} at {current}")

        if my_writer == folder:
            writ = 1
        index, _ = documents_vectorized(folder, csv_name, index, writ)

        now = datetime.now()
        current = now.strftime("%H:%M:%S")
        print(f"Finishing doc {folder} at {current}")


if __name__ == "__main__":
    building_vectors('data')
