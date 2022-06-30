import csv
from collections import Counter
from dataVector.sentVector import SentenceVector
from dataVector.utils import nlp


class DocumentVector:
    '''
    Convert a document into a vector
    The vector caracteristics are based on writing styles caracteristics
    '''

    def __init__(self, document: str, id: int, writer: int) -> None:
        self.id = id
        self.document = nlp(document)
        self.writer = writer
        self.sentences_count = 0

        # to set
        self.words_count = 0
        self.words_length = 0
        self.words_diff = Counter()
        self.word_small = 0
        self.punt_count = 0
        self.stopwords = 0
        self.roots_count = 0
        self.noun_chunks_count = 0
        self.verbal_phrases = 0
        self.noun_adj_phrases = 0
        self.adj_noun_phrases = 0
        self.noun_verbal_phrases = 0

        self.calculate_values()

    def calculate_values(self) -> None:
        for sentence in self.document.sents:
            self.sentences_count += 1
            s = SentenceVector(sentence)

            # to build set
            self.words_count += s.words_count
            self.words_length += s.words_length
            self.word_small += s.words_small
            self.words_diff.update(s.words)

            self.punt_count += s.punt_count
            self.stopwords += s.stopwords
            self.roots_count += s.roots_count

            self.noun_chunks_count += s.noun_chunks_count
            self.verbal_phrases += s.verbal_phrases
            self.noun_adj_phrases += s.noun_adj_phrases
            self.adj_noun_phrases += s.adj_noun_phrases
            self.noun_verbal_phrases += s.noun_verbal_phrases

    def construct_vector(self) -> list:
        '''
            mean of words x sentence
            mean of words lenfth x words
            small words x words count
            different words x words count
        '''
        m_count = self.words_count/self.sentences_count
        m_length = self.words_length/self.words_count
        m_small = self.word_small/self.words_count
        m_dif = len(self.words_diff)/self.words_count
        rowf1 = [f'{self.id}', f'{self.writer}', f'{m_count}',
                 f'{m_length}', f'{m_small}', f'{m_dif}']

        '''
            puct frec normalized by total words
            stopwords frec normalized by total words
            roots frec normalized by total words
        '''
        m_punct = self.punt_count/self.words_count
        m_stopw = self.stopwords/self.words_count
        m_roots = self.roots_count/self.words_count
        rowf2 = rowf1+[f'{m_punct}', f'{m_stopw}', f'{m_roots}']

        '''
            noun_chuncks frec normalized by total words
            verbal phrases frec normalized by total words
            noun+adj frec normalized by total words
            adj+noun frec normalized by total words
            verb+noun frec normalized by total words
        '''
        m1 = self.noun_chunks_count/self.sentences_count
        m2 = self.verbal_phrases/self.sentences_count
        m3 = self.noun_adj_phrases/self.sentences_count
        m4 = self.adj_noun_phrases/self.sentences_count
        m5 = self.noun_verbal_phrases/self.sentences_count

        return rowf2 + [f'{m1}', f'{m2}', f'{m3}', f'{m4}', f'{m5}']

    def append_vector(self, csv_name):
        row3 = self.construct_vector()
        with open(csv_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row3)
        f.close()
