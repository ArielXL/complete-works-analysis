
from dataVector.utils import get_adj_noun_phrases, get_noun_adj_phrases, get_noun_verb_phrases, get_verbal_phrases


class SentenceVector:
    '''
    Convert a sentence into a vector
    The vector caracteristics are based on writing styles caracteristics
    '''

    def __init__(self, sentence) -> None:
        self.sentence = sentence

        # to set
        self.words = []
        self.words_count = 0
        self.words_length = 0
        self.words_small = 0
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
        for word in self.sentence:

            if word.pos_ != 'PUNCT':
                self.words.append(word.text)
                self.words_count += 1
                self.words_length += len(word.text)
                if len(word.text) <= 3:
                    self.words_small += 1

            if word.pos_ == 'PUNCT':
                self.punt_count += 1
            if word.is_stop:
                self.stopwords += 1
            if word.text == word.lemma_:
                self.roots_count += 1

        for _ in self.sentence.noun_chunks:
            self.noun_chunks_count += 1

        self.verbal_phrases = get_verbal_phrases(self.sentence)
        self.noun_adj_phrases = get_noun_adj_phrases(self.sentence)
        self.adj_noun_phrases = get_adj_noun_phrases(self.sentence)
        self.noun_verbal_phrases = get_noun_verb_phrases(self.sentence)
