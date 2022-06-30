from spacy import load, matcher, util

nlp = load("es_core_news_sm")


def get_verbal_phrases(text):
    m = matcher.Matcher(nlp.vocab)
    pattern = [{'POS': 'VERB', 'OP': '?'},
               {'POS': 'ADV', 'OP': '*'},
               {'POS': 'AUX', 'OP': '*'},
               {'POS': 'VERB', 'OP': '+'}]
    m.add("verb-phrases", [pattern])
    matches = m(text)
    spans = [text[start:end] for _, start, end in matches]
    return len(util.filter_spans(spans))

# Meth used for the data processing


def get_noun_adj_phrases(text):
    m = matcher.Matcher(nlp.vocab)
    pattern = [{'POS': 'DET', 'OP': '?'},
               {'POS': 'NOUN', 'OP': '+'},
               {'POS': 'ADJ', 'OP': '+'}]
    m.add("adj-noun-phrases", [pattern])
    matches = m(text)
    spans = [text[start:end] for _, start, end in matches]
    return len(util.filter_spans(spans))

# Meth used for the data processing


def get_adj_noun_phrases(text):
    m = matcher.Matcher(nlp.vocab)
    pattern = [{'POS': 'DET', 'OP': '?'},
               {'POS': 'ADV', 'OP': '?'},
               {'POS': 'ADJ', 'OP': '+'},
               {'POS': 'NOUN', 'OP': '+'}]

    m.add("noun-adj-phrases", [pattern])
    matches = m(text)
    spans = [text[start:end] for _, start, end in matches]
    return len(util.filter_spans(spans))

# Meth used for the data processing


def get_noun_verb_phrases(text):
    m = matcher.Matcher(nlp.vocab)
    pattern = [{'POS': 'DET', 'OP': '?'},
               {'POS': 'NOUN', 'OP': '+'},
               {'POS': 'VERB', 'OP': '?'},
               {'POS': 'ADV', 'OP': '*'},
               {'POS': 'AUX', 'OP': '*'},
               {'POS': 'VERB', 'OP': '+'}]
    m.add("noun-verb-phrases", [pattern])
    matches = m(text)
    spans = [text[start:end] for _, start, end in matches]
    return len(util.filter_spans(spans))
