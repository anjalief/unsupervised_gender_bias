import spacy
nlp = spacy.load('en_core_web_sm')

from spacy.lang.en import English
spacy_tokenizer = English().Defaults.create_tokenizer(nlp)


def short_filter(text):
    return len(str(text).split()) > 4

def do_tok(text):
    return " ".join([str(t.text).strip() for t in spacy_tokenizer(str(text))])
