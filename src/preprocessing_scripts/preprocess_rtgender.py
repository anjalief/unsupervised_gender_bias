# Reads in a csv file, tokenizes the specified column, and converts to tsv
import spacy
nlp = spacy.load('en_core_web_sm')

from spacy.lang.en import English
spacy_tokenizer = English().Defaults.create_tokenizer(nlp)

import pandas
import argparse
import os

from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_dir")
    parser.add_argument("--header_to_tokenize")
    args = parser.parse_args()

    def do_tok(text):
        return " ".join([str(t.text).strip() for t in spacy_tokenizer(str(text))])

    df = pandas.read_csv(args.input_file)
    df[args.header_to_tokenize] = df[args.header_to_tokenize].apply(do_tok)

    new_name = os.path.basename(args.input_file).replace(".csv", ".tok.tsv")
    df.to_csv(os.path.join(args.output_dir, new_name), sep="\t")


if __name__ == "__main__":
    main()
