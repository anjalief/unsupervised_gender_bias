import pandas
import argparse
from collections import Counter
from find_pred_switch import get_english_lemma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", help="that contains text to analyze")
    parser.add_argument("--bias_lexicon", help="if specified, print out scores for words in lexicon")
    args = parser.parse_args()

    text_count = Counter(open(args.text_file).read().split())
    word_count = sum([s for i,s in text_count.items()])

    df = pandas.read_csv(args.bias_lexicon)

    for col in df.columns:
        lemmas = df[col].apply(get_english_lemma)
        lemmas = [w for w in lemmas if w != "nan"]
        if col =='arrogant':
            for w in lemmas:
                if w == "nan":
                    continue
                print(w, text_count[w])
        scores = sum([text_count[w] for w in lemmas])
        print(col, scores, scores / word_count)


if __name__ == "__main__":
    main()
