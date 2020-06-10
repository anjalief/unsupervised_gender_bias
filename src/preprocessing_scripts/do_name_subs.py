import argparse
from collections import defaultdict, Counter
import pandas
import sys
import re

def load_subs(subs_file):
    word_to_sub = {}
    for line in open(subs_file).readlines():
        line = line.strip()
        words = line.split(":")
        word_to_sub[words[0]] = "<" + words[1] + ">"
    return word_to_sub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_file", help="File to replace blacklist words; we should only do this on training files")
    parser.add_argument("--subs_file", help="Formatted as word1:word2 where we replace word1 with word2")
    parser.add_argument("--output_file", help="place to write new file")
    args = parser.parse_args()

    # op_id   op_gender       post_id responder_id    response_text   op_name op_category
    df = pandas.read_csv(args.training_file, sep="\t")
    word_to_all_sub = load_subs(args.subs_file)

    def process_row(row):
        names = row["op_name"].lower().split()
        word_to_sub = {n:"<name>" for n in names}
        word_to_sub.update(word_to_all_sub)

        text = str(row["response_text"])
        text = text.lower()
        row["response_text"] = re.sub(r'\b(%s)\b' % '|'.join(word_to_sub.keys()), lambda m:word_to_sub.get(m.group(1), m.group(1)), text)
        return row

    new_rows = df.apply(process_row, axis=1)
    df = pandas.DataFrame(new_rows)
    df.to_csv(args.output_file, sep="\t", index=False)

if __name__ == "__main__":
    main()
