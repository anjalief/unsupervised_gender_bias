# Given a tokenized (training) data set, extact log_odds features
# And add them as a new column. Then reformat data, dropping uneeded
# columns. If log_odds is set to false, just do the reformatting
import argparse
from collections import Counter, defaultdict
import sys
sys.path.append("../analysis_scripts")
from basic_log_odds import write_log_odds
import pandas
import math
import numpy as np
import csv

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Heading is         Unnamed: 0      op_id   op_gender       post_id responder_id    response_text   op_name op_category
def write_row(row, out_fp, log_odds):
  line = "%s\t%s\t%s\t%s\n" % (str(row["post_id"]), str(row["response_text"]), row["op_gender"], log_odds)
  out_fp.write(line)
  # print(row["post_id"])
  # df = row.to_frame()
  # df = df.rename(columns= {3: 'post_id', 5:'response_text', 2:'op_gender', 7:'log_ods'})
  # print(df)
  # print(df.columns)
  # df.to_csv(out_fp, index=False, header=False, sep="\t",
  #           #columns=[3, 5, 2, 7])
  #           columns=["post_id", "response_text", "op_gender", "log_odds"])


def write_csv(df, args):
    if args.odds_column:
        df.to_csv(args.output_file, index=False, header=False, sep="\t",
                  columns=["post_id", "response_text", "op_gender", "log_odds"])
    else:
        df.to_csv(args.output_file, index=False, header=False, sep="\t",
                  columns=["post_id", "response_text", "op_gender"])

def compute_filtered_word_scores(df, odds_column):
    header_to_count = Counter()
    prior = Counter()

    header_to_counter = defaultdict(Counter)
    for i, row in df.iterrows():
        words = str(row["response_text"]).split()
        prior.update(words)

        if row['op_gender'] == 'W':
          header = row[odds_column]
          header_to_counter[header].update(words)
          header_to_count[header] += 1

    print("Done loading raw counts", len(header_to_counter))

    word_to_scores = defaultdict(list)
    header_to_py = []
    for h1,c1 in header_to_counter.items():
        alt_counter = Counter()
        for h2,c2 in header_to_counter.items():
            if h2 != h1:
                alt_counter.update(c2)
        word_to_score = write_log_odds(c1, alt_counter, prior)
        for w,s in word_to_score.items():
            word_to_scores[w].append(sigmoid(s))
        header_to_py.append(header_to_count[h1] / len(df))

    # NOTE: we can get to here without memory issues
    word_to_scores = {w:np.array(s) for w,s in word_to_scores.items()}
    print("Done computing word scores", len(word_to_scores))
    return word_to_scores, header_to_py

# This scores w:[p(w | y1), p(w | y2),...]
def compute_word_scores(df, odds_column):
    header_to_count = Counter()

    header_to_counter = defaultdict(Counter)
    for i, row in df.iterrows():
        words = str(row["response_text"]).split()
        header = row[odds_column]
        header_to_counter[header].update(words)
        header_to_count[header] += 1

    print("Done loading raw counts")

    word_to_scores = defaultdict(list)
    header_to_py = []
    for h1,c1 in header_to_counter.items():
        alt_counter = Counter()
        prior = Counter()
        for h2,c2 in header_to_counter.items():
            prior.update(c2)
            if h2 != h1:
                alt_counter.update(c2)
        word_to_score = write_log_odds(c1, alt_counter, prior)
        for w,s in word_to_score.items():
            word_to_scores[w].append(sigmoid(s))
        header_to_py.append(header_to_count[h1] / len(df))

    # NOTE: we can get to here without memory issues
    print("Done computing word scores")
    word_to_scores = {w:np.array(s) for w,s in word_to_scores.items()}
    return word_to_scores, header_to_py

def score_and_write_rows(df, word_to_scores, header_to_py, args):
    out_fp = open(args.output_file, "w")
    csvwriter = csv.writer(out_fp, delimiter='\t')

    for i,row in df.iterrows():
        words = str(row["response_text"]).split()
        # When we down-filter, we might have words that we haven't seen
        # just skip them for now
        scores = [word_to_scores[w] for w in words if w in word_to_scores]
        pw = np.prod(scores, axis=0) # multiply over words
        final_scores = np.multiply(pw, header_to_py) # multiple by prior (elementwise)

        log_odds = " ".join([str(s) for s in final_scores])
        csvwriter.writerow([str(row["post_id"]), str(row["response_text"]), row["op_gender"], log_odds])
        # write_row(row, out_fp, log_odds)
    print("Done writing")
    out_fp.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--add_log_odds", action='store_true')
    parser.add_argument("--odds_column")
    parser.add_argument("--output_file")
    parser.add_argument("--filter_to_W", action='store_true', help="if specified, computed filtered scores")
    args = parser.parse_args()


    df = pandas.read_csv(args.input_file, sep="\t")

    if not args.odds_column:
        write_csv(df, args)
        return

    if args.filter_to_W:
      word_to_scores, header_to_py = compute_filtered_word_scores(df, args.odds_column)
    else:
      word_to_scores, header_to_py = compute_word_scores(df, args.odds_column)
    score_and_write_rows(df, word_to_scores, header_to_py, args)

if __name__ == "__main__":
    main()
