# Pull comments based on matched propensity scores
import argparse
import sys
sys.path.append("../analysis_scripts")
from basic_log_odds import write_log_odds
from collections import Counter
from nltk import tokenize
import pandas
from sklearn.model_selection import train_test_split
import os
from utils import short_filter


def write_df(df, matched_df, outfilename):
    final_df = pandas.DataFrame()
    num_m_length = 0
    num_w_length = 0

    for i,row in matched_df.iterrows():
        m_post_id = int(row['m_post_id'])
        w_post_id = int(row['w_post_id'])

        m_df = df[df['post_id'] == m_post_id]
        w_df = df[df['post_id'] == w_post_id]

        m_length = len(m_df)
        w_length = len(w_df)

        num_m_length += m_length
        num_w_length += w_length

        # if w_length > m_length:
        #     w_df = w_df.sample(m_length)
        # elif m_length > w_length:
        #     m_df = m_df.sample(w_length)

        final_df = final_df.append(w_df, ignore_index=True)
        final_df = final_df.append(m_df, ignore_index=True)

    final_df.to_csv(outfilename, sep="\t", index=False)
    print("Num M comments", outfilename, num_m_length)
    print("Num W comments", num_w_length)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_data", help="Should be tokenized version")
    parser.add_argument("--match_scores", help="output of matching algorithms, specifys OPs to keep")
    parser.add_argument("--suffix", help="appended to outfile names")
    parser.add_argument("--outdirname", help="place to write output")
    args = parser.parse_args()

    # w_post_id,m_post_id,w_score,m_score
    matched_df = pandas.read_csv(args.match_scores)

    # op_id,op_gender,post_id,responder_id,response_text,op_name,op_category
    if args.response_data.endswith(".tsv"):
        df = pandas.read_csv(args.response_data, sep="\t")
    else:
        df = pandas.read_csv(args.response_data)

    print("Before short removal", len(df))
    df = df[df["response_text"].apply(short_filter)]
    print("After short removal", len(df))

    write_df(df, matched_df, os.path.join(args.outdirname, "train." + args.suffix + ".txt"))
    # train, test_valid = train_test_split(matched_df, test_size=0.2)
    # test, valid = train_test_split(matched_df, test_size=0.5)

    # write_df(df, train, os.path.join(args.outdirname, "train." + args.suffix + ".txt"))
    # write_df(df, test, os.path.join(args.outdirname, "test." + args.suffix + ".txt"))
    # write_df(df, valid, os.path.join(args.outdirname, "valid." + args.suffix + ".txt"))

if __name__ == "__main__":
    main()
