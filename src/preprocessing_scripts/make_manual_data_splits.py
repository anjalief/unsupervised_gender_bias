# Read in op_ids from files and make data
# splits accordingly
# replaces both make_clean_data_splits and write_train_op_posts.py
import argparse
import pandas
import os
from write_train_op_posts import write_op_posts
from utils import short_filter

def write_response_posts(id_filename, outfilename, response_df):
    ids = open(id_filename).read().strip().split()

    to_write = response_df[response_df['op_id'].isin(ids)]
    to_write = to_write[to_write["response_text"].apply(short_filter)]

    to_write.to_csv(outfilename, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts_csv", help="file containing original posts")
    parser.add_argument("--response_tsv", help="tokenized file containing responses")
    parser.add_argument("--train_ids", help="text file containing train_ids")
    parser.add_argument("--test_ids", help="text file containing train_ids")
    parser.add_argument("--valid_ids", help="text file containing train_ids")
    parser.add_argument("--test_suffix", help="seperate different test files", default="")
    parser.add_argument("--outdirname", help="directory to write data")
    args = parser.parse_args()

    if args.posts_csv:
        write_op_posts(args.posts_csv, args.train_ids, os.path.join(args.outdirname, "train_op_posts.tsv"), skip_filter=False)
        write_op_posts(args.posts_csv, args.valid_ids, os.path.join(args.outdirname, "valid_op_posts.tsv"), skip_filter=False)
        write_op_posts(args.posts_csv, args.test_ids, os.path.join(args.outdirname, "test_op_posts.tsv"), skip_filter=True)

    response_df = pandas.read_csv(args.response_tsv, sep="\t")

    if args.train_ids:
        write_response_posts(args.train_ids, os.path.join(args.outdirname, "train.txt"), response_df)
    if args.test_ids:
        write_response_posts(args.test_ids, os.path.join(args.outdirname, "test" + args.test_suffix + ".txt"), response_df)
    if args.valid_ids:
        write_response_posts(args.valid_ids, os.path.join(args.outdirname, "valid.txt"), response_df)

if __name__ == "__main__":
    main()
