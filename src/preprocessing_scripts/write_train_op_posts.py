# Write training op_posts in a format that can be used
# as input to the CNN/RNN classifiers
import argparse
import pandas
from utils import short_filter, do_tok

def write_op_posts(posts_csv_filename, id_filename, outfilename, skip_filter=False):
    op_posts = pandas.read_csv(posts_csv_filename)
    train_ids = open(id_filename).read().strip().split()

    train_op_posts = op_posts[op_posts['op_id'].isin(train_ids)]

    train_op_posts['post_text'] = train_op_posts['post_text'].apply(do_tok)
    if not skip_filter:
        train_op_posts = train_op_posts[train_op_posts["post_text"].apply(short_filter)]

    train_op_posts.to_csv(outfilename, index=False, header=False, sep="\t",
                          columns=["post_id", "post_text", "op_gender"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts_csv", help="file containing original posts")
    parser.add_argument("--train_ids", help="text file containing train_ids")
    parser.add_argument("--outfilename", help="file to write training data")
    parser.add_argument("--skip_filter", action='store_true', help="if specified, don't filter short posts (used for test data)")
    args = parser.parse_args()

    write_op_posts(args.posts_csv, args.train_ids, args.outfilename, args.skip_filter)


if __name__ == '__main__':
    main()
