# Write the top log-odds words and scores for original posts
import argparse
import sys
sys.path.append("../analysis_scripts")
from basic_log_odds import write_log_odds
from collections import Counter
from nltk import tokenize
import pandas
from statistics import stdev

def get_col_names(df):
    # return 'post_text', 'post_id', 'op_gender'  # Over original
    return df.columns[1], df.columns[0], df.columns[2] # Over training split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data")
    parser.add_argument("--outfile", help="file to write log odds scores")
    parser.add_argument("--match_scores", help="if specified, only write log-odds for the matched scores")
    parser.add_argument("--posts_outfile", help="if specified, write text of matched posts to outfile")
    parser.add_argument("--keep_all", action='store_true', help="if specified, use all posts in prior, even for non-matched scores")
    args = parser.parse_args()

    out_fp = None
    if args.posts_outfile:
        out_fp = open(args.posts_outfile, "w")

    # w_post_id,m_post_id,w_score,m_score
    w_posts, m_posts = None, None
    if args.match_scores:
        matched_df = pandas.read_csv(args.match_scores)
        w_posts = set(matched_df['w_post_id'])
        m_posts = set(matched_df['m_post_id'])

    # headings: op_id,op_gender,post_id,post_text,post_type
    if args.raw_data.endswith(".tsv") or args.raw_data.endswith(".txt"):
        df = pandas.read_csv(args.raw_data, sep="\t")
    else:
        df = pandas.read_csv(args.raw_data)
    m_counter = Counter()
    w_counter = Counter()
    prior = Counter()

    post_text, post_id, op_gender = get_col_names(df)

    # m_set = set()
    # w_set = set()
    for i,row in df.iterrows():
        tokens = tokenize.word_tokenize(str(row[post_text]))
        if args.keep_all:
            prior.update(tokens)

        if row[op_gender] == 'W':
            if w_posts is not None and not row[post_id] in w_posts:
                continue
            if out_fp is not None:
                out_fp.write("\t".join([str(row[post_id]), str(row[post_text]), row[op_gender]]))
                out_fp.write("\n")
            w_counter.update(tokens)
            # w_set.add(row['op_id'])
            if not args.keep_all:
                prior.update(tokens)

        if row[op_gender] == 'M':
            if m_posts is not None and not row[post_id] in m_posts:
                continue
            if out_fp is not None:
                out_fp.write("\t".join([str(row[post_id]), str(row[post_text]), row[op_gender]]))
                out_fp.write("\n")
            m_counter.update(tokens)
            # m_set.add(row['op_id'])
            if not args.keep_all:
                prior.update(tokens)

    delta = write_log_odds(m_counter, w_counter, prior)

    # delta = {x:delta[x] for x in delta if prior[x] > 10}
    delta_keys = sorted(delta, key=delta.get)
    # print("Number of W ops", len(w_set))
    # print("Number of M ops", len(m_set))

    if args.outfile:
        fp = open(args.outfile, "w")
        for w in delta_keys[:500]:
            fp.write("%s %s\n" % (w, delta[w]))
        fp.write("################################################################################\n")
        for w in delta_keys[-500:]:
            fp.write("%s %s\n" % (w, delta[w]))
        fp.close()

    d = ["women",
         "Obamacare",
         "Congresswoman",
         "Iran",
         "sexual",
         "EPA",
         "assault",
         "spending",
         "her",
         "Fox",
         "Women",
         "government"]
    d = ["you",
         "tonight",
         "her",
         "match",
         "Gracias",
         "he",
         "his",
         "president",
         "possible",
         "temps"]
    for w in d:
        if w in delta:
            print(w, delta[w])
        else:
            print("Skipping", w)

    vals = [s for x,s in delta.items()]
    pol = [abs(v) for v in vals]
    print("Average score", sum(pol) / len(vals))
    print("StdDev", stdev(pol))
    print("Min", min(vals))
    print("Max", max(vals))

if __name__ == "__main__":
    main()
