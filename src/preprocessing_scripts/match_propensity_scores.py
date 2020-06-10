import argparse
import pandas
import numpy as np
from scipy.spatial import distance_matrix
from collections import Counter

from munkres import Munkres

def get_attention_posts(score_file):
    df = pandas.read_csv(score_file, sep="\t")
    gender_header = df.columns[1]
    id_header = df.columns[3]
    score_header = df.columns[5] # take the "first val" column, not the "max val" column

    m_posts = df[df[gender_header] == "M"]
    m_posts = m_posts.reset_index(drop=True)
    w_posts = df[df[gender_header] == "W"]
    print("Count M posts:", len(m_posts), "Count F posts:", len(w_posts))

    return m_posts, w_posts, score_header, id_header

def get_bow_posts(score_file):
    #headings are op_id,op_gender,propensity_score
    df = pandas.read_csv(score_file)

    m_posts = df[df["op_gender"] == "M"]
    m_posts = m_posts.reset_index(drop=True)
    w_posts = df[df["op_gender"] == "W"]
    print("Count M posts:", len(m_posts), "Count F posts:", len(w_posts))

    return m_posts, w_posts, "propensity_score", "op_id"

def write_greedy_matches(m_posts, w_posts, score_header, id_header, outfile, max_match_dist):
    with open(outfile, "w") as out_fp:
        out_fp.write("w_post_id,m_post_id,w_score,m_score\n")
        for i,row in w_posts.iterrows():
            dists = np.absolute(m_posts[score_header].to_numpy() - row[score_header])
            idx = np.argmin(dists)
            if max_match_dist and dists[idx] > max_match_dist:
                continue

            out_fp.write("%s,%s,%s,%s\n" % (row[id_header], m_posts.iloc[idx][id_header], row[score_header], m_posts.iloc[idx][score_header]))

            m_posts = m_posts.drop(idx, axis=0).reset_index(drop=True)

def write_munkres_matches(m_posts, w_posts, score_header, id_header, outfile, max_match_dist):
    # first construct a "cost" matrix, which is distances between m and f posts
    m_matrix = m_posts[score_header].to_numpy()
    w_matrix = w_posts[score_header].to_numpy()
    m_matrix = m_matrix.reshape((m_matrix.shape[0], 1))
    w_matrix = w_matrix.reshape((w_matrix.shape[0], 1))
    print(m_matrix.shape)
    print(w_matrix.shape)

    cost_matrix = distance_matrix(m_matrix, w_matrix)

    print(cost_matrix.shape)
    m = Munkres()
    indexes = m.compute(cost_matrix)
    # print_matrix(matrix, msg='Lowest cost through this matrix:')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--propensity_score_file", help="output of propensity score generation")
    parser.add_argument("--response_file", help="response tsv, used to sort scores by most commented")
    parser.add_argument("--outfile", help="file to write matched_posts_to")
    parser.add_argument("--match_type", help="type of matching to run, can be greedy or munkres", choices=["greedy", "munkres"])
    parser.add_argument("--max_match_dist", type=float, help="if matches are further apart than this threshold, don't match")
    args = parser.parse_args()

    if "attention" in args.propensity_score_file:
        m_posts, w_posts, score_header, id_header = get_attention_posts(args.propensity_score_file)
    else:
        m_posts, w_posts, score_header, id_header = get_bow_posts(args.propensity_score_file)

    # Sort data by number of comments, so that we prefer posts with more replies
    if args.response_file:
        all_data = pandas.read_csv(args.response_file, sep="\t")
        id_to_count = Counter(all_data['op_id'])


        m_posts["num_comments"] = m_posts[id_header].apply(id_to_count.get)
        w_posts["num_comments"] = w_posts[id_header].apply(id_to_count.get)

        m_posts = m_posts.sort_values(by=['num_comments'], ascending=False)
        w_posts = w_posts.sort_values(by=['num_comments'], ascending=False)
    # Shuffle data
    else:
        m_posts = m_posts.sample(frac=1)
        w_posts = w_posts.sample(frac=1)

    if args.match_type == "greedy":
        write_greedy_matches(m_posts, w_posts, score_header, id_header, args.outfile, args.max_match_dist)
    else:
        write_munkres_matches(m_posts, w_posts, score_header, id_header, args.outfile, args.max_match_dist)

if __name__ == "__main__":
    main()
