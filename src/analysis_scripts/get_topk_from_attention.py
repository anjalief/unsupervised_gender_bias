from collections import defaultdict

import pickle
import sys
import argparse
import os
import numpy as np
import math
import pandas
from get_gender_ids import get_op_gender_ids
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter

SKIP=set(["?", ".", ":", ","])

def getTopK(word2attention, k):
    retpair = sorted(word2attention.items(), key=lambda x:x[1], reverse=True)[:k]
    retwords = [x for (x,y) in retpair]
    return retwords

def process_attns(wordattns):
    attns = []
    words = []

    for wordattn in wordattns:
        if len(wordattn.split(":")) > 2:
            continue

        [word, attn] = wordattn.split(":")
        attns.append(float(attn))
        words.append(word)

    # I think this happens if the post only has poorly formatted words with : in them (e.g. URLs)
    if len(attns) == 0:
        return [], []

    exps = [np.exp(i) for i in attns]
    sum_exp = sum(exps)
    exps = [e / sum_exp for e in exps]
    attns = [e * len(exps) for e in exps]
    # attns = np.array(attns)
    # mean = np.mean(attns)
    # std = np.std(attns)
    # attns = (attns-mean)/std
    return words, attns

# for now skip the beginning, just take bigram
def process_bigrams(wordattns):
    attns = []
    words = []

    for i,wordattn1 in enumerate(wordattns):
        if i == len(wordattns) - 1:
            continue

        wordattn2 = wordattns[i+1]

        if len(wordattn1.split(":")) > 2 or len(wordattn2.split(":")) > 2:
            continue

        [word1, attn1] = wordattn1.split(":")
        [word2, attn2] = wordattn2.split(":")
        if word1 in SKIP or word2 in SKIP:
            continue

        attns.append(float(attn1) + float(attn2))
        words.append(word1 + "_" + word2)

    # I think this happens if the post only has poorly formatted words with : in them (e.g. URLs)
    if len(attns) == 0:
        return [], []

    attns = np.array(attns)
    mean = np.mean(attns)
    std = np.std(attns)
    attns = (attns-mean)/std
    return words, attns

# For each post, compute the normalized attn weight for
# each word in the post
# Then, sum over the weights for each word across all posts to get a final
# score for each word
def all_atn_words(args, aggregate=sum, verbose=True):
    if verbose:
        print("Printing highest attended to words overall using", str(aggregate))

    y_true = []
    y_pred = []
    word2attentions = defaultdict(list)
    f = open(args.attention_file)
    c, C = 0.0, 0.0
    for l in f:
        items = l.strip().split("\t")
        attentions, label, predicted = items[0], items[1], items[2]
        wordattns = attentions.split()
        C += 1
        y_true.append(label)
        y_pred.append(predicted)
        if label == predicted:
            c+=1
        attns = []
        words = []

        words, attns = process_attns(wordattns)
        for word, attn in zip(words, attns):
            word2attentions[word].append(attn)

    word2attention = defaultdict(float)

    for w, attns in word2attentions.items():
        if len(attns > 50):
            word2attention[w] = aggregate(attns)

    f.close()

    if verbose:
        print ("Accuracy out of", c/C, C)
        print ("F1 score", f1_score(y_true, y_pred))
        print (", ".join(getTopK(word2attention, args.topk)))
        print ("")
    else:
        for k in getTopK(word2attention, args.topk):
            print ("Accuracy out of", c/C, C)
            print ("F1 score", f1_score(y_true, y_pred))
            print ("Macro F1 score", f1_score(y_true, y_pred, average='macro'))
            print(k)

# Print attn words for each label instead of accross entire corpus
# Optionally only include posts that were labeled correctly
def attn_words_by_label(attention_file, labels, aggregate=sum, ids = None, outdir = None, correct_only = False, bigrams = False, topk = 50):
    label2word2attentions = {}
    label2int = {}
    for i,l in enumerate(labels):
        label2word2attentions[l] = defaultdict(list)
        label2int[l] = i
    print(label2int)

    f = open(attention_file)

    c, C = 0.0, 0.0
    y_true, y_pred = [], []
    for l in f:
        items = l.strip().split("\t")
        attentions, label, predicted, post_id = items[0], items[1], items[2], items[3]
        if ids is not None and not int(post_id) in ids:
            continue

        wordattns = attentions.split()

        C += 1
        y_true.append(label2int[label])
        y_pred.append(label2int[predicted])
        if label == predicted:
            c += 1
        else:
            if correct_only:
                continue

        if bigrams:

            words, attns = process_bigrams(wordattns)
        else:
            words, attns = process_attns(wordattns)
        for word, attn in zip(words, attns):
            label2word2attentions[predicted][word].append(attn)

    label2word2attention = {}
    for l in labels:
        label2word2attention[l] = defaultdict(float)

    for l, word2attentions in label2word2attentions.items():
        for w, attns in word2attentions.items():
            if len(attns) > 100:
                label2word2attention[l][w] = aggregate(attns)

    f.close()

    print ("Accuracy out of", c/C, C)
    print ("F1 score", f1_score(y_true, y_pred))
    print ("Precision", precision_score(y_true, y_pred))
    print ("Recall", recall_score(y_true, y_pred))
    print ("Macro F1 score", f1_score(y_true, y_pred, average='macro'))
    return

    if outdir:
        if "sum" in str(aggregate):
            a_type = "sum"
        else:
            a_type = "mean"
        if "baseline" in attention_file:
            m_type = "baseline"
        else:
            m_type = "sk"
        for l in labels:
            print(l)
            fp = open(os.path.join(outdir, attention_file.replace(".txt", "") + "attn_scores_" + a_type + "_" + m_type + "_" + l + ".txt"), "w")
            for k in getTopK(label2word2attention[l], topk):
                fp.write(k)
                fp.write("\n")
            fp.close()
    else:
        retVal = {}
        for l in labels:
            retVal[l] = set([k for k in getTopK(label2word2attention[l], topk)])
        return retVal

def freq_attn_words(attention_file, labels, ids = None, outdir = None, correct_only = False, topk = 50, confident_only = False):
    label2word2attentions = {}
    label2int = {}
    for i,l in enumerate(labels):
        label2word2attentions[l] = Counter()
        label2int[l] = i

    f = open(attention_file)

    c, C = 0.0, 0.0
    y_true, y_pred = [], []
    for l in f:
        items = l.strip().split("\t")
        attentions, label, predicted, post_id, score = items[0], items[1], items[2], items[3], float(items[4])
        if ids is not None and not int(post_id) in ids:
            continue

        wordattns = attentions.split()

        C += 1
        y_true.append(label2int[label])
        y_pred.append(label2int[predicted])
        if label == predicted:
            c += 1
        else:
            if correct_only:
                continue

        if confident_only and score < 0.9:
            continue

        words, attns = process_attns(wordattns)
        num_keep = int(len(attns)/4)
        word_to_attn = {}
        for word, attn in zip(words, attns):
            word_to_attn[word] = attn
        keep_attn_words = getTopK(word_to_attn, num_keep)
        for x in keep_attn_words:
            label2word2attentions[predicted][word] += 1
    f.close()

    print ("Accuracy out of", c/C, C)
    print ("F1 score", f1_score(y_true, y_pred))

    if outdir:
        for l in labels:
            print(l)
            fp = open(os.path.join(outdir, attention_file.replace(".txt", "") + "attn_scores_freq_" + l + ".txt"), "w")
            for k in label2word2attentions[l].most_common(topk):
                fp.write(str(k) + "\n")
            fp.close()

def main():
    parser = argparse.ArgumentParser(description='Remove topical words')
    parser.add_argument('--attention_file', type=str, required=True,
                        help='file containing the word:attention data')
    parser.add_argument('--topk', type=int, default=100,
                        help='file containing the label names')
    # Control the format of the output
    parser.add_argument('--aggregate', type=str, choices = ["sum", "mean", "freq"], default="sum",
                        required=True,
                        help='how to aggregate attention scores accross posts')
    parser.add_argument('--bigrams', action='store_true')

    #  These parameters all control ways to break the data into more interesting subsets
    parser.add_argument('--correct_only', action='store_true', help="Only include samples that were classified correctly")
    parser.add_argument('--confident_only', action='store_true', help="Only include samples that were classified with high confidence")
    parser.add_argument('--subreddit_filter' ,type=str, help="If specified, only include posts from this subreddit. Requires subreddit_meta")
    parser.add_argument('--subreddit_meta' ,type=str, help="Meta-file for posts, for when we need more info for filtering") # need this to apply subreddit filter
    parser.add_argument('--original_posts' ,type=str, help="If specified, only include posts if they are responses to a post that we can easily infer op_gender from")
    parser.add_argument('--micro_posts' ,type=str, help="If specified, only include posts if they were annotatated as offensive. Requires micro_processed")
    parser.add_argument('--micro_processed' ,type=str, help="Need this file to map int ids in the attention file to post_ids in the micro_posts file")
    args = parser.parse_args()

    ids = None
    if args.subreddit_filter and args.subreddit_meta:
        df = pandas.read_csv(args.subreddit_meta, sep="\t")
        df = df[df["subreddit"] == args.subreddit_filter]
        ids = set([int(x) for x in df["post_id"]])
        print(len(ids))
        print(type(ids))
    elif args.original_posts:
        # find the ids of original posts that we can easily guess the gender of
        op_data = pandas.read_csv(args.original_posts)
        # response ids and post ids should be parallel, so we can use the post_ids returned here as the ones to filter on
        # TODO: use cross validation to select ids, just use the same train and test data for now
        #op_id,op_gender,post_id,post_text,subreddit,op_gender_visible
        ids = get_op_gender_ids(op_data["post_text"], op_data["post_text"], op_data["op_gender"], op_data["op_gender"], op_data["post_id"], keep_thresh = 0.9)
    elif args.micro_posts and args.micro_processed:
        post_ids = set()
        micro_df = pandas.read_csv(args.micro_posts)

        # first get the post ids
        for i,row in micro_df.iterrows():
            if row["avg_ann"] and row["avg_ann"] > 0:
                post_ids.add(row["reply_id"])

        # map them to the ids in the attention file. They actually might be in the same order but this is safer
        micro_processed_df = pandas.read_csv(args.micro_processed, sep="\t")
        ids = set()
        for i,row in micro_processed_df.iterrows():
            if row[3] in post_ids:
                ids.add(row[0])
        print("Getting attention words for", len(ids), "microagression posts")

    labels = ["M", "W"]

    outdir = os.path.dirname(args.attention_file)

    if args.aggregate == "sum":
        attn_words_by_label(args.attention_file, labels, sum, ids, outdir, args.correct_only, args.bigrams, args.topk)
    elif args.aggregate == "mean":
        attn_words_by_label(args.attention_file, labels, np.mean, ids, outdir, args.correct_only, args.bigrams, args.topk)
    elif args.aggregate == "freq":
        freq_attn_words(args.attention_file, labels, ids, outdir, args.correct_only, args.topk, args.confident_only)

if __name__ == "__main__":
    main()



