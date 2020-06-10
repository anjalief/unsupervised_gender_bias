# Take in a preds file (output of model run over test data
# For the 500 posts with the highest score (e.g. closest to W),
# write the post with each word masked out one by one
import argparse
import pandas

# text = "my lovely beautiful victoria justice i kiss your sweet beautiful hands and i also cover your legs with kisses everywhere <3"
# outfile = "/projects/tir3/users/anjalief/adversarial_gender/rt_gender_clean_facebook_wiki/test_masked.txt"
global_count = 0
def write_masked(words, fp, pred_score):
    global global_count
    new_words = []
    prev_word = None
    for w in words:
        if prev_word is None or w != prev_word:
            prev_word = w
            new_words.append(w)
    words = new_words
    len_words = len(words)
    if len_words < 4:
        return

    for i in range(len_words):
        fp.write("%s\t" % global_count)
        global_count += 1
        for j in range(len_words):
            if i != j:
                fp.write("%s " % words[j])
        fp.write("\t")
        fp.write("W")
        fp.write("\t")
        fp.write("%s" % words[i])
        fp.write("\t")
        fp.write("%s" % pred_score)
        fp.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_file")
    parser.add_argument("--outfile")
    args = parser.parse_args()

    fp = open(args.outfile, "w")
    df = pandas.read_csv(args.preds_file, sep="\t", header=None, names = ["text", "gold", "predicted", "op_post_id", "pred_score", "w_score"], dtype = {4: float, 3: int}, error_bad_lines=False)
    df = df[df['predicted'] == 'W']
    print("Num W predicted", len(df))

    count = 0
    for i,row in df.sort_values(by=["pred_score"], ascending=False).iterrows():
        count += 1
        if count > 500:
            break
        words = [w.split(':')[0] for w in row['text'].split()]
        write_masked(words, fp, row['pred_score'])
    fp.close()

if __name__ == "__main__":
    main()
