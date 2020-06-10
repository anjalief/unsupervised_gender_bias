# After we run model over the masked input, figure out
# which words have the greatest impact when they are masked out
import argparse
import pandas
from collections import Counter, defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_file", help="model output after running model_input_file through model")
    parser.add_argument("--model_input_file", help="file with masked inputs. e.g. output of mask_input.py")
    args = parser.parse_args()

    preds = pandas.read_csv(args.preds_file, sep="\t", header=None, names = ["text", "gold", "predicted", "op_post_id", "pred_score", "w_score"], dtype = {4: float, 3: int}, error_bad_lines=False)
    inputs = pandas.read_csv(args.model_input_file, header=None, sep="\t", error_bad_lines=False)

    print(len(preds), len(inputs))
    id_to_word = {}
    id_to_score = {}
    for i,row in inputs.iterrows():
        id_to_word[row[0]] = row[3]
        # id_to_score[row[0]] = row[4]
        id_to_score[row[0]] = 1
    print(len(id_to_word))

    word_score_diff = defaultdict(list)
    for i,row in preds.iterrows():
        post_id = row['op_post_id']
        if not post_id in id_to_word:
            continue
        masked_word = id_to_word[post_id]
        # we want the "female" score. If the prediction was M,
        # we need to take 1 - male score
        if row['predicted'] != 'W':
            new_score = 1 - row['pred_score']
        else:
            new_score = row['pred_score']
        # how much did female association decrease by
        word_score_diff[masked_word].append(id_to_score[post_id] - new_score)

    print(word_score_diff['iworked'])
    print(len(word_score_diff['iworked']))
    avg_score_diff = {w:sum(scores) / len(scores) for w,scores in word_score_diff.items() if len(scores) > 12}
    for w,s in sorted(avg_score_diff.items(), key=lambda v: v[1], reverse=True)[:100]:
        print(w, s)


if __name__ == "__main__":
    main()
