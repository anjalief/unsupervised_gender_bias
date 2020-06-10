import argparse
from sklearn.metrics import f1_score, precision_score, recall_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention_file', type=str, required=True,
                        help='file containing the word:attention data')
    args = parser.parse_args()

    labels = ["M", "W"]
    label2int = {}
    for i,l in enumerate(labels):
        label2int[l] = i

    f = open(args.attention_file)

    c, C = 0.0, 0.0
    y_true, y_pred = [], []
    for l in f:
        items = l.strip().split("\t")
        attentions, label, predicted, post_id = items[0], items[1], items[2], items[3]
        C += 1
        y_true.append(label2int[label])
        y_pred.append(label2int[predicted])
        if label == predicted:
            c += 1
    f.close()

    print ("Accuracy out of", c/C, C)
    print ("F1 score", f1_score(y_true, y_pred))
    print ("Precision", precision_score(y_true, y_pred))
    print ("Recall", recall_score(y_true, y_pred))
    print ("Macro F1 score", f1_score(y_true, y_pred, average='macro'))

if __name__ == "__main__":
    main()



