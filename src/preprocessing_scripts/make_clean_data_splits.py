# Split dat set by people, balancing on op_gender
# and printing number of comments
# Filter out short posts and stratify split the remaining
# post to have balanced splits

# python make_data_splits.py "/projects/tir3/users/anjalief/adversarial_gender/rt_gender_fb"

import pandas
import argparse
import glob
from sklearn.model_selection import train_test_split
from collections import Counter
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file", help="Should be all tokenized responses")
    parser.add_argument("--output_dir", help="Directory to write split data")
    args = parser.parse_args()

    all_data = pandas.read_csv(args.response_file, sep="\t")

    def short_filter(text):
        return len(str(text).split()) > 4

    print("Before short removal", len(all_data))
    all_data = all_data[all_data["response_text"].apply(short_filter)]
    print("After short removal", len(all_data))

    no_dupls = all_data.drop_duplicates(subset="op_id")
    id_to_gender = {}
    for i,row in no_dupls.iterrows():
        id_to_gender[row['op_id']] = row['op_gender']

    train, test_valid = train_test_split(no_dupls['op_id'], test_size=0.4, stratify=no_dupls['op_gender'])
    test_valid_genders = [id_to_gender[i] for i in test_valid]
    test, valid = train_test_split(test_valid, test_size=0.5, stratify=test_valid_genders)


    id_to_count = Counter(all_data['op_id'])
    def print_stats(op_ids):
        print("Num people", len(op_ids))
        num_M_comments = 0
        num_F_comments = 0
        num_M = 0
        num_F = 0
        for i in op_ids:
            if id_to_gender[i] == 'M':
                num_M_comments += id_to_count[i]
                num_M += 1
            else:
                num_F_comments += id_to_count[i]
                num_F += 1
        total = num_M_comments + num_F_comments
        print("%s %s %s %s %s" % ((num_M_comments / total), (num_F_comments / total), total, num_M, num_F))


    def save_files(name, op_ids):
        filename = os.path.join(args.output_dir, name +  ".txt")
        print("Saving", filename)
        print_stats(op_ids)
        write_data = all_data[all_data['op_id'].isin(op_ids)]
        write_data.to_csv(filename, sep="\t")
        fp = open(os.path.join(args.output_dir, name + "_op_ids.txt"), "w")
        write_str = " ".join([str(i) for i in op_ids])
        fp.write(write_str)
        fp.close()


    save_files("train", train)
    save_files("test", test)
    save_files("valid", valid)

if __name__ == "__main__":
    main()
