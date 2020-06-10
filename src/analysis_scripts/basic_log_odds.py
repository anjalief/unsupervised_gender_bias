import argparse
from collections import defaultdict, Counter
import pandas
import math

def write_log_odds(counts1, counts2, prior, outfile_name = None):
    # COPIED FROM LOG_ODDS FILE
    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    for word in prior.keys():
        prior[word] = int(prior[word] + 0.5)

    for word in counts2.keys():
        counts1[word] = int(counts1[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    for word in counts1.keys():
        counts2[word] = int(counts2[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    n1  = sum(counts1.values())
    n2  = sum(counts2.values())
    nprior = sum(prior.values())


    for word in prior.keys():
        if prior[word] > 0:
            l1 = float(counts1[word] + prior[word]) / (( n1 + nprior ) - (counts1[word] + prior[word]))
            l2 = float(counts2[word] + prior[word]) / (( n2 + nprior ) - (counts2[word] + prior[word]))
            sigmasquared[word] =  1/(float(counts1[word]) + float(prior[word])) + 1/(float(counts2[word]) + float(prior[word]))
            sigma[word] =  math.sqrt(sigmasquared[word])
            delta[word] = ( math.log(l1) - math.log(l2) ) / sigma[word]

    if outfile_name:
      outfile = open(outfile_name, 'w')
      for word in sorted(delta, key=delta.get):
        outfile.write(word)
        outfile.write(" %.3f\n" % delta[word])

      outfile.close()
    else:
      return delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file")
    args = parser.parse_args()

    df = pandas.read_csv(args.data_file, sep="\t")

    m_counter = Counter()
    w_counter = Counter()

    for i, row in df.iterrows():
        words = str(row[1]).split()
        if row[2] == "M":
            m_counter.update(words)
        else:
            assert row[2] == "W"
            w_counter.update(words)

    background = Counter()
    background.update(m_counter)
    background.update(w_counter)

    delta = write_log_odds(m_counter, w_counter, background)
    delta = sorted(delta, key=delta.get)

    for w in delta[:500]:
        print(w)
    print("################################################################################")
    for w in delta[-500:]:
        print(w)


if __name__ == "__main__":
    main()
