#!/usr/bin/python
import re
from collections import Counter
from argparse import ArgumentParser

if __name__ == "__main__":
    """
    """
    parser = ArgumentParser()
    parser.add_argument('--input', dest='input')
    parser.add_argument('--output', dest='output')
    args = parser.parse_args()

    word2cnt = Counter()
    with open(args.input, encoding='utf-8') as f:
        for l in f:
            words = re.split(' |-', l.strip())
            word2cnt.update(Counter(words))

    with open(args.output, 'w') as fout:
        for word, cnt in word2cnt.most_common():
            line = word + ' ' + ' '.join(word) + ' |'
            fout.write(line + '\n')
