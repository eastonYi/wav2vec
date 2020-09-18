"""
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
Modified by Easton.
"""
import logging
from argparse import ArgumentParser
from collections import Counter


def make_vocab(fpaths, fname):
    """Constructs vocabulary.
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `fname`.
    """
    word2cnt = Counter()
    for fpath in fpaths.split(','):
        with open(fpath, encoding='utf-8') as f:
            for l in f:
                words = l.strip().split()
                # words = l.strip().split()[1:]
                # words = l.strip().split(',')[1].split()
                word2cnt.update(Counter(words))
    with open(fname, 'w', encoding='utf-8') as fout:
        for word, cnt in word2cnt.most_common():
            fout.write(u"{} {}\n".format(word, cnt))
    logging.info('Vocab path: {}\t size: {}'.format(fname, len(word2cnt)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, dest='src_vocab')
    parser.add_argument('--input', type=str, dest='src_path')
    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)
    make_vocab(args.src_path, args.src_vocab)
    # pre_processing(args.src_path, args.src_vocab)
    logging.info("Done")
