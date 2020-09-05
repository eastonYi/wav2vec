"""
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
Modified by Easton.
"""
import logging
from argparse import ArgumentParser
from collections import Counter
from tqdm import tqdm
import re


def make_vocab(fpath, fname):
    """Constructs vocabulary.
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `fname`.
    """
    word2cnt = Counter()
    with open(fpath, encoding='utf-8') as f:
        for l in f:
            words = l.strip().split()[1:]
            # words = l.strip().split(',')[1].split()
            word2cnt.update(Counter(words))
    with open(fname, 'w', encoding='utf-8') as fout:
        for word, cnt in word2cnt.most_common():
            if re.findall('[0-9a-zA-Z]', word):
                continue
            fout.write(u"{} 1\n".format(word))
    logging.info('Vocab path: {}\t size: {}'.format(fname, len(word2cnt)))



def pre_processing(fpath, fname):
    import re
    with open(fpath, errors='ignore') as f, open(fname, 'w') as fw:
        for line in tqdm(f):
            line = line.strip().split(maxsplit=1)
            idx = line[0]
            # list_tokens = re.findall('\[[^\[\]]+\]|[a-zA-Z0-9^\[^\]]+|[^x00-xff]', line[1])
            list_tokens = re.findall('\[[^\[\]]+\]|[^x00-xff]|[A-Za-z]', line[1])
            list_tokens = [token.upper() for token in list_tokens]

            fw.write(idx+' '+' '.join(list_tokens)+'\n')


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
