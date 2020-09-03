#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a wav2letter++ dataset
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.tsv, "r") as tsv, \
         open(os.path.join(args.output_dir, args.output_name + ".ltr.txt"), "w") as ltr_out, \
         open(os.path.join(args.output_dir, args.output_name + ".wrd.txt"), "w") as wrd_out:
        root = next(tsv).strip()
        for line in tsv:
            line = line.strip()
            file = line.split()[0]
            path = os.path.join(root, file.replace('wav', 'label'))
            # assert os.path.exists(path)
            texts = {}
            with open(path, "r") as trans_f:
                for tline in trans_f:
                    items = tline.strip().split()
                    texts[items[0]] = " ".join(items[1:])


if __name__ == "__main__":
    main()
