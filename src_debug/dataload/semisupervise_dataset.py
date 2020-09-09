# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random

from . import AddTargetDataset
from . import data_utils


class SemiSuperviseDataset(AddTargetDataset):
    def __init__(self, dataset, text, minidataset, labels, pad, eos, batch_targets, process_label=None, add_to_input=False):
        super().__init__(dataset, labels, pad, eos, batch_targets, process_label, add_to_input)
        self.text = text
        self.minidataset = minidataset

    def get_text(self, index=None):
        if not index:
            text = random.choice(self.text)
        else:
            text = self.labels[index]
        if self.process_label:
            text = self.process_label(text)

        return text

    def __getitem__(self, index):
        item = self.dataset[index]
        item["text"] = self.get_text()

        rand = random.randint(0, len(self.minidataset))
        item['minisource'] = self.minidataset[rand]
        item["label"] = self.get_label(rand)

        return item

    def size(self, index):
        print('===========================\n\n\n\n====****************************************===================================================')
        sz = self.dataset.size(index)
        own_sz = len(self.get_label(index)[0])
        return (sz, own_sz, )

    def collater(self, samples):
        from tools import pdb; pdb.set_trace()
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        text = [s["text"] for s in samples]
        target = [s["label"] for s in samples]

        # assert len(target) == len(text), 'the number of target and text is not equal'

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
            text = data_utils.collate_tokens(text, pad_idx=self.pad, left_pad=False)
            collated["ntokens"] = collated["target_lengths"].sum().item()
        else:
            collated["ntokens"] = sum([len(t) for t in target])

        collated["target"] = target
        collated["net_input"]["text"] = text

        if self.add_to_input:
            eos = target.new_full((target.size(0), 1), self.eos)
            collated["target"] = torch.cat([target, eos], dim=-1).long()
            collated["net_input"]["prev_output_tokens"] = torch.cat([eos, target], dim=-1).long()
            collated["ntokens"] += target.size(0)

        return collated
