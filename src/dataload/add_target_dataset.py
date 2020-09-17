# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import BaseWrapperDataset
from . import data_utils


class AddTargetDataset(BaseWrapperDataset):
    def __init__(self, dataset, labels, pad, bos, eos, batch_targets, process_label=None, add_to_input=False):
        super().__init__(dataset)
        self.labels = labels
        self.batch_targets = batch_targets
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.process_label = process_label
        self.add_to_input = add_to_input

        assert batch_targets

    def get_label(self, index):
        return self.labels[index] if self.process_label is None else self.process_label(self.labels[index])

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.get_label(index)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_label(index))
        return (sz, own_sz)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())

        if self.add_to_input:
            if self.eos:
                eos = torch.ones([1]).int() * self.eos
                target = [torch.cat([s["label"], eos], dim=-1) for s in samples if s["id"] in indices]
                bos = torch.ones([1]).int() * self.bos
                prev_output_tokens = [torch.cat([bos, s["label"]], dim=-1) for s in samples if s["id"] in indices]
                collated["net_input"]["prev_output_tokens"] = \
                    data_utils.collate_tokens(prev_output_tokens, pad_idx=self.pad, left_pad=False)
            else:
                target = [s["label"] for s in samples if s["id"] in indices]
                # bos = torch.ones([1]).int() * self.bos
                # prev_output_tokens = [torch.cat([bos, s["label"][:-1]], dim=-1) for s in samples if s["id"] in indices]
                # collated["net_input"]["prev_output_tokens"] = \
                #     data_utils.collate_tokens(prev_output_tokens, pad_idx=self.pad, left_pad=False)
        else:
            target = [s["label"] for s in samples if s["id"] in indices]

        collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
        collated["net_input"]["target_lengths"] = collated["target_lengths"]
        target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
        collated["ntokens"] = collated["target_lengths"].sum().item()
        collated["target"] = target

        return collated
