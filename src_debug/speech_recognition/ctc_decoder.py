#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
CTC decoders.
"""
import itertools as it
import torch
from ctcdecode import CTCBeamDecoder


class CTCDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest
        self.beam = args.beam
        self.blank = (
            tgt_dict.index("<ctc_blank>") if "<ctc_blank>" in tgt_dict.indices else tgt_dict.bos()
        )

        self.decode_fn = CTCBeamDecoder(tgt_dict.symbols,
                                         beam_width=self.beam,
                                         blank_id=self.blank,
                                         num_processes=10)

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)

        return self.decode(emissions)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        # encoder_out = models[0].encoder(**encoder_input)
        encoder_out = models[0](**encoder_input)
        emissions = models[0].get_normalized_probs(encoder_out, log_probs=False)

        return emissions.transpose(0, 1)

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)

        return torch.LongTensor(list(idxs))

    def decode(self, emissions):
        hypos = []
        beam_results, beam_scores, timesteps, out_seq_len = self.decode_fn.decode(emissions)
        for beam_result, scores, lengthes in zip(beam_results, beam_scores, out_seq_len):
            # beam_ids: beam x id; score: beam; length: beam
            top = []
            for result, score, length in zip(beam_result, scores, lengthes):
                top.append({'tokens': self.get_tokens(result[:length]),
                            "score": score})
            hypos.append(top)

        return hypos
