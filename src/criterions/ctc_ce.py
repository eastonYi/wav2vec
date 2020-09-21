# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import logging
import editdistance

import torch
import torch.nn.functional as F
from loggings import metrics
from tools import utils
from criterions import FairseqCriterion, register_criterion
from loggings.meters import safe_round


@register_criterion("ctc_ce")
class CtcCeCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.blk_idx = task.target_dictionary.index("<ctc_blank>")
        self.pad_idx = task.target_dictionary.pad()
        self.bos_idx = task.target_dictionary.bos()
        self.eos_idx = task.target_dictionary.eos()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--zero-infinity", action="store_true", help="zero inf loss"
        )
        try:
            parser.add_argument(
                "--remove-bpe",
                "--post-process",
                default="letter",
                help="remove BPE tokens before scoring (can be set to sentencepiece, letter, and more)",
            )
        except:
            pass  # this option might have been added from eval args

    def forward(self, model, sample, reduce=True):
        encoder_output, logits = model(**sample["net_input"])
        
        # (B, T, C) from the encoder
        ctc_logits = encoder_output["ctc_logits"]
        ctc_lprobs, lprobs = model.get_normalized_probs(ctc_logits, logits, log_probs=True)

        len_ctc_logits = (~encoder_output["padding_mask"]).long().sum(-1)
        target = sample["target"]
        target_lengths = sample["target_lengths"]

        ctc_loss = self.ctc_loss(ctc_lprobs, len_ctc_logits, target, target_lengths)
        ce_loss, correct, total = self.ce_loss(lprobs, target)
        loss = ctc_loss + ce_loss

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else sample["target_lengths"].sum().item()
        )
        sample_size = ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ctc_loss": utils.item(ctc_loss.data),
            "ce_loss": utils.item(ce_loss.data),
            "ntokens": ntokens,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:

            ctc_lprobs = ctc_lprobs.float().cpu()

            c_err, c_len = self.ctc_greedy_eval(
                ctc_lprobs, len_ctc_logits, target,
                pad_idx=self.pad_idx,
                eos_idx=self.eos_idx,
                blk_idx=self.blk_idx)

            logging_output["c_errors"] = c_err
            logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def ctc_greedy_eval(lprobs, len_lprobs, target, pad_idx, eos_idx, blk_idx):
        """
            lprobs: B x T x V
        """
        with torch.no_grad():
            lprobs_t = lprobs.float().cpu()

            c_err = 0
            c_len = 0
            for lp, t, inp_l in zip(lprobs_t, target, len_lprobs):
                lp = lp[:inp_l].unsqueeze(0)

                p = (t != pad_idx) & (t != eos_idx)
                targ = t[p]
                targ_units_arr = targ.tolist()

                toks = lp.argmax(dim=-1).unique_consecutive()
                pred_units_arr = toks[toks != blk_idx].tolist()

                c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                c_len += len(targ_units_arr)

        return c_err, c_len

    def ctc_loss(self, lprobs, len_lprobs, target, target_lengths):
        if getattr(lprobs, "batch_first", True):
            lprobs = lprobs.transpose(0, 1) # T x B x V

        pad_mask = (target != self.pad_idx) & (target != self.eos_idx)
        target_lengths = target_lengths - 1
        targets_flat = target.masked_select(pad_mask)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                len_lprobs,
                target_lengths,
                blank=self.blk_idx,
                reduction="sum",
                zero_infinity=True,
            )

        return loss

    def ce_loss(self, lprobs, target):
        # N, T -> N * T
        target = target.view(-1).long()

        # N, T, D -> N * T, D
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = F.nll_loss(
            lprobs, target, ignore_index=self.padding_idx, reduction="sum"
        )

        mask = target != self.padding_idx
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)
        )
        total = torch.sum(mask)

        return loss, correct, total

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(sum(log.get("ctc_loss", 0) for log in logging_outputs))
        ce_loss_sum = utils.item(sum(log.get("ce_loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("ce_loss", ce_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("acc", correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0)
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3)
                if meters["_c_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
