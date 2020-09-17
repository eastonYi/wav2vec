# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F

from loggings import metrics
from tools import utils
from criterions import FairseqCriterion, register_criterion
from loggings.meters import safe_round


@register_criterion('cross_entropy')
class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion('cross_entropy_uer')
class CrossEntropyUerCriterion(CrossEntropyCriterion):

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        sample_size = sample['ntokens']
        loss, losses = self.compute_loss(net_output, sample)

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        if 'ce_loss' in losses:
            logging_output['ce_loss'] = losses['ce_loss']

        if 'qua_loss' in losses:
            logging_output['qua_loss'] = losses['qua_loss']

        # if not model.training:
        #     import editdistance
        #
        #     with torch.no_grad():
        #         probs = F.softmax(logits, dim=-1).float().cpu()
        #
        #         c_err = 0
        #         c_len = 0
        #         for p, t, inp_l in zip(
        #             probs,
        #             sample["target"],
        #             input_lengths,
        #         ):
        #             p = p[:inp_l].unsqueeze(0)
        #
        #             p = (t != self.task.target_dictionary.pad()) & (
        #                 t != self.task.target_dictionary.eos()
        #             )
        #             targ = t[p]
        #             targ_units_arr = targ.tolist()
        #
        #             toks = p.argmax(dim=-1).unique_consecutive()
        #             pred_units_arr = toks.tolist()
        #
        #             c_err += editdistance.eval(pred_units_arr, targ_units_arr)
        #             c_len += len(targ_units_arr)
        #
        #         logging_output["c_errors"] = c_err
        #         logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    def compute_loss(self, net_output, sample):
        logits = net_output['logits']
        target = sample['target']
        target_lengths = sample['target_lengths']
        target_paddings = 1 - utils.sequence_mask(target_lengths)
        losses = {}
        loss = ce_loss = cal_ce_loss(logits, target, target_paddings)
        losses['ce_loss'] = ce_loss

        if 'num_output' in net_output:
            _number = net_output['num_output']
            number = target_lengths.float()
            qua_loss = torch.sqrt(torch.pow(_number - number, 2)).sum()
            losses['qua_loss'] = qua_loss
            loss = loss * 100 + qua_loss * 1

        return loss, losses

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=3)

        if 'ce_loss' in logging_outputs[0]:
            ce_loss = sum(log['ce_loss'] for log in logging_outputs) / ntokens
            metrics.log_scalar('ce_loss', ce_loss, ntokens, round=3)
        if 'qua_loss' in logging_outputs[0]:
            qua_loss = sum(log['qua_loss'] for log in logging_outputs) / nsentences
            metrics.log_scalar('qua_loss', qua_loss, nsentences, round=3)

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


def cal_ce_loss(logits, target_labels, target_paddings, label_smooth=0.0):
    losses = _compute_cross_entropy_losses(logits, target_labels.long(), target_paddings)
    loss = losses.sum()
    if label_smooth > 0:
        loss = loss * (1-label_smooth) + _uniform_label_smooth(logits, target_paddings)*label_smooth

    return loss


def _uniform_label_smooth(logits, paddings):
    log_probs = F.log_softmax(logits, dim=-1)
    nlabel = log_probs.shape[-1]
    ent_uniform = -torch.sum(log_probs, dim=-1)/nlabel

    return torch.sum(ent_uniform*(1-paddings).float())


def _compute_cross_entropy_losses(logits, labels, paddings):
    B, T, V = logits.shape
    losses = F.cross_entropy(logits.contiguous().view(-1, V),
                             labels.contiguous().view(-1),
                             reduction="none").view(B, T) * (1-paddings).float()

    return losses
