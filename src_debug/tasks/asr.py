# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import sys
import torch

from dataload import FileAudioDataset, Dictionary, AddTargetDataset, SemiSuperviseDataset
from .audio_pretraining import AudioUnsuperviseTrainingTask
from . import register_task


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@register_task("audio_ctc")
class AudioCtcTask(AudioUnsuperviseTrainingTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--labels",
            type=str,
            help="extension of the label file to load, if any",
        )
        AudioUnsuperviseTrainingTask.add_args(parser)

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries)."""
        dict_path = os.path.join(args.data, f"dict.{args.labels}.txt")
        if not os.path.isfile(dict_path):
            raise FileNotFoundError("Dict not found: {}".format(dict_path))
        tgt_dict = Dictionary.load(dict_path)
        tgt_dict.add_symbol("<ctc_blank>")

        print("| dictionary: {} types".format(len(tgt_dict)))
        return cls(args, tgt_dict)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        AudioUnsuperviseTrainingTask.load_dataset(self, split, **kwargs)

        label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
        labels = self.load_labels(label_path)

        process_label = LabelEncoder(self.dictionary)

        self.datasets[split] = AddTargetDataset(
            self.datasets[split],
            labels,
            bos=self.dictionary.bos(),
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            batch_targets=True,
            process_label=process_label,
        )

    @staticmethod
    def load_labels(label_path):
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                labels.append(line)

        return labels

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary


@register_task("audio_cif")
class AudioCifTask(AudioCtcTask):

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        AudioUnsuperviseTrainingTask.load_dataset(self, split, **kwargs)

        label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
        labels = self.load_labels(label_path)

        process_label = LabelEncoder(self.dictionary)

        self.datasets[split] = AddTargetDataset(
            self.datasets[split],
            labels,
            bos=self.dictionary.bos(),
            pad=self.dictionary.pad(),
            eos=None,
            batch_targets=True,
            process_label=process_label
        )


@register_task("audio_ctc_ce")
class AudioCtcCeTask(AudioCtcTask):

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        AudioUnsuperviseTrainingTask.load_dataset(self, split, **kwargs)

        label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
        labels = self.load_labels(label_path)

        process_label = LabelEncoder(self.dictionary)

        self.datasets[split] = AddTargetDataset(
            self.datasets[split],
            labels,
            bos=self.dictionary.bos(),
            pad=self.dictionary.pad(),
            eos=None,
            batch_targets=True,
            process_label=process_label
        )


@register_task("audio_gan_pretraining")
class AudioGANTrainingTask(AudioCtcTask):

    def load_dataset(self, split, **kwargs):
        # vocab
        dict_path = os.path.join(self.args.data, f"dict.{self.args.labels}.txt")
        self._target_dictionary = Dictionary.load(dict_path)
        process_label = LabelEncoder(self.dictionary)

        split = split.split(',')
        if len(split) == 1:
            split = split[0]
            manifest = os.path.join(self.args.data, "{}.tsv".format(split))
            self.datasets[split] = FileAudioDataset(
                manifest,
                sample_rate=self.args.sample_rate,
                max_sample_size=self.args.max_sample_size,
                min_sample_size=self.args.max_sample_size,
                min_length=self.args.min_sample_size,
                pad=self.args.labels is not None or self.args.enable_padding,
                normalize=self.args.normalize)
            # label
            label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
            labels = []
            with open(label_path, "r") as f:
                for line in f:
                    labels.append(line)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.dictionary.pad(),
                eos=self.dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=not self.is_ctc)
        else:
            train, untrain, text = split
            manifest = os.path.join(self.args.data, "{}.tsv".format(train))
            untrain_path = os.path.join(self.args.data, "{}.tsv".format(untrain))
            text_path = os.path.join(self.args.data, "{}.tsv".format(text))
            # x
            self.datasets[train] = FileAudioDataset(
                manifest,
                sample_rate=self.args.sample_rate,
                max_sample_size=self.args.max_sample_size,
                min_sample_size=self.args.max_sample_size,
                min_length=self.args.min_sample_size,
                pad=self.args.labels is not None or self.args.enable_padding,
                normalize=self.args.normalize)
            # y
            label_path = os.path.join(self.args.data, f"{train}.{self.args.labels}")
            labels = []
            with open(label_path, "r") as f:
                for line in f:
                    labels.append(line)
            # x'
            self.undatasets = FileAudioDataset(
                untrain_path,
                sample_rate=self.args.sample_rate,
                max_sample_size=self.args.max_sample_size,
                min_sample_size=self.args.max_sample_size,
                min_length=self.args.min_sample_size,
                pad=self.args.labels is not None or self.args.enable_padding,
                normalize=self.args.normalize)
            # z
            text_path = os.path.join(self.args.data, f"text.{self.args.labels}")
            text = []
            with open(text_path, "r") as f:
                for line in f:
                    text.append(line)

            self.datasets[train] = SemiSuperviseDataset(
                self.undatasets,
                text,
                self.datasets[train],
                labels,
                pad=self.dictionary.pad(),
                eos=self.dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=not self.is_ctc)

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        return loss, sample_size, logging_output
