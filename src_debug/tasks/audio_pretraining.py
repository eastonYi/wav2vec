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
from . import FairseqTask, register_task


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@register_task("audio_pretraining")
class AudioPretrainingTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--sample-rate",
            default=16000,
            type=int,
            help="target sample rate. audio files will be up/down sampled to this rate",
        )
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="if set, normalizes input to have 0 mean and unit variance",
        )
        parser.add_argument(
            "--max-sample-size",
            default=None,
            type=int,
            help="max sample size to crop to for batching. default = min sample length",
        )
        parser.add_argument(
            "--min-sample-size",
            default=None,
            type=int,
            help="min sample size to crop to for batching. default = same as --max-sample-size",
        )

        parser.add_argument(
            "--enable-padding",
            action="store_true",
            help="pad shorter samples instead of cropping",
        )

        parser.add_argument(
            "--labels",
            type=str,
            default=None,
            help="extension of the label file to load, if any",
        )

    def __init__(self, args, source_dictionary=None):
        super().__init__(args)
        self._target_dictionary = None
        self._source_dictionary = source_dictionary
        self.is_ctc = args.criterion == "ctc"

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        manifest = os.path.join(self.args.data, "{}.tsv".format(split))
        self.datasets[split] = FileAudioDataset(
            manifest,
            sample_rate=self.args.sample_rate,
            max_sample_size=self.args.max_sample_size,
            min_sample_size=self.args.max_sample_size,
            min_length=self.args.min_sample_size,
            pad=self.args.labels is not None or self.args.enable_padding,
            normalize=self.args.normalize,
        )

        if self.args.labels:
            dict_path = os.path.join(self.args.data, f"dict.{self.args.labels}.txt")
            self._target_dictionary = Dictionary.load(dict_path)
            label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
            labels = []
            with open(label_path, "r") as f:
                for line in f:
                    labels.append(line)

            process_label = LabelEncoder(self.target_dictionary)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                bos=self.target_dictionary.bos(),
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=not self.is_ctc,
            )

    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self._target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
            self,
            indices,
            dataset,
            max_positions=None,
            ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices


@register_task("audio_cif_pretraining")
class AudioCIFPretrainingTask(AudioPretrainingTask):

    def __init__(self, args, source_dictionary=None):
        super().__init__(args)
        self._target_dictionary = None
        self._source_dictionary = source_dictionary

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        manifest = os.path.join(self.args.data, "{}.tsv".format(split))
        self.datasets[split] = FileAudioDataset(
            manifest,
            sample_rate=self.args.sample_rate,
            max_sample_size=self.args.max_sample_size,
            min_sample_size=self.args.max_sample_size,
            min_length=self.args.min_sample_size,
            pad=self.args.labels is not None or self.args.enable_padding,
            normalize=self.args.normalize,
        )

        dict_path = os.path.join(self.args.data, f"dict.{self.args.labels}.txt")
        self._target_dictionary = Dictionary.load(dict_path)
        label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                labels.append(line)

        process_label = LabelEncoder(self.target_dictionary)

        self.datasets[split] = AddTargetDataset(
            self.datasets[split],
            labels,
            bos=self.target_dictionary.bos(),
            pad=self.target_dictionary.pad(),
            eos=None,
            batch_targets=True,
            process_label=process_label,
            add_to_input=False,
        )


@register_task("audio_gan_pretraining")
class AudioGANTrainingTask(AudioPretrainingTask):

    def load_dataset(self, split, **kwargs):
        # vocab
        dict_path = os.path.join(self.args.data, f"dict.{self.args.labels}.txt")
        self._target_dictionary = Dictionary.load(dict_path)
        process_label = LabelEncoder(self.target_dictionary)

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
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
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
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
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
