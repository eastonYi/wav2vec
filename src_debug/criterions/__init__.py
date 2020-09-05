# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import os

import registry
from criterions.fairseq_criterion import FairseqCriterion


build_criterion, register_criterion, CRITERION_REGISTRY = registry.setup_registry(
    '--criterion',
    base_class=FairseqCriterion,
    default='cross_entropy',
)

# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('criterions.' + module)

        # # expose `task_parser` for sphinx
        # if module in CRITERION_REGISTRY:
        #     parser = argparse.ArgumentParser(add_help=False)
        #     group_criterion = parser.add_argument_group('Task name')
        #     # fmt: off
        #     group_criterion.add_argument('--criterion', metavar=module,
        #                             help='Enable this task with: ``--criterion=' + module + '``')
        #     # fmt: on
        #     group_args = parser.add_argument_group('Additional command-line arguments')
        #     CRITERION_REGISTRY[module].add_args(group_args)
        #     globals()[module + '_parser'] = parser
