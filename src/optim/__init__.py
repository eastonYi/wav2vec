# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from utils import registry
from optim.fairseq_optimizer import FairseqOptimizer


build_optimizer, register_optimizer, OPTIMIZER_REGISTRY = registry.setup_registry(
    '--optimizer',
    base_class=FairseqOptimizer,
    required=True,
)


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('optim.' + module)
