# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import math
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import utils

from models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
    TransposeLast,
    Fp32LayerNorm,
    Fp32GroupNorm
)
from .wav2vec2_ctc import add_common_args, Wav2VecEncoder, Linear, base_architecture


def add_decoder_args(parser):
    """Add model-specific arguments to the parser."""
    # fmt: off
    parser.add_argument('--dropout', type=float, metavar='D',
                        help='dropout probability')
    parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                        help='encoder embedding dimension')
    parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                        help='path to pre-trained encoder embedding')
    parser.add_argument('--encoder-freeze-embed', action='store_true',
                        help='freeze encoder embeddings')
    parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                        help='encoder hidden size')
    parser.add_argument('--encoder-layers', type=int, metavar='N',
                        help='number of encoder layers')
    parser.add_argument('--encoder-bidirectional', action='store_true',
                        help='make all layers of encoder bidirectional')
    parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                        help='decoder embedding dimension')
    parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                        help='path to pre-trained decoder embedding')
    parser.add_argument('--decoder-freeze-embed', action='store_true',
                        help='freeze decoder embeddings')
    parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                        help='decoder hidden size')
    parser.add_argument('--decoder-layers', type=int, metavar='N',
                        help='number of decoder layers')
    parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                        help='decoder output embedding dimension')
    parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                        help='decoder attention')
    parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                        help='comma separated list of adaptive softmax cutoff points. '
                             'Must be used with adaptive_loss criterion')
    parser.add_argument('--share-decoder-input-output-embed', default=False,
                        action='store_true',
                        help='share decoder input and output embeddings')
    parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                        help='share encoder, decoder and output embeddings'
                             ' (requires shared dictionary and embed dim)')

    # Granular dropout settings (if not specified these default to --dropout)
    parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                        help='dropout probability for encoder input embedding')
    parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                        help='dropout probability for encoder output')
    parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                        help='dropout probability for decoder input embedding')
    parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                        help='dropout probability for decoder output')


@register_model("wav2vec_ctc_lm")
class CTC_LM(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_decoder_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        tgt_dict = task.dictionary

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, tgt_dict)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args):
        return Wav2VecEncoder(args)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_dim):
        from models.lstm import LSTMEncoder

        encoder = LSTMEncoder(
            dictionary=tgt_dict,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=False,
            pretrained_embed=False,
            max_source_positions=1000,
        )
        return encoder

    def forward(self, **kwargs):
        encoder_out = self.encoder(tbc=False, **kwargs)
        decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def get_normalized_probs(self, logits, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        if log_probs:
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            res = utils.softmax(logits.float(), dim=-1)
        res.batch_first = True

        return res


class FCDecoder(FairseqDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~dataload.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, input_dim):
        super().__init__(dictionary)
        self.proj = Linear(input_dim, len(dictionary))

    def forward(self, encoded):
        """
        Args:
            encoder_out (Tensor): output from the encoder, used for
                encoder-side attention
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
        """
        x = self.proj(encoded)
        return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


@register_model_architecture("wav2vec_ctc_lm", "wav2vec_ctc_lm")
def ctc_lm_architecture(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = getattr(args, "decoder_layers", 10)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.decoder_dropout = getattr(args, "decoder_dropout", 0)
    args.decoder_attention_dropout = getattr(args, "decoder_attention_dropout", 0)
    args.decoder_activation_dropout = getattr(args, "decoder_activation_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    base_architecture(args)
