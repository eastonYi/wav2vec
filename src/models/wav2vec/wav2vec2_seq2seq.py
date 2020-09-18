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
    parser.add_argument(
        "--decoder-embed-dim",
        type=int,
        metavar="N",
        help="decoder embedding dimension",
    )
    parser.add_argument(
        "--decoder-ffn-embed-dim",
        type=int,
        metavar="N",
        help="decoder embedding dimension for FFN",
    )
    parser.add_argument(
        "--decoder-layers", type=int, metavar="N", help="num decoder layers"
    )
    parser.add_argument(
        "--decoder-layerdrop",
        type=float,
        metavar="D",
        help="decoder layerdrop chance",
    )
    parser.add_argument(
        "--decoder-attention-heads",
        type=int,
        metavar="N",
        help="num decoder attention heads",
    )
    parser.add_argument(
        "--decoder-learned-pos",
        action="store_true",
        help="use learned positional embeddings in the decoder",
    )
    parser.add_argument(
        "--decoder-normalize-before",
        action="store_true",
        help="apply layernorm before each decoder block",
    )
    parser.add_argument(
        "--no-token-positional-embeddings",
        default=False,
        action="store_true",
        help="if set, disables positional embeddings (outside self attention)",
    )

    parser.add_argument(
        "--decoder-dropout",
        type=float,
        metavar="D",
        help="dropout probability in the decoder",
    )
    parser.add_argument(
        "--decoder-attention-dropout",
        type=float,
        metavar="D",
        help="dropout probability for attention weights inside the decoder",
    )
    parser.add_argument(
        "--decoder-activation-dropout",
        type=float,
        metavar="D",
        help="dropout probability after activation in FFN inside the decoder",
    )


@register_model("wav2vec_seq2seq")
class TransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_decoder_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        seq2seq_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return TransformerModel(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args):
        return Wav2VecEncoder(args)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(args, tgt_dict, embed_tokens)

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


@register_model("wav2vec_cif")
class CIFModel(TransformerModel):
    def __init__(self, args, encoder, assigner, decoder):
        super().__init__(args, encoder, decoder)
        self.assigner = assigner

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_decoder_args(parser)
        parser.add_argument(
            "--assigner-conv-layers",
            type=str,
            metavar="EXPR",
            help="convolutional feature extraction layers [(dim, kernel_size, stride), ...]",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        cif_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048

        tgt_dict = task.target_dictionary

        # def build_embedding(dictionary, embed_dim):
        #     num_embeddings = len(dictionary)
        #     padding_idx = dictionary.pad()
        #     emb = Embedding(num_embeddings, embed_dim, padding_idx)
        #     return emb

        encoder = cls.build_encoder(args)
        assigner = cls.build_assigner(args, encoder.d)
        decoder = cls.build_decoder(args, tgt_dict, encoder.d)
        return CIFModel(args, encoder, assigner, decoder)

    @classmethod
    def build_assigner(cls, args, dim_input):
        return Assigner(args, dim_input)

    @classmethod
    def build_decoder(cls, args, tgt_dict, input_dim):
        return FCDecoder(args, tgt_dict, input_dim)

    def forward(self, **kwargs):
        encoder_out = self.encoder(tbc=False, **kwargs)
        alphas = self.assigner(encoder_out)
        _alphas, num_output = self.resize(alphas, kwargs['target_lengths'])
        # if self.training:
        #     _alphas = self.resize(alphas, kwargs['target_lengths'])
        # else:
        #     print('eval mode forward')
        #     _alphas = alphas
        cif_outputs = self.cif(encoder_out, _alphas)
        logits = self.decoder(cif_outputs)
        return {'logits': logits, 'num_output': num_output}

    def cif(self, encoder_out, alphas, threshold=0.95, log=False):
        hidden = encoder_out['encoder_out']
        device = hidden.device
        batch_size, len_time, hidden_size = hidden.size()

        # loop varss
        integrate = torch.zeros([batch_size]).to(device)
        frame = torch.zeros([batch_size, hidden_size]).to(device)
        # intermediate vars along time
        list_fires = []
        list_frames = []

        for t in range(len_time):
            alpha = alphas[:, t]
            distribution_completion = torch.ones([batch_size]).to(device) - integrate

            integrate += alpha
            list_fires.append(integrate)

            fire_place = integrate > threshold
            integrate = torch.where(fire_place,
                                    integrate - torch.ones([batch_size]).to(device),
                                    integrate)
            cur = torch.where(fire_place,
                              distribution_completion,
                              alpha)
            remainds = alpha - cur

            frame += cur[:, None] * hidden[:, t, :]
            list_frames.append(frame)
            frame = torch.where(fire_place[:, None].repeat(1, hidden_size),
                                remainds[:, None] * hidden[:, t, :],
                                frame)
            if log:
                print('t: {}\t{:.3f} -> {:.3f}|{:.3f}'.format(
                    t, integrate[0].numpy(), cur[0].numpy(), remainds[0].numpy()))

        fires = torch.stack(list_fires, 1)
        frames = torch.stack(list_frames, 1)
        list_ls = []
        len_labels = torch.round(alphas.sum(-1)).int()
        max_label_len = len_labels.max()
        for b in range(batch_size):
            fire = fires[b, :]
            l = torch.index_select(frames[b, :, :], 0, torch.where(fire > threshold)[0])
            pad_l = torch.zeros([max_label_len - l.size(0), hidden_size]).to(device)
            list_ls.append(torch.cat([l, pad_l], 0))

        if log:
            print('fire:\n', fires.numpy())

        return torch.stack(list_ls, 0)

    @staticmethod
    def resize(alphas, target_lengths):
        device = alphas.device
        # sum
        _num = alphas.sum(-1)
        # scaling
        num = target_lengths.float()
        num_noise = num + 0.9 * torch.rand(alphas.size(0)).to(device) - 0.45
        alphas *= (num_noise / _num)[:, None].repeat(1, alphas.size(1))

        return alphas, _num


class TransformerDecoder(FairseqIncrementalDecoder):
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

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)

        self.dropout = args.decoder_dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_embed_dim
        args.encoder_embed_dim = embed_dim

        self.layerdrop = args.decoder_layerdrop

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings else None
        )

        args = copy.deepcopy(args)
        args.dropout = args.decoder_dropout
        args.attention_dropout = args.decoder_attention_dropout
        args.activation_dropout = args.decoder_activation_dropout

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, attn, _ = layer(
                    x,
                    encoder_out["encoder_out"] if encoder_out is not None else None,
                    encoder_out["encoder_padding_mask"] if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x)
                    if incremental_state is None else None,
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class Assigner(FairseqEncoder):
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

    def __init__(self, args, dim_input):
        super().__init__()
        assigner_conv_layers = eval(args.assigner_conv_layers)
        self.embed = assigner_conv_layers[-1][0]
        self.feature_extractor = Conv2DFeatureExtractionModel(
            dim_input=dim_input,
            conv_layers=assigner_conv_layers,
            dropout=0.1,
            mode=args.extractor_mode,
            conv_bias=True,
            output='same'
        )
        self.proj = Linear(self.embed, 1)

    def forward(self, encoder_out):
        """
        Args:
            encoded (FloatTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoded_lengths (Tensor): output from the encoder, used for
                encoder-side attention
        Returns:
            the decoder's output of shape `(batch, src_len)`
        """
        encoded, padding_mask = encoder_out['encoder_out'], encoder_out['padding_mask']

        x = self.feature_extractor(encoded)
        x = self.proj(x)[:, :, 0]
        x = torch.sigmoid(x)
        x = x * (~padding_mask)

        return x


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


class Conv2DFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        dim_input: int,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        output: str = "valid", # ["valid", "same"]
    ):
        super().__init__()
        assert mode in {"default", "layer_norm"}
        assert output in {"valid", "same"}
        self.output = output

        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = dim_input
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(in_d, dim, k, stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias)
            )
            in_d = dim

    def forward(self, x):
        if self.output == 'same':
            length = x.size(1)
            x = F.pad(x, [0,0,0,20,0,0])
        x = x.transpose(1,2)

        for conv in self.conv_layers:
            x = conv(x)

        x = x.transpose(1,2)

        if self.output == 'same':
            x = x[:, :length, :]

        return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


@register_model_architecture("wav2vec_seq2seq", "wav2vec_seq2seq")
def seq2seq_architecture(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = getattr(args, "decoder_layers", 10)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.decoder_dropout = getattr(args, "decoder_dropout", 0)
    args.decoder_attention_dropout = getattr(args, "share-decoder-input-output-embed", 0)
    args.decoder_activation_dropout = getattr(args, "decoder_activation_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    base_architecture(args)


@register_model_architecture("wav2vec_cif", "wav2vec_cif")
def cif_architecture(args):
    args.extractor_mode = getattr(args, "extractor_mode", 'default')
    args.conv_bias = getattr(args, "conv-bias", False)
    seq2seq_architecture(args)
