import logging
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import BaseFairseqModel, register_model
from modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    TransposeLast,
)

logger = logging.getLogger(__name__)


@register_model("discriminator")
class CLM(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()

        if args.activation == "relu":
            activation = nn.ReLU()
        elif args.activation == "gelu":
            activation = nn.GELU()
        else:
            raise Exception("unknown activation " + args.activation)

        feature_enc_layers = eval(args.conv_feature_layers)
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            log_compression=args.log_compression,
            skip_connections=args.skip_connections_feat,
            residual_scale=args.residual_scale,
            non_affine_group_norm=args.non_affine_group_norm,
            activation=activation,
        )

    @classmethod
    def build_model(cls, args):

        model = cls(args)
        logger.info(model)
        return model

    def forward(self, src, src_lengths=None, **kwargs):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        """
        logits = self.feature_extractor(src)

        return logits


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers,
        dropout,
        skip_connections,
        residual_scale,
        non_affine_group_norm,
        activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=n_out, affine=not non_affine_group_norm
                ),
                activation,
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.final_fc = nn.dense(dim, 1)

        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.skip_connections and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[..., :: r_tsz // tsz][..., :tsz]
                x = (x + residual) * self.residual_scale

        logits = self.final_fc(x)[:, 0]

        return logits


def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod
