import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from tools import checkpoint_utils, utils
import tasks
from typing import List, Tuple

from models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from modules import (
    LayerNorm,
    Fp32GroupNorm,
    TransposeLast,
    Fp32LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer)
from .wav2vec2_asr import Wav2VecCtc, Wav2VecEncoder, add_common_args, base_architecture


@register_model("wav2vec_ctc_gan")
class Wav2VecCTC_GAN(Wav2VecCtc):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        add_common_args(parser)
        parser.add_argument(
            "--generator-path",
            type=str,
            help="the generator mdoel path (asr model)",
        )
        parser.add_argument(
            "--conv-feature-layers",
            type=str,
            metavar="EXPR",
            help="convolutional feature extraction layers [(dim, kernel_size, stride), ...]",
        )
        parser.add_argument(
            "--conv-bias", action="store_true", help="include bias in conv encoder"
        )
        parser.add_argument(
            "--extractor-mode",
            choices=["default", "layer_norm"],
            help='''mode for feature extractor. default has a single group norm
            with d groups in the first conv block, whereas layer_norm has
            layer norms in every block (meant to use with --normalize)''',
        )

    def __init__(self, w2v_encoder, discriminator, args):
        super().__init__(w2v_encoder, args)
        self.D = discriminator
        self.embed_dim = len(discriminator.tgt_dict)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        ctc_gan_architecture(args)
        if args.generator_path:
            generator, _ = cls.load_generator(args.generator_path, args, task=task)
        discriminator = CLM(args, task.target_dictionary)

        return cls(generator, discriminator, args)

    def forward(self, **source):
        x = source['source']
        padding_mask = source['padding_mask']
        _x = source['_source']
        _padding_mask = source['_padding_mask']
        text = source['text']
        len_text = source['len_text']

        # supervise ['encoder_out', 'encoder_padding_mask', 'padding_mask']
        _res = self.w2v_encoder(source=_x, padding_mask=_padding_mask) #

        # neg score
        res = self.w2v_encoder(source=x, padding_mask=padding_mask)
        logits_ctc = res['encoder_out'].permute(1,0,2)
        logits_G, len_decode_G = utils.ctc_shrink(
            logits_ctc, pad=self.D.tgt_dict.pad(), blk=self.D.tgt_dict.bos())
        probs_G = F.softmax(logits_G, -1)
        mask = utils.sequence_mask(len_decode_G).unsqueeze(-1).repeat(1, 1, probs_G.size(-1))
        score_neg = self.D(probs_G, mask)

        # pos score
        feature_text = F.one_hot(text.long(), self.embed_dim).float()
        mask = utils.sequence_mask(len_text).unsqueeze(-1).repeat(1, 1, probs_G.size(-1))
        score_pos = self.D(feature_text, mask)

        min_len = min(feature_text.size(1), probs_G.size(1))
        gp = 1.0 * self.D.gradient_penalty(
            real_data=feature_text[:, :min_len, :],
            fake_data=probs_G[:, :min_len, :])

        result = {'_res': _res,
                  'score_neg': score_neg,
                  'score_pos': score_pos,
                  'gp': gp}

        return result

    @classmethod
    def load_generator(cls, filename, args, arg_overrides=None, task=None):

        if arg_overrides is None:
            arg_overrides = {}

        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))
        state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)
        args = state["args"]
        model = task.build_model(args)
        model.load_state_dict(state["model"], strict=True)

        return model, args

class CLM(FairseqEncoder):

    def __init__(self, args, tgt_dict):
        super().__init__(None)
        self.tgt_dict = tgt_dict

        feature_enc_layers = eval(args.conv_feature_layers)
        self.feature_extractor = Conv2DFeatureExtractionModel(
            dim_input=len(tgt_dict),
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=args.extractor_mode,
            conv_bias=args.conv_bias,
        )
        self.proj = Linear(feature_enc_layers[-1][0], 1)

    def forward(self, source, padding_mask=None):

        if padding_mask is not None:
            source *= padding_mask

        # BxTxC -> BxCxT
        source = source.permute(0,2,1)
        x = self.feature_extractor(source)
        x = x.permute(0,2,1)
        logits = self.proj(x)[:, 0]

        return {
            "score": logits,  # B
        }

    def gradient_penalty(self, real_data, fake_data):
        device = real_data.device
        B = real_data.size(0)
        alpha = torch.rand(B, 1, 1).to(device)
        alpha = alpha.expand(real_data.size())

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self(interpolates)["score"]
        # TODO: Make ConvBackward diffentiable
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


class Conv2DFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        dim_input: int,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()
        assert mode in {"default", "layer_norm"}

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
        # BxT -> BxCxT
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise NotImplementedError('error input ndim')
        for conv in self.conv_layers:
            x = conv(x)

        return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("wav2vec_ctc_gan", "wav2vec_ctc_gan")
def ctc_gan_architecture(args):
    base_architecture(args)
    discriminator_architecture(args)


def discriminator_architecture(args):

    conv_feature_layers = "[(512, 10, 5)]"
    conv_feature_layers += " + [(512, 8, 4)]"
    conv_feature_layers += " + [(512, 4, 2)] * 3"
    conv_feature_layers += " + [(512, 1, 1)]"
    args.conv_feature_layers = getattr(args, "conv_feature_layers", conv_feature_layers)

    args.extractor_mode = getattr(args, "extractor_mode", "default")

    args.conv_bias = getattr(args, "conv_bias", False)
