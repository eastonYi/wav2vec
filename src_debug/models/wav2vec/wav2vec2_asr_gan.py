import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import checkpoint_utils, utils
import tasks

from models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer
from .wav2vec2_asr import Wav2VecCtc, Wav2VecEncoder, add_common_args, base_architecture
from .wav2vec2 import ConvFeatureExtractionModel


@register_model("wav2vec_ctc_gan")
class Wav2VecCTC_GAN(Wav2VecCtc):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        add_common_args(parser)
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
        w2v_encoder = Wav2VecEncoder(args, task.target_dictionary)
        discriminator = CLM(args, task.target_dictionary)

        return cls(w2v_encoder, discriminator, args)

    def forward(self, x, len_x, x_un, len_x_un, text, len_text):
        # supervise
        logits = self.w2v_encoder(x, len_x)

        # neg score
        logits_ctc_G = self.w2v_encoder(x_un, len_x_un)
        logits_G, len_decode_G = utils.ctc_shrink(logits_ctc_G)
        probs_G = F.softmax(logits_G, -1)
        score_neg = self.D(probs_G)

        # pos score
        feature_text = F.one_hot(text, self.embed_dim)
        score_pos = self.D(feature_text, len_text, reuse=True)

        gp = 1.0 * self.D.gradient_penalty(
            real=feature_text[: len(logits_G)],
            fake=probs_G,
            len_inputs=len_decode_G)

        result = {'logits': logits, 'score_neg': score_neg, 'score_pos': score_pos, 'gp': gp}

        return result


class CLM(FairseqEncoder):

    def __init__(self, args, tgt_dict):
        super().__init__(None)
        self.tgt_dict = tgt_dict

        feature_enc_layers = eval(args.conv_feature_layers)
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=args.extractor_mode,
            conv_bias=args.conv_bias,
        )
        self.proj = Linear(len(tgt_dict), 1)

    def forward(self, source, padding_mask=None):

        if padding_mask is not None:
            source *= padding_mask
        x = self.feature_extractor(source)

        logits = self.proj(x)[:, 0]

        return {
            "score": logits  # B
        }

    def gradient_penalty(self, real_data, fake_data):
        device = real_data.device
        B = real_data.size(0)
        alpha = torch.rand(B, 1, 1).to(device)
        alpha = alpha.expand(real_data.size())

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self(interpolates)

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
