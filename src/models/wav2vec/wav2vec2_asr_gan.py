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

        parser.add_argument('--loss-weights', type=str, default=None,
                            help='weights for additional loss terms (not first one)')

    def __init__(self, w2v_encoder, discriminator, args):
        super().__init__()
        self.w2v_encoder = w2v_encoder
        self.D = discriminator
        self.args = args
        self.embed_dim = args.embed_dim

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

    @staticmethod
    def add_args(parser):
        add_common_args(parser)

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

    def __init__(self, args, tgt_dict):
        self.apply_mask = args.apply_mask

        feature_enc_layers = eval(args.conv_feature_layers)
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=args.extractor_mode,
            conv_bias=args.conv_bias,
        )
        self.proj = Linear(len(tgt_dict), 1)

        super().__init__(None)

        self.num_updates = 0

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
    args.no_pretrained_weights = getattr(args, "no_pretrained_weights", False)
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.final_dropout = getattr(args, "final_dropout", 0)
    args.apply_mask = getattr(args, "apply_mask", False)
    args.dropout = getattr(args, "dropout", 0)
    args.attention_dropout = getattr(args, "attention_dropout", 0)
    args.activation_dropout = getattr(args, "activation_dropout", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.5)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)

    args.freeze_finetune_updates = getattr(args, "freeze_finetune_updates", 0)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0)
    args.layerdrop = getattr(args, "layerdrop", 0.0)

    base_architecture(args)


def discriminator_architecture(args):
    args.no_pretrained_weights = getattr(args, "no_pretrained_weights", False)
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.final_dropout = getattr(args, "final_dropout", 0)
    args.apply_mask = getattr(args, "apply_mask", False)
    args.dropout = getattr(args, "dropout", 0)
    args.attention_dropout = getattr(args, "attention_dropout", 0)
    args.activation_dropout = getattr(args, "activation_dropout", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.5)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)

    args.freeze_finetune_updates = getattr(args, "freeze_finetune_updates", 0)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0)
    args.layerdrop = getattr(args, "layerdrop", 0.0)
