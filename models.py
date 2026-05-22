import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions
from cfm import CFM

try:
    import monotonic_align
except:
    pass
import S_monotonic_align as sma

try:  # -! triton
    import S_monotonic_align_Triton as smat
except:
    pass

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
import utils
import duration_pred

import datetime as dt

from pqmf import PQMF

import zipformer as zf

from hifigan.config import v1
from hifigan.denoiser import Denoiser
from hifigan.env import AttrDict
from hifigan.models import Generator as HiFiGAN


class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels, # inter
                 hidden_channels, # hidden
                 filter_channels, # filter
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 gin_channels=0,
                 enc_pre=False,
                 n_spks=1,
                 ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.n_spks = n_spks

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        self.enc_prenet = enc_pre

        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        if self.enc_prenet:
            self.prenet = attentions.ConvReluNorm(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )
        else:
            self.prenet = lambda x, x_mask: x

        self.encoder = attentions.Encoder(
            hidden_channels + (gin_channels if n_spks > 1 else 0),
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )

        self.n_feats = out_channels #80
        self.proj = nn.Conv1d(hidden_channels + (gin_channels if n_spks > 1 else 0), self.n_feats, 1)

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.prenet(x, x_mask)
        if self.n_spks > 1:
            x = torch.cat([x, g.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)

        x = self.encoder(x * x_mask, x_lengths, x_mask, g=g)
        stats = self.proj(x) * x_mask
        return stats, x, x_mask # mu_x, x, x_mask = self.enc_p(x, x_lengths, g=g)


class Generator(torch.nn.Module):
    def __init__(self, vocoder_path):
        super(Generator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.HIFIGAN_CHECKPOINT = vocoder_path #get_user_data_dir() / "hifigan_T2_v1"
        self.vocoder = self.load_vocoder(self.HIFIGAN_CHECKPOINT)
        self.denoiser = Denoiser(self.vocoder, mode='zeros')


    def load_vocoder(self, checkpoint_path):
        h = AttrDict(v1)
        hifigan = HiFiGAN(h).to(self.device)
        hifigan.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['generator'])
        _ = hifigan.eval()
        hifigan.remove_weight_norm()
        return hifigan

    @torch.inference_mode()
    def forward(self, mel, g=None):
        audio = self.vocoder(mel).clamp(-1, 1) # before dn : torch.Size([1, 1, 85504])
        audio = self.denoiser(audio.squeeze(0), strength=0.00025).cpu() # torch.Size([1, 85504])
        return audio#.unsqueeze(1)


class SynthesizerTrn(nn.Module):
    """
    'Model'
    """
    def __init__(self,
                 n_vocab,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 n_speakers=0,
                 gin_channels=0,
                 dp_type='dp',
                 **kwargs):

        super().__init__()
        self.n_vocab = n_vocab
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.segment_size = segment_size

        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_spk_conditioned_encoder = kwargs.get("use_spk_conditioned_encoder", False)
        self.dp_type = dp_type
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)

        self.current_mas_noise_scale = self.mas_noise_scale_initial
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        else:
            self.enc_gin_channels = 0
        self.enc_pre = kwargs.get("enc_pre", False)
        self.enc_p = TextEncoder(n_vocab,
                                 inter_channels,
                                 hidden_channels,
                                 filter_channels,
                                 n_heads,
                                 n_layers,
                                 kernel_size,
                                 p_dropout,
                                 gin_channels=self.enc_gin_channels,
                                 enc_pre=self.enc_pre,
                                 n_spks=n_speakers,
                                 )

        self.dec = Generator(vocoder_path=kwargs.get("vocoder_path", "UNIVERSAL_V1/g_02500000"))

        if dp_type == 'fmdp':
            self.dp = duration_pred.FlowMatchingDurationPredictor(hidden_channels + (gin_channels if n_speakers > 1 else 0), 256, 3, 0.5, sigma_min=1e-4, n_steps=10, gin_channels=gin_channels)
        elif dp_type == 'sdp':
            self.dp = duration_pred.StochasticDurationPredictor(hidden_channels + (gin_channels if n_speakers > 1 else 0), 192, 3, 0.5, 4, gin_channels=gin_channels)
        else:
            self.dp = duration_pred.DurationPredictor(hidden_channels + (gin_channels if n_speakers > 1 else 0), 256, 3, 0.5, gin_channels=gin_channels)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

        # - options for MAS : "sma_v1", "sma_v2", "sma_triton", "ma"
        self.monotonic_align = kwargs.get("monotonic_align", "ma").lower()
        self.n_feats = inter_channels #80
        self.fmatch = CFM(
            in_channels=2 * self.n_feats,
            out_channel=self.n_feats,
            cfm_params="pass",
            n_spks=n_speakers,
            spk_emb_dim=gin_channels,
        )
        self.prior_loss = True

    def MAS(self, neg_cent, attn_mask):
        if self.use_noise_scaled_mas:
            epsilon = torch.std(neg_cent) * torch.randn_like(neg_cent) * self.current_mas_noise_scale
            neg_cent = neg_cent + epsilon

        if self.monotonic_align == "sma_triton":
            attn = smat.maximum_path(neg_cent, attn_mask.squeeze(1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(1).detach()
        elif self.monotonic_align == "sma_v1":
            attn = sma.maximum_path1(neg_cent, attn_mask.squeeze(1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(1).detach()
        elif self.monotonic_align == "sma_v2":
            attn = sma.maximum_path2(neg_cent, attn_mask.squeeze(1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(1).detach()
        else:
            attn = monotonic_align.maximum_path(neg_cent.transpose(-1, -2), attn_mask.squeeze(1).transpose(-1, -2)).unsqueeze(1).detach()

        return attn.detach()


    def forward(self, x, x_lengths, y, y_lengths, mas_attn = None, sid=None):
        if self.n_speakers > 0:
            g = self.emb_g(sid)#.unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        mu_x, x, x_mask = self.enc_p(x, x_lengths, g=g)
        y_max_length = y.shape[-1]
        y_mask = utils.sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)

        with torch.no_grad():
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            const = torch.sum(-0.5 * math.log(2 * math.pi) - torch.zeros_like(mu_x), [1]).unsqueeze(-1)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)

            neg_cent = y_square - y_mu_double + mu_square + const
            attn = self.MAS(neg_cent, attn_mask)

        w = attn.sum(2)
        logw_ = torch.log(w + 1e-8) * x_mask
        # logw_ = torch.log(w + 1e-6) * x_mask

        if self.dp_type == 'fmdp':
            logw = None
            l_length = None
            fmdp_loss = self.dp.compute_loss(logw_, x, x_mask, g=g)
        elif self.dp_type == 'sdp':
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=1.)
            l_length = self.dp(x, x_mask, w, g=g) / torch.sum(x_mask)
            fmdp_loss = None
        else:
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)
            fmdp_loss = None

        mu_y = torch.matmul(attn.squeeze(1), mu_x.transpose(1, 2)).transpose(1, 2)

        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        diff_loss, _ = self.fmatch.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=g)

        return diff_loss, prior_loss, l_length, attn, x_mask, y_mask, (x, logw, logw_), fmdp_loss


    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None, sol="euler", steps=10):
        if self.n_speakers > 0:
            g = self.emb_g(sid)#.unsqueeze(-1)
        else:
            g = None

        mu_x, x, x_mask = self.enc_p(x, x_lengths, g=g)

        """
        if self.dp_type == 'sdp':
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        """
        logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w, temperature=1)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()

        y_max_length = y_lengths.max() # ->  y_max_length = y_lengths.max()
        y_max_length_ = utils.fix_len_compatibility(y_max_length)

        y_mask = commons.sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        mu_y = torch.matmul(attn.squeeze(1), mu_x.transpose(1, 2)).transpose(1, 2)
        z = self.fmatch(mu_y, y_mask, steps, noise_scale, g, sol)

        o = self.dec((z * y_mask)[:, :, :y_max_length], g=g)

        return o, attn, y_mask, z