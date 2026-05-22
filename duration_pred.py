import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack
from fm_decoder import SinusoidalPosEmb, TimestepEmbedding
import modules
from modules import LayerNorm, LayerNorm_legacy


class DurationPredictorNetworkWithTimeStep(nn.Module): # from Matcha, but with some modifications
    """
    Similar architecture but with a time embedding support
    https://www.isca-archive.org/interspeech_2024/mehta24b_interspeech.pdf
    """
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.time_embeddings = SinusoidalPosEmb(filter_channels)
        self.time_mlp = TimestepEmbedding(
            in_channels=filter_channels,
            time_embed_dim=filter_channels,
            act_fn="silu",
        )

        self.proj_pre = nn.Conv1d(in_channels+1, in_channels, 1)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm_legacy(filter_channels)
        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm_legacy(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, enc_outputs, t, g=None):
        t = self.time_embeddings(t)
        t = self.time_mlp(t).unsqueeze(-1)

        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)

        x = pack([x, enc_outputs], "b * t")[0]
        x = self.proj_pre(x)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = x + t
        x = self.norm_1(x)
        x = self.drop(x)

        #x = self.conv_2(x * x_mask)
        x1 = self.conv_2(x * x_mask)
        x1 = torch.relu(x1)
        x1 = x1 + t
        x1 = self.norm_2(x1)
        x1 = self.drop(x1)
        x = x + x1 #-!

        x = self.proj(x * x_mask)

        return x * x_mask


class FlowMatchingDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, sigma_min=1e-4, n_steps=10, gin_channels=0):
        super().__init__()
        self.estimator = DurationPredictorNetworkWithTimeStep(
            in_channels,
            filter_channels,
            kernel_size,
            p_dropout,
            gin_channels
        )
        self.sigma_min = sigma_min
        self.n_steps = n_steps

    @torch.inference_mode()
    def forward(self, enc_outputs, mask, g=None, **kwargs):
        b, _, t = enc_outputs.shape
        z = torch.randn((b, 1, t), device=enc_outputs.device, dtype=enc_outputs.dtype) * kwargs['temperature']
        t_span = torch.linspace(0, 1, self.n_steps + 1, device=enc_outputs.device)
        return self.solve_midpoint(z, t_span=t_span, mu=enc_outputs, mask=mask)
        #return self.solve_euler(z, t_span=t_span, enc_outputs=enc_outputs, mask=mask)

    def solve_euler(self, x, t_span, enc_outputs, mask, g=None):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated, shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder, shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask, shape: (batch_size, 1, mel_timesteps)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, enc_outputs, t, g)
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x

    def solve_midpoint(self, x, t_span, mu, mask, g=None): #-!
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, g)
            dphi_dt_2 = self.estimator(x + dt * 0.5 * dphi_dt, mask, mu, t + dt * 0.5, g)
            x = x + dt * dphi_dt_2
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x

    def compute_loss(self, x1, enc_outputs, mask, g=None): # logw_, x
        """
        Args:
            x1 (torch.Tensor): Target, shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask, shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder, shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None., shape: (batch_size, spk_emb_dim)
        Returns:
            loss: conditional flow matching loss
            y: conditional flow, shape: (batch_size, n_feats, mel_timesteps)
        """
        enc_outputs = enc_outputs.detach()  # don't update encoder from the duration predictor
        b, _, t = enc_outputs.shape
        t = torch.rand([b, 1, 1], device=enc_outputs.device, dtype=enc_outputs.dtype) # random timestep
        z = torch.randn_like(x1) # sample noise p(x_0)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        loss = F.mse_loss(self.estimator(y, mask, enc_outputs, t.squeeze(), g), u, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss


#-! legacy
class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0, **kwargs):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm_legacy(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm_legacy(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None, **kwargs):
        x = torch.detach(x)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask