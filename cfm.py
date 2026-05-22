from abc import ABC
import math
import torch
import torch.nn.functional as F
import numpy as np
from fm_decoder import Decoder


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=64,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

        """
        M = 128
        d_min = 1.0 / M
        self.dlist = []
        for i in range(int(math.log2(M))):
            self.dlist.append(1 / (2 ** (i + 1)))
        """

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps=10, temperature=1.0, spks=None, sol="euler"):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)

        if sol=="euler":
            return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks)
        elif sol=="RK4":
            return self.solve_RK4(z, t_span=t_span, mu=mu, mask=mask, spks=spks)
        elif sol=="midpoint":
            return self.solve_midpoint(z, t_span=t_span, mu=mu, mask=mask, spks=spks)
        elif sol=="heun":
            return self.solve_heun(z, t_span=t_span, mu=mu, mask=mask, spks=spks)
        else: # failproof
            return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks)


    def solve_euler(self, x, t_span, mu, mask, spks):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
        """
        # t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        #t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t_span[0]
        sol = []

        dt = 1 / (len(t_span)-1) # 'bulk' timestep
        dt = torch.tensor([dt], device=mu.device, dtype=mu.dtype)

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, dt, spks)
            x = x + dt.item() * dphi_dt
            t = t + dt.item()
            sol.append(x)

            #if step < len(t_span) - 1:
            #    dt = t_span[step + 1] - t

        return sol[-1]


    def solve_midpoint(self, x, t_span, mu, mask, spks):
        """
        Fixed midpoint solver for ODEs.
        """
        t = t_span[0]
        sol = []

        dt = 1 / (len(t_span) - 1)  # 'bulk' timestep
        dt = torch.tensor([dt], device=mu.device, dtype=mu.dtype)

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, dt, spks)
            dphi_dt_2 = self.estimator(x + dt * 0.5 * dphi_dt, mask, mu, t + dt * 0.5, dt, spks)

            x = x + dt.item() * dphi_dt_2
            t = t + dt.item()

            sol.append(x)
            #if step < len(t_span) - 1:
            #    dt = t_span[step + 1] - t

        return sol[-1]


    def solve_heun(self, x, t_span, mu, mask, spks):
        """
        Fixed heun solver for ODEs.
        """
        t = t_span[0]
        sol = []

        dt = 1 / (len(t_span) - 1)  # 'bulk' timestep
        dt = torch.tensor([dt], device=mu.device, dtype=mu.dtype)

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, dt, spks)
            dphi_dt_2 = self.estimator(x + dt * dphi_dt, mask, mu, t + dt, dt, spks)

            x = x + dt.item() * 0.5 * (dphi_dt + dphi_dt_2)
            t = t + dt.item()

            sol.append(x)
            #if step < len(t_span) - 1:
            #    dt = t_span[step + 1] - t

        return sol[-1]


    def solve_RK4(self, x, t_span, mu, mask, spks):
        """
        Fixed RK4 solver for ODEs.
        """
        t = t_span[0]
        sol = []

        dt = 1 / (len(t_span) - 1)  # 'bulk' timestep
        dt = torch.tensor([dt], device=mu.device, dtype=mu.dtype)

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, dt, spks)
            dphi_dt_2 = self.estimator(x + 0.5 * dt * dphi_dt, mask, mu, t + 0.5 * dt, dt, spks)
            dphi_dt_3 = self.estimator(x + 0.5 * dt * dphi_dt_2, mask, mu, t + 0.5 * dt, dt, spks)
            dphi_dt_4 = self.estimator(x + dt * dphi_dt_3, mask, mu, t + dt, dt, spks)

            x = x + dt.item() * (dphi_dt + 2*dphi_dt_2 + 2*dphi_dt_3 + dphi_dt_4) / 6
            t = t + dt.item()

            sol.append(x)
            #if step < len(t_span) - 1:
            #    dt = t_span[step + 1] - t

        return sol[-1]


    def solve_RKDP(self, x, t_span, mu, mask, spks): #-! FENRIR HOME INDUSTRIES™
        """
        Adaptive Dormand–Prince solver for FUN (WIP - needs more labor).
        DO NOT TRY THIS.
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        sol = []

        d = 1 / (len(t_span) - 1)
        d = torch.tensor([d], device=mu.device, dtype=mu.dtype)

        dtmin = 1/128
        dtmax = 1/2

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, d, spks)
            dphi_dt_2 = self.estimator(x + 0.2 * dt * dphi_dt, mask, mu, t + 0.2 * dt, d, spks)
            dphi_dt_3 = self.estimator(x + 0.3 * dt * (dphi_dt + 3 * dphi_dt_2) * 0.25, mask, mu, t + 0.3 * dt, d, spks)
            dphi_dt_4 = self.estimator(x + 0.8*dt * (11*dphi_dt-42*dphi_dt_2+40*dphi_dt_3)/9, mask, mu, t + 0.8*dt, d, spks)
            dphi_dt_5 = self.estimator(x + 8/9*dt * (4843*dphi_dt-19020*dphi_dt_2+16112*dphi_dt_3-477*dphi_dt_4)/1458, mask, mu, t + 8/9*dt, d, spks)
            dphi_dt_6 = self.estimator(x + dt * (477901*dphi_dt-1806240*dphi_dt_2+1495424*dphi_dt_3+46746*dphi_dt_4-45927*dphi_dt_5)/167904, mask, mu, t + dt, d, spks)
            z = x + dt * (12985*dphi_dt+64000*dphi_dt_3+92750*dphi_dt_4-45927*dphi_dt_5+18656*dphi_dt_6)/142464
            dphi_dt_7 = self.estimator(z, mask, mu, t + dt, dt, spks)
            y = x + dt * (dphi_dt * 1921409 + dphi_dt_3 * 9690880 + dphi_dt_4 * 13122270 - dphi_dt_5 * 5802111 + dphi_dt_6 * 1902912 + dphi_dt_7 * 534240)/21369600
            abeps = 1e-5
            alpha = (dt*abeps/(2*abs(y-z)))**0.25
            if alpha > 1 or dt == dtmin:
                x = z
                t = t + dt
            if 0.9 * alpha < 0.5:
                dt = 0.5 * dt
            elif 0.9 * alpha > 2:
                dt = 2 * dt
            else:
                dt = 0.9 * alpha * dt
            if dt < dtmin:
                dt = dtmin
            elif dt > dtmax:
                dt = dtmax
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1]


    def loss_fm(self, x1, mask, mu, spks=None):
        b, _, l = mu.shape
        device = mu.device
        t = torch.rand([b, 1, 1], device=device, dtype=mu.dtype)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1  # Xt
        u = x1 - (1 - self.sigma_min) * z  # TGT

        d = torch.zeros([b], device=device, dtype=mu.dtype)
        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), d, spks), u, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y

    def loss_sc(self, x1, mask, mu, spks=None):
        b, _, l = mu.shape
        device = mu.device

        list_d = torch.tensor(
            [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128],  # self.dlist,
            device=device, dtype=mu.dtype)
        d_min = list_d.min()

        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype) * (1.0 - 2 * d_min)
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1  # Xt

        d1_list = []
        d2_list = []
        for i in range(b):  # such that t + d1 + d2 <= 1
            #- d1
            last1 = 1.0 - t[i].item() - d_min
            list_d1 = list_d[list_d <= last1]

            idx1 = torch.randint(0, len(list_d1), (1,), device=device)
            d1 = list_d1[idx1]

            #- d2
            last2 = 1.0 - t[i].item() - d1
            list_d2 = list_d[list_d <= last2]

            idx2 = torch.randint(0, len(list_d2), (1,), device=device)
            d2 = list_d2[idx2]

            d1_list.append(d1)
            d2_list.append(d2)

        d1 = torch.stack(d1_list).view(b, 1, 1)
        d2 = torch.stack(d2_list).view(b, 1, 1)

        vel1 = self.estimator(y, mask, mu, t.squeeze(), d1.squeeze(), spks)
        mid = y + d1 * vel1
        vel2 = self.estimator(mid, mask, mu, t.squeeze() + d1.squeeze(), d2.squeeze(), spks)

        #final = self.estimator(y, mask, mu, t.squeeze(), (d1+d2).squeeze(), spks)

        tgt = ((d1 * vel1 + d2 * vel2) / (d1 + d2)).detach()
        #tgt = (((d1/(d1 + d2)) * vel1 + (d2/(d1 + d2)) * vel2)).detach()

        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), (d1+d2).squeeze(), spks), tgt)
        #loss = F.mse_loss(final, tgt)

        return loss, y


    def compute_loss(self, x1, mask, mu, spks=None):
        """Computes loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """

        b, _, t = mu.shape
        device = mu.device

        b_sp = b // 4 # split -> 1/4 to sc
        shuf = torch.randperm(b, device=device)

        sc_idx = shuf[:b_sp]
        fm_idx = shuf[b_sp:]

        loss_fm, _ = self.loss_fm(x1[fm_idx], mask[fm_idx], mu[fm_idx], spks[fm_idx] if spks is not None else None)
        loss_sc, _ = self.loss_sc(x1[sc_idx], mask[sc_idx], mu[sc_idx], spks[sc_idx] if spks is not None else None)

        loss = loss_fm + loss_sc
        y = None # leaving it as None for now

        return loss, y


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel)