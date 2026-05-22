import os
import logging
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["TORCH_CPP_LOG_LEVEL"] = "WARNING"
os.environ["NUMBA_DEBUG"] = "0"
os.environ["NUMBA_LOG_LEVEL"] = "WARNING"
os.environ["NUMBA_WARNINGS"] = "0"
logging.basicConfig(level=logging.INFO)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.core").setLevel(logging.WARNING)
logging.getLogger("numba.cuda").setLevel(logging.WARNING)

import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
# import torch.multiprocessing as mp
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import tqdm

import models
from pqmf import PQMF
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    TextAudioLoader,
    TextAudioCollate,
    DistributedBucketSampler
)
from models import (
    SynthesizerTrn,
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols


ddp_find_unused_parameters = True
# torch.autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(False)

torch.backends.cudnn.benchmark = True
global_step = 0

num_workers = 1


addr = "localhost"
port = 6060
backop = ''
if os.name == 'nt':
    backop = 'gloo'
else:
    backop = 'nccl'

"""
dist.init_process_group(
    backend=f"cpu:{backop},cuda:{backop}",
    rank=0,
    world_size=1,
    init_method=f"tcp://{addr}:{port}?use_libuv=0",
)
dist.destroy_process_group()
"""

def format_params(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)

def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = '6666'

    hps = utils.get_hparams()
    # mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
    run(0, 1, hps)


def run(rank, n_gpus, hps):
    global global_step

    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    """
    if os.name == 'nt':
        dist.init_process_group(backend='gloo', init_method='env://?use_libuv=0', world_size=n_gpus, rank=rank)
    else:
        dist.init_process_group(backend='nccl', init_method='env://?use_libuv=0', world_size=n_gpus, rank=rank)
    """

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    if hps.data.n_speakers == 0:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
        collate_fn = TextAudioCollate()
        if rank == 0:
            eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
            eval_loader = DataLoader(eval_dataset, num_workers=num_workers, shuffle=False,
                                     batch_size=hps.train.batch_size, pin_memory=True,
                                     drop_last=False, collate_fn=collate_fn)
    else:
        train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
        collate_fn = TextAudioSpeakerCollate()
        if rank == 0:
            eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
            eval_loader = DataLoader(eval_dataset, num_workers=num_workers, shuffle=False,
                                     batch_size=hps.train.batch_size, pin_memory=True,
                                     drop_last=False, collate_fn=collate_fn)

    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)

    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
                              collate_fn=collate_fn, batch_sampler=train_sampler)

    if "use_spk_conditioned_encoder" in hps.model.keys() and hps.model.use_spk_conditioned_encoder == True:
        if hps.data.n_speakers == 0:
            raise ValueError("n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model")
        use_spk_conditioned_encoder = True
    else:
        print("Using normal encoder")
        use_spk_conditioned_encoder = False

    if "use_noise_scaled_mas" in hps.model.keys() and hps.model.use_noise_scaled_mas == True:
        print("Using noise scaled MAS")
        use_noise_scaled_mas = True
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS")
        use_noise_scaled_mas = False
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    epochs = int(hps.train.epochs)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model).cuda(rank)

    mdec_id = [id(i) for i in net_g.dec.parameters()]
    exdec = [i for i in net_g.parameters() if id(i) not in mdec_id]

    optim_g = torch.optim.AdamW(
        exdec,
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    #net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=ddp_find_unused_parameters)
    net_g = net_g.cuda()

    try:
        print("load case 1")
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        # return model, optimizer, learning_rate, iteration
        epoch_str = int(epoch_str)

        if hps.train.steps_from_filename is True:
            # utils.latest_checkpoint_path2(hps.model_dir, "G_*.pth")
            fname = os.path.basename(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"))  # ex) G_123_456.pth
            global_step = fname.replace("G_", "").replace(".pth", "")
        else:
            global_step = (epoch_str-1) * len(train_loader)  # len(train_loader) -> 395

    except:
        print("load case 2")
        if hps.train.steps_from_filename is True:  # case for pretrained file
            epoch_str = int(epoch_str)
            fname = os.path.basename(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"))
            global_step = fname.replace("G_", "").replace(".pth", "")
        else:
            epoch_str = 1
            global_step = 0


    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    # scaler = GradScaler(enabled=hps.train.fp16_run)
    if hps.train.fp16_run:
        scaler = GradScaler(init_scale=2.0 ** 8, enabled=hps.train.fp16_run)
    else:
        scaler = GradScaler(enabled=hps.train.fp16_run)


    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g],
                               [optim_g],
                               [scheduler_g], scaler,
                               [train_loader, eval_loader],
                               logger, [writer, writer_eval], phase=0)
        else:
            train_and_evaluate(rank, epoch, hps, [net_g],
                               [optim_g],
                               [scheduler_g], scaler,
                               [train_loader, None], None,
                               None, phase=0)
        scheduler_g.step()

        print(f"global_step at epoch{epoch}: {global_step}")
        if global_step == int(hps.train.cutoff_step) and int(hps.train.cutoff_step)>0:
            print(f"cutoff activated {global_step} == {int(hps.train.cutoff_step)}")
            break


    print(f"global_step final: {global_step}")


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, phase=0):
    net_g, = nets
    optim_g, = optims
    scheduler_g, = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()

    ld1, ld2, ld3 = hps.train.lambda_w # "lambda": [1.0, 1.0, 1.0]

    if rank == 0:
        loader = tqdm.tqdm(train_loader, desc='Loading train data')
    else:
        loader = train_loader

    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, *speakers) in enumerate(loader):
        speakers = speakers[0] if speakers else None

        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

        if speakers is not None:
            speakers = speakers.cuda(rank, non_blocking=True)

        if net_g.use_noise_scaled_mas:
            current_mas_noise_scale = net_g.mas_noise_scale_initial - net_g.noise_scale_delta * global_step  # -!
            net_g.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

        with (autocast(enabled=hps.train.fp16_run)):
            diff_loss, prior_loss, l_length, attn, x_mask, y_mask, (hidden_x, logw, logw_), fmdp_loss = net_g(x, x_lengths, spec, spec_lengths, None, speakers)
            with autocast(enabled=False):
                loss = ld3*diff_loss + ld2*prior_loss
                if hps.model.dp_type == "fmdp":
                    loss_dur = fmdp_loss
                else:
                    loss_dur = torch.sum(l_length.float())
                loss += ld1*loss_dur

        optim_g.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [diff_loss, loss_dur]

                logger.info('Train Epoch: {} [{:.0f}%]'.format(epoch, 100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss, "learning_rate": lr, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/diff": diff_loss, "loss/g/dur": loss_dur})

                image_dict = {
                    "all/attn": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy())
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict)


            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                save_removal(hps.train.save_policy, hps.model_dir, global_step, hps.train.eval_interval)

                #- save
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, f"G_{global_step}.pth"))


        if global_step == int(hps.train.cutoff_step) and int(hps.train.cutoff_step)>0:
            print(f"cutoff activated in func")
            return  # break
        else:
            global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def save_removal(save_policy, model_dir, step, eval_interval):
    if save_policy.startswith("last"):
        pidx = int(save_policy.split("last", 2)[-1])
        # print(f"pidx : {pidx}")
        fs = step - pidx * eval_interval
        prev_g = os.path.join(model_dir, f"G_{fs}.pth")
        if os.path.exists(prev_g):
            os.remove(prev_g)


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        with autocast(enabled=hps.train.fp16_run):
            if hps.data.n_speakers == 0:
                for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
                    x, x_lengths = x.cuda(0), x_lengths.cuda(0)
                    spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
                    y, y_lengths = y.cuda(0), y_lengths.cuda(0)

                    x = x[:1]
                    x_lengths = x_lengths[:1]
                    spec = spec[:1]
                    spec_lengths = spec_lengths[:1]
                    y = y[:1]
                    y_lengths = y_lengths[:1]
                    break

                y_hat, attn, mask, *_ = generator.infer(x, x_lengths, max_len=1000)
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
            else:
                for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
                    x, x_lengths = x.cuda(0), x_lengths.cuda(0)
                    spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
                    y, y_lengths = y.cuda(0), y_lengths.cuda(0)
                    speakers = speakers.cuda(0)

                    x = x[:1]
                    x_lengths = x_lengths[:1]
                    spec = spec[:1]
                    spec_lengths = spec_lengths[:1]
                    y = y[:1]
                    y_lengths = y_lengths[:1]
                    speakers = speakers[:1]
                    break
                y_hat, attn, mask, *_ = generator.infer(x, x_lengths, speakers, max_len=1000)
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

        mel = spec

        try:
            y_hat_mel = mel_spectrogram_torch(
                y_hat.float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
        except:
            y_hat_mel = torch.zeros_like(mel, device=y_hat.device, dtype=torch.float32)
            print("- Exception from eval")

    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
        "gen/audio": y_hat[0, :y_hat_lengths[0]]
    }

    if global_step == 0:
        image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({"gt/audio": y[0, :, :y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    main()


