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

import librosa
import matplotlib.pyplot as plt

import json
import math

import requests
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import langdetect

from scipy.io.wavfile import write
import re
from scipy import signal

import numpy as np
#import pyaudio

#'''
from phonemizer.backend.espeak.wrapper import EspeakWrapper
_ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
EspeakWrapper.set_library(_ESPEAK_LIBRARY)
#'''

# - paths
path_to_config = ""# "put_your_config_path_here" # path to .json
path_to_model = ""#"put_your_model_path_here" # path to G_xxxx.pth


#- text input
input = "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent."


# check device
if torch.cuda.is_available() is True:
    device = "cuda:0"
else:
    device = "cpu"

hps = utils.get_hparams_from_file(path_to_config)

net_g = SynthesizerTrn(
    len(symbols),
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers, #- >0 for multi speaker
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint(path_to_model, net_g)


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def langdetector(text):  # from PolyLangVITS
    try:
        lang = langdetect.detect(text).lower()
        if lang == 'ko':
            return f'[KO]{text}[KO]'
        elif lang == 'ja':
            return f'[JA]{text}[JA]'
        elif lang == 'en':
            return f'[EN]{text}[EN]'
        elif lang == 'zh-cn':
            return f'[ZH]{text}[ZH]'
        else:
            return text
    except Exception as e:
        return text


speed = 1
sid = 0

sol="euler"
n_steps=2

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)


def vcss(inputstr): # single
    fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
    #fltstr = langdetector(fltstr) #- optional for cjke/cjks type cleaners
    stn_tst = get_text(fltstr, hps)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16 if hps.train.fp16_run is True else torch.float32):
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1/speed, sol=sol, steps=n_steps)[0][0].data.cpu().float().numpy()
    write(f'./{output_dir}/output_{sid}.wav', hps.data.sampling_rate, audio)
    print(f'./{output_dir}/output_{sid}.wav Generated!')


def vcms(inputstr, sid): # multi
    fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
    #fltstr = langdetector(fltstr) #- optional for cjke/cjks type cleaners
    stn_tst = get_text(fltstr, hps)

    sid_ = torch.LongTensor([sid]).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16 if hps.train.fp16_run is True else torch.float32):
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid_, noise_scale=.667, noise_scale_w=0.8, length_scale=1/speed, sol=sol, steps=n_steps)[0][0].data.cpu().float().numpy()
    write(f'{output_dir}/{sid}.wav', hps.data.sampling_rate, audio)
    print(f'{output_dir}/{sid}.wav Generated!')


if hps.data.n_speakers > 1:
    vcms(input, sid)
else:
    vcss(input)

