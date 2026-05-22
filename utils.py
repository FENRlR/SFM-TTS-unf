import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      logger.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  logger.info("Loaded checkpoint '{}' (iteration {})" .format(
    checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration


def load_checkpoint2(checkpoint_path, model, optimizer=None, optimizer2=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  #iteration = checkpoint_dict['iteration']
  iteration_p1 = checkpoint_dict['iteration_p1']
  iteration_p2 = checkpoint_dict['iteration_p2']
  learning_rate = checkpoint_dict['learning_rate']
  #p = checkpoint_dict['p']
  #loss_p2_past = checkpoint_dict['loss_p2_past']
  #ema_loss = checkpoint_dict['ema_loss']
  #ema_loss_prev = checkpoint_dict['ema_loss_prev']
  #stable_count = checkpoint_dict['stable_count']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  if optimizer2 is not None:
    optimizer2.load_state_dict(checkpoint_dict['optimizer2'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      logger.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  #logger.info("Loaded checkpoint '{}' (iteration {})" .format(checkpoint_path, iteration))
  logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration_p1}, {iteration_p2})")
  #return model, optimizer, optimizer2, learning_rate, iteration, iteration_p1, iteration_p2, #p, loss_p2_past, ema_loss, ema_loss_prev, stable_count
  return model, optimizer, optimizer2, learning_rate, iteration_p1, iteration_p2


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  logger.info("Saving model and optimizer state at iteration {} to {}".format(iteration, checkpoint_path))
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)

def save_checkpoint2(model, optimizer, optimizer2, learning_rate, iteration_p1, iteration_p2,
                     #p, loss_p2_past, ema_loss, ema_loss_prev, stable_count,
                     checkpoint_path):
  logger.info(f"Saving model and optimizer state at iteration p1:{iteration_p1}, p2:{iteration_p2} to {checkpoint_path}")
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              #'iteration': iteration,
              'iteration_p1': iteration_p1,
              'iteration_p2': iteration_p2,
              'optimizer': optimizer.state_dict(),
              'optimizer2': optimizer2.state_dict(),
              #"p" : p,
              #"loss_p2_past" : loss_p2_past,
              #"ema_loss" : ema_loss,
              #"ema_loss_prev" : ema_loss_prev,
              #"stable_count" : stable_count,
              'learning_rate': learning_rate}, checkpoint_path)


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x


def latest_checkpoint_path2(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  cap = regex.split("*")[0]

  def dual_key(f):
    f_name = os.path.basename(f)
    nums = f_name.replace(cap, "").replace(".pth", "").split("_")
    num_main = int(nums[0])
    num_sub = int(nums[1])
    return (num_main, num_sub)

  f_list.sort(key=dual_key)
  x = f_list[-1]
  print(x)
  return x


def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                  interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
      xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text


def get_hparams(init=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Model name')

  args = parser.parse_args()
  model_dir = os.path.join("./logs", args.model)

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  config_path = args.config
  config_save_path = os.path.join(model_dir, "config.json")
  if init:
    with open(config_path, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)
  else:
    with open(config_save_path, "r") as f:
      data = f.read()
  config = json.loads(data)

  hparams = HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams = HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams = HParams(**config)
  return hparams


def check_git_hash(model_dir):
  source_dir = os.path.dirname(os.path.realpath(__file__))
  if not os.path.exists(os.path.join(source_dir, ".git")):
    logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
      source_dir
    ))
    return

  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  path = os.path.join(model_dir, "githash")
  if os.path.exists(path):
    saved_hash = open(path).read()
    if saved_hash != cur_hash:
      logger.warn("git hash values are different. {}(saved) != {}(current)".format(
        saved_hash[:8], cur_hash[:8]))
  else:
    open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


#- From Matcha tts
def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
  factor = torch.scalar_tensor(2).pow(num_downsamplings_in_unet)
  length = (length / factor).ceil() * factor
  if not torch.onnx.is_in_onnx_export():
    return length.int().item()
  else:
    return length


def convert_pad_shape(pad_shape):
  inverted_shape = pad_shape[::-1]
  pad_shape = [item for sublist in inverted_shape for item in sublist]
  return pad_shape


def generate_path(duration, mask):
  device = duration.device

  b, t_x, t_y = mask.shape
  cum_duration = torch.cumsum(duration, 1)
  path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

  cum_duration_flat = cum_duration.view(b * t_x)
  path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
  path = path.view(b, t_x, t_y)
  path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
  path = path * mask
  return path


def duration_loss(logw, logw_, lengths):
  loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
  return loss


def normalize(data, mu, std):
  if not isinstance(mu, (float, int)):
    if isinstance(mu, list):
      mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
    elif isinstance(mu, torch.Tensor):
      mu = mu.to(data.device)
    elif isinstance(mu, np.ndarray):
      mu = torch.from_numpy(mu).to(data.device)
    mu = mu.unsqueeze(-1)

  if not isinstance(std, (float, int)):
    if isinstance(std, list):
      std = torch.tensor(std, dtype=data.dtype, device=data.device)
    elif isinstance(std, torch.Tensor):
      std = std.to(data.device)
    elif isinstance(std, np.ndarray):
      std = torch.from_numpy(std).to(data.device)
    std = std.unsqueeze(-1)

  return (data - mu) / std


def denormalize(data, mu, std):
  if not isinstance(mu, float):
    if isinstance(mu, list):
      mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
    elif isinstance(mu, torch.Tensor):
      mu = mu.to(data.device)
    elif isinstance(mu, np.ndarray):
      mu = torch.from_numpy(mu).to(data.device)
    mu = mu.unsqueeze(-1)

  if not isinstance(std, float):
    if isinstance(std, list):
      std = torch.tensor(std, dtype=data.dtype, device=data.device)
    elif isinstance(std, torch.Tensor):
      std = std.to(data.device)
    elif isinstance(std, np.ndarray):
      std = torch.from_numpy(std).to(data.device)
    std = std.unsqueeze(-1)

  return data * std + mu


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
