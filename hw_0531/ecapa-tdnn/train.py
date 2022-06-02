import os
import random
import numpy as np
import tensorflow as tf

seed = 1234
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tfrecs", nargs="+", default=[]) 
parser.add_argument("--weights", nargs="+", default=[]) 
parser.add_argument("--vocab", type=str, required=True) 
parser.add_argument("--batch-size", type=int, required=False, default=64) 
parser.add_argument("--eval-step", type=int, required=False, default=100) 
parser.add_argument("--save-step", type=int, required=False, default=1000) 
parser.add_argument("--val-step", type=int, required=False, default=1000) 
parser.add_argument("--train-step", type=int, required=False, default=100000) 
parser.add_argument("--begin-lr", type=float, required=False, default=1e-4) 
parser.add_argument("--val-lr-update", type=float, required=False, default=3) 
parser.add_argument("--output", type=str, required=True) 
args = parser.parse_args()

if len(args.tfrecs) == 0: sys.exit(0)
assert len(args.tfrecs) == len(args.weights)
weights = [int(e) for e in args.weights]
weights = weights / np.sum(weights)

import json
samp_len = None
mix_lens = []

for tfrec in args.tfrecs:
  tfrec_args = os.path.join(tfrec, "ARGS")
  with open(tfrec_args, "r") as f:
    _json = json.loads(f.readlines()[-1])
    _samp_len = _json["samp_len"]

    _mix_len = 0
    if "mixup_multiplier" in _json:
      _mix_len = _json["mixup_multiplier"]
    
    if samp_len is None: samp_len = _samp_len
    assert samp_len == _samp_len
    mix_lens.append(_mix_len)

vocab = [e.strip() for e in open(args.vocab).readlines()]

import types
import sys
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR

args_file = os.path.join(args.output, "ARGS")
if os.path.isdir(args.output):
  msg = 'directory {} exists. Do you want to proceed?'.format(args.output)
  cont = input("%s (y/N) " % msg).lower() == 'y'
  if not cont: sys.exit(0)
  if os.path.isfile(args_file):
    os.chmod(args_file, S_IWUSR|S_IREAD)

os.makedirs(args.output, exist_ok=True)
with open(args_file, "w") as f:
  f.write(" ".join([sys.executable] + sys.argv))
  f.write("\n")
  f.write(json.dumps(vars(args)))
os.chmod(args_file, S_IREAD|S_IRGRP|S_IROTH)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import parse_data
import glob

datasets = []
for tfrec, mix_len in zip(args.tfrecs, mix_lens):
  tfrec_list = glob.glob(os.path.join(tfrec, "train-*.tfrecord"))
  
  if mix_len == 0:
    dataset = parse_data.gen_train_v1(tfrec_list,
      samp_len, mix_len, len(vocab),
      batch_size=args.batch_size, seed=seed)
  else:
    dataset = parse_data.gen_train_v2(tfrec_list,
      samp_len, mix_len, len(vocab),
      batch_size=args.batch_size, seed=seed)

  datasets.append(dataset)

dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=weights, seed=seed)

lr = tf.Variable(args.begin_lr, trainable=False)
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#   lr, decay_steps=1000, decay_rate=0.96, staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=lr)

import model
m = model.tdnn(len(vocab))

specs = [val.__spec__ for name, val in sys.modules.items() \
  if isinstance(val, types.ModuleType) and not ('_main_' in name)]
origins = [spec.origin for spec in specs if spec is not None]
origins = [e for e in origins if e is not None and os.getcwd() in e]

import shutil
for origin in origins + [os.path.abspath(__file__)]:
  shutil.copy(origin, args.output)

import datetime
logdir = os.path.join(args.output, "logs")
log_writer = tf.summary.create_file_writer(logdir)
log_writer.set_as_default()

@tf.function
def run_step(step, pcm, target, training=True):
  with tf.GradientTape() as tape, log_writer.as_default():
    loss = m((pcm, target), training=training)
    loss = tf.math.reduce_mean(loss)
    tf.summary.scalar("loss", loss, step=step)

  if training:
    grads = tape.gradient(loss, m.trainable_weights)
    grads, _ = tf.clip_by_global_norm(grads, 5.)
    opt.apply_gradients(zip(grads, m.trainable_weights))

  return loss

import logging
logger = tf.get_logger()
logger.setLevel(logging.INFO)

logfile = os.path.join(args.output, "train.log")
if os.path.isfile(logfile): os.remove(logfile)
fh = logging.FileHandler(logfile)
logger.addHandler(fh)

ckpt = tf.train.Checkpoint(model=m)
prev_val_loss = None; stall_cnt = 0

for idx, data in enumerate(dataset):
  if idx > args.train_step: break

  loss = run_step(tf.cast(idx, tf.int64),
    data["pcm"], data["target"])
  log_writer.flush()

  if idx > 0 and idx % args.eval_step == 0:
    logger.info("gstep[{}] loss[{:.2f}] lr[{:.2e}]".format(
      idx, loss, lr.numpy()))

  if idx > 0 and idx % args.save_step == 0:
    modelname = "model-{}.ckpt".format(idx)
    modelpath = os.path.join(args.output, modelname)
    ckpt.write(modelpath)

    optname = "adam-{}-weight".format(idx)
    optpath = os.path.join(args.output, optname)
    np.save(optpath, opt.get_weights())
    
    cfgname = "adam-{}-config".format(idx)
    cfgpath = os.path.join(args.output, cfgname)
    np.save(cfgpath, opt.get_config())

    logger.info("model is saved as {}".format(modelpath))
