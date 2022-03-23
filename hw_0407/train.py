import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tfrec", type=str, required=True) 
parser.add_argument("--vocab", type=str, required=True) 
parser.add_argument("--batch-size", type=int, required=False, default=64) 
parser.add_argument("--eval-step", type=int, required=False, default=100) 
parser.add_argument("--save-step", type=int, required=False, default=500) 
parser.add_argument("--train-step", type=int, required=False, default=10000) 
parser.add_argument("--begin-lr", type=float, required=False, default=1e-3) 
parser.add_argument("--output", type=str, required=True) 
args = parser.parse_args()

import os
import json

tfrec_args = os.path.join(args.tfrec, "ARGS")
with open(tfrec_args, "r") as f:
  samp_len = json.load(f)["samp_len"]

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
  f.write(json.dumps(vars(args)))
os.chmod(args_file, S_IREAD|S_IRGRP|S_IROTH)

import tensorflow as tf
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

tfrec_list = glob.glob(os.path.join(args.tfrec, "train-*.tfrecord"))
dataset = parse_data.gen_train(tfrec_list, samp_len, batch_size=args.batch_size)

import model
m = model.tdnn(len(vocab))
lr = args.begin_lr
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
  lr, decay_steps=1000, decay_rate=0.96, staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

origins = [val.__spec__.origin for name, val in globals().items() \
  if isinstance(val, types.ModuleType)]
origins = [e for e in origins if e is not None and os.getcwd() in e]

import shutil
for origin in origins + [os.path.abspath(__file__)]:
  shutil.copy(origin, args.output)

@tf.function
def train_step(pcm, ref):
  with tf.GradientTape() as tape:
    loss = m((pcm, ref), training=True)

  grads = tape.gradient(loss, m.trainable_weights)
  opt.apply_gradients(zip(grads, m.trainable_weights))
  return loss

import logging
logger = tf.get_logger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(args.output, "train.log"))
logger.addHandler(fh)

ckpt = tf.train.Checkpoint(m)
for idx, data in enumerate(dataset):
  if idx > args.train_step: break

  loss = (train_step(data["pcm"], data["speaker"]))
  if idx > 0 and idx % args.eval_step == 0:
    logger.info("gstep[{}] loss[{:.2f}] lr[{:.2e}]".format(
      idx, loss, lr_schedule(idx).numpy()))

  if idx > 0 and idx % args.save_step == 0:
    modelname = "model-{}.ckpt".format(idx)
    ckpt.write(os.path.join(args.output, modelname))
    logger.info("model is saved as {}".format(modelname))
