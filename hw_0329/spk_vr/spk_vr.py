import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num-mixture", type=int, required=False, default=32)
parser.add_argument("--random-seed", type=int, required=False, default=1234)
parser.add_argument("--speaker-dirs", nargs='+', required=True)
parser.add_argument("--num-mel", type=int, required=False, default=40)
parser.add_argument("--num-mfcc", type=int, required=False, default=13)
parser.add_argument("--feat-type", type=str, required=False,
  default="mfcc", choices=["logmel", "mfcc"])
parser.add_argument("--eval", action='store_true')
parser.add_argument("--verbose", type=int, required=False,
  default=0, choices=[0, 1, 2])
parser.add_argument("--top-k", type=int, required=False, default=1)
args = parser.parse_args()

import sys
if len(args.speaker_dirs) == 0:
  sys.exit("--speaker-dirs must contain list of directories")

import os
feat_dim = args.num_mel if args.feat_type == "logmel" else args.num_mfcc
feat_base = '{}-{}'.format(args.feat_type, feat_dim) 

umfile = "ubm-{}-{}-{}.mdl".format(args.num_mixture, args.feat_type, feat_dim)
if os.path.isfile(umfile) and not args.eval:
  msg = 'UBM model {} exists. Do you want to proceed?'.format(umfile)
  cont = input("%s (y/N) " % msg).lower() == 'y'
  if not cont: sys.exit(0)

import glob
import pickle
import numpy as np
np.set_printoptions(3, suppress=True)
np.random.seed(args.random_seed)
import sklearn.mixture

if args.eval:
  if not os.path.isfile(umfile):
    sys.exit("UBM model {} is missing. train without --eval option")

  with open(umfile, 'rb') as f:
    um = pickle.load(f)
  sys.exit(0)

X = None
print("Loading data...")
for target_dir in args.speaker_dirs:
  basename = [e.strip() for e in target_dir.split('/') if e.strip() != ''][-1]

  feat_dir = os.path.join(target_dir, feat_base)
  feats = glob.glob(os.path.join(feat_dir, "*.npy"))
  if len(feats) == 0: continue

  f = np.concatenate([np.load(feat) for feat in feats], -1)
  if X is None:
    X = np.transpose(f)
  else:
    X = np.concatenate([X, np.transpose(f)], axis=0)

print("Training model...")
um = sklearn.mixture.GaussianMixture(n_components=args.num_mixture, random_state=0)
um.fit(X)

with open(umfile, "wb") as f:
  pickle.dump(um, f)
