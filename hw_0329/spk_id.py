import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num-mixture", type=int, required=False, default=3)
parser.add_argument("--random-seed", type=int, required=False, default=1234)
parser.add_argument("--speaker-dirs", nargs='+', required=True)
parser.add_argument("--num-mel", type=int, required=False, default=40)
parser.add_argument("--num-mfcc", type=int, required=False, default=13)
parser.add_argument("--feat-type", type=str, required=False,
  default="mfcc", choices=["logmel", "mfcc"])
parser.add_argument("--eval", action='store_true')
args = parser.parse_args()

import sys
if len(args.speaker_dirs) == 0:
  sys.exit("--speaker-dirs must contain list of directories")

import os
feat_dim = args.num_mel if args.feat_type == "logmel" else args.num_mfcc
feat_base = '{}-{}'.format(args.feat_type, feat_dim) 

gmfile = "gmms-{}-{}-{}.mdl".format(args.num_mixture, args.feat_type, feat_dim)
if os.path.isfile(gmfile) and not args.eval:
  msg = 'GMM model {} exists. Do you want to proceed?'.format(gmfile)
  cont = input("%s (y/N) " % msg).lower() == 'y'
  if not cont: sys.exit(0)

import glob
import pickle
import numpy as np
np.set_printoptions(3, suppress=True)
np.random.seed(args.random_seed)
import sklearn.mixture

if args.eval:
  if not os.path.isfile(gmfile):
    sys.exit("GMM model {} is missing. run without --eval option")

  with open(gmfile, 'rb') as f:
    gms = pickle.load(f)

  for target_dir in args.speaker_dirs:
    test_dir = os.path.join(target_dir, 'test', feat_base)
    feats = glob.glob(os.path.join(test_dir, '*.npy'))
    if len(feats) == 0:
      print("dir[{}] contains no npy".format(test_dir))
      continue
   
    for feat in feats:
      f = np.transpose(feat)
      for gm_name, gm in gms:
        print(gm.predict(f))

  sys.exit(0)

gms = []
for target_dir in args.speaker_dirs:
  basename = [e.strip() for e in target_dir.split('/') if e.strip() != ''][-1]
  print("processing spk[{}]...".format(basename))

  feat_dir = os.path.join(target_dir, feat_base)
  feats = glob.glob(os.path.join(feat_dir, "*.npy"))
  if len(feats) == 0:
    print("dir[{}] contains no npy".format(feat_dir))
    continue

  f = np.concatenate([np.load(feat) for feat in feats], -1)
  X = np.transpose(f)

  gm = sklearn.mixture.GaussianMixture(n_components=args.num_mixture, random_state=0)
  gm.fit(X)
  gms.append((basename, gm))

with open(gmfile, "wb") as f:
  pickle.dump(gms, f)
