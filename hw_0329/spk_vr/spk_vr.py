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
parser.add_argument("--num-enroll", type=int, required=False, default=5)
parser.add_argument("--rel-factor", type=int, required=False, default=5)
parser.add_argument("--max-iter", type=int, required=False, default=15)
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
import random
import numpy as np
np.set_printoptions(3, suppress=True)
np.random.seed(args.random_seed)
import sklearn.mixture

if args.eval:
  if not os.path.isfile(umfile):
    sys.exit("UBM model {} is missing. train without --eval option")

  with open(umfile, 'rb') as f:
    um = pickle.load(f)
  import copy
  um_ = copy.deepcopy(um)

  feats_dict = {}
  for target_dir in args.speaker_dirs:
    basename = [e.strip() for e in target_dir.split('/') if e.strip() != ''][-1]

    test_dir = os.path.join(target_dir, 'test', feat_base)
    feats = glob.glob(os.path.join(test_dir, '*.npy'))
    if len(feats) == 0: continue
    feats_dict[basename] = [(f, np.transpose(np.load(f))) for f in feats]

  basenames = list(feats_dict.keys())
  for basename in basenames:
    feats = feats_dict[basename]
    feats_shuf = random.sample(feats, len(feats))
    enrolls = feats_shuf[:args.num_enroll]

    ta = feats_shuf[args.num_enroll]
    def get_fa():
      fa_basename = random.choice(basenames)
      while fa_basename == basename:
        fa_basename = random.choice(basenames)
      return random.choice(feats_dict[fa_basename])
    fa = get_fa()

    enroll_feat = np.concatenate([e[1] for e in enrolls], axis=0)
    for n_iter in range(args.max_iter):
      resp = um.predict_proba(enroll_feat)
      nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
      alpha = np.expand_dims(nk / (nk + args.rel_factor), -1)
      means = np.dot(resp.T, enroll_feat) / nk[:, np.newaxis]
      um.means_ = alpha * means + (1 - alpha) * um.means_

    ta_score = um.score(ta[1]) - um_.score(ta[1])
    fa_score = um.score(fa[1]) - um_.score(fa[1])
    print("{} target".format(ta_score))
    print("{} nontarget".format(fa_score))

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
