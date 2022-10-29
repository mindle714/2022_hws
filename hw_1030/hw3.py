# -*- coding: utf-8 -*-
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mse", type=float, required=False, default=1.)
parser.add_argument("--lpips", type=float, required=False, default=0.)
parser.add_argument("--cosine", type=float, required=False, default=0.)
parser.add_argument("--latent-type", type=str, required=False,
  default="ws", choices=["z", "w", "ws"])
parser.add_argument("--lr", type=float, required=False, default=0.01)
parser.add_argument("--iters", type=int, required=False, default=1000)
parser.add_argument("--viz-freq", type=int, required=False, default=200)
parser.add_argument("--loss-schedule", action="store_true")
parser.add_argument("--pc-grad", action="store_true")
args = parser.parse_args()

import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torchvision import transforms
if args.pc_grad:
    from pcgrad import PCGrad

torch.manual_seed(1234)

import sys
sys.path.insert(0, os.path.abspath('stylegan2-pytorch'))
from model import Generator

device = torch.device("cuda:0")

from util import *
base_directory = '.'

"""# Prepare StyleGAN"""
# Generator configuration: do not change
image_size = 256
crop_size = 192
latent_dim = 512
num_mlp = 8

# directory for pretrained weight (enter your directory)
weight = f'{base_directory}/stylegan2-ffhq-config-256.pt'
pth = torch.load(weight)['g_ema']

# basic image transform
transform = transforms.Compose(
    [
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# load pretrained StyleGAN
G = Generator(image_size, latent_dim, num_mlp).to(device)
_ = G.load_state_dict(pth, strict=False)  # load state dict

l_mse = nn.MSELoss()
l_lpips = VGGPerceptualLoss().to(device)  # LPIPS loss; from VGGNet
l_cosine = CosineDistanceLoss().to(device)  # Cosine Distance loss; from FaceNet

import glob
files = glob.glob("{}/targets/*".format(base_directory))
assert len(files) == 20
targets = [Image.open(f).resize((image_size, image_size)) for f in files]
targets = [transform(t).unsqueeze(0).to(device) for t in targets]

# initialize random latent
z = torch.randn(1, latent_dim, device=device)

dist_l2_list = []
dist_lpips_list = []
dist_cosine_list = []

def fetch_latent(latent_type):
    latent = None
    if latent_type == "z":
        latent = nn.Parameter(z.clone())

    elif latent_type == "w":
        with torch.no_grad():
            w = G.style(z)  # w = f(z)
        latent = nn.Parameter(w)

    elif latent_type == "ws":
        with torch.no_grad():
            w = G.style(z)  # w = f(z)
            wp = w.unsqueeze(1).repeat(1,14,1)
            # shape: [1,14,512]
        latent = nn.Parameter(wp)

    input_is_latent = (latent_type == "w") or (latent_type == "ws")
    return latent, input_is_latent

def invert_latent(latent, input_is_latent, target, lr, iters, viz_freq):
    # initialize optimizer & loss function
    optimizer = optim.Adam([latent], lr=lr)  # Adam Optimizer
    if args.pc_grad:
        optimizer = PCGrad(optimizer)

    image_sequence = []
    mse_losses = []; lpips_losses = []; cosine_losses = []

    # Perform optimization
    for i in range(iters):
        x = G([latent], input_is_latent=input_is_latent)[0]
        x = stylegan_postprocess(x)

        losses = []

        loss_mse = l_mse(x, target)
        loss = args.mse * loss_mse
        losses.append(loss)
        loss_mse = loss_mse.cpu().detach().numpy()

        add_lpips = (args.lpips > 0.) and \
            ((not args.loss_schedule) or (i > iters // 2))

        loss_lpips = 0.
        if add_lpips:
            loss_lpips = l_lpips(x, target)
            loss = args.lpips * loss_lpips
            losses.append(loss)
            loss_lpips = loss_lpips.cpu().detach().numpy()
        
        add_cosine = (args.cosine > 0.) and \
            ((not args.loss_schedule) or (i > iters // 2))

        loss_cosine = 0.
        if add_cosine:
            loss_cosine = l_cosine(x, target).flatten()[0]
            loss = args.cosine * loss_cosine
            losses.append(loss)
            loss_cosine = loss_cosine.cpu().detach().numpy()

        mse_losses.append(loss_mse)
        lpips_losses.append(loss_lpips)
        cosine_losses.append(loss_cosine)
    
        optimizer.zero_grad()
        if args.pc_grad:
            optimizer.pc_backward(losses)
        else:
            sum(losses).backward()
        optimizer.step()
    
        if i % viz_freq == 0 or i == iters - 1:
            # save image sequence
            image_sequence.append(x)

    loss_dict = {
        "mse": mse_losses,
        "lpips": lpips_losses,
        "cosine": cosine_losses
    }

    return image_sequence, loss_dict

fig = plt.figure(figsize=(16,32))

from tqdm import tqdm
for it, target in tqdm(enumerate(targets), total=len(targets)):
    latent, input_is_latent = fetch_latent(args.latent_type) 
    image_sequence, loss_dict = invert_latent(latent, input_is_latent,
        target, args.lr, args.iters, args.viz_freq)
    
    # visualize & save image sequence
    imseq = process_image_sequence(image_sequence, target)
    ax = fig.add_subplot(len(targets), 1, it+1)
    ax.imshow(imseq)

    output = image_sequence[-1]  # final output
    dist_l2 = F.mse_loss(output, target)   # L2 distance
    dist_lpips = l_lpips(output, target)   # LPIPS distance
    dist_cosine = l_cosine(output, target) # Cosine distance

    dist_l2_list.append(dist_l2)
    dist_lpips_list.append(dist_lpips)
    dist_cosine_list.append(dist_cosine)

fig.suptitle("L2: {:.4f}({:.4f}) LPIPS: {:.4f}({:.4f}) Cosine: {:.4f}({:.4f})".format(
    np.mean([e.item() for e in dist_l2_list]), np.std([e.item() for e in dist_l2_list]),
    np.mean([e.item() for e in dist_lpips_list]), np.std([e.item() for e in dist_lpips_list]),
    np.mean([e.item() for e in dist_cosine_list]), np.std([e.item() for e in dist_cosine_list])))

name = "hw3_{}_mse{}_lp{}_cos{}".format(
    args.latent_type, args.mse, args.lpips, args.cosine)
if args.loss_schedule:
    name = "{}_ls".format(name)
if args.pc_grad:
    name = "{}_pcg".format(name)

plt.savefig("{}.png".format(name))
plt.clf()

num_loss = 1
if args.lpips > 0.: num_loss += 1
if args.cosine > 0.: num_loss += 1

fig = plt.figure(figsize=(6.4, 4.8 * num_loss))
idx = 1

ax_mse = fig.add_subplot(num_loss, 1, idx)
ax_mse.plot(np.arange(args.iters), loss_dict["mse"])
idx += 1

if args.lpips > 0.:
    ax_lpips = fig.add_subplot(num_loss, 1, idx)
    ax_lpips.plot(np.arange(args.iters), loss_dict["lpips"])
    idx += 1

if args.cosine > 0.:
    ax_cosine = fig.add_subplot(num_loss, 1, idx)
    ax_cosine.plot(np.arange(args.iters), loss_dict["cosine"])
    idx += 1

plt.savefig("{}_loss.png".format(name))
