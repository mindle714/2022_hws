# -*- coding: utf-8 -*-
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

torch.manual_seed(1234)

import sys
sys.path.insert(0, os.path.abspath('stylegan2-pytorch'))
from model import Generator

device = torch.device("cuda:0")

from util import *

"""# Prepare StyleGAN"""

# base directory for this file: USE YOUR DIRECTORY
base_directory = '.'

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

"""# Prepare LPIPS and Cosine Distance
Because LPIPS and Cosine Distance is measured by **VGGNet** and **FaceNet**, we need to load these models too.
"""

l_lpips = VGGPerceptualLoss().to(device)  # LPIPS loss
l_cosine = CosineDistanceLoss().to(device)  # Cosine Distance loss

"""# 1. Naive GAN Inversion using MSE Loss (L2 Distance)

Here we will only use MSE loss for the loss function, i.e., $\mathcal{L}(x, G(z)) = \|x-G(z)\|_2$.  
With Pytorch, we can simply use the **nn.MSELoss()** for this.  
### This is just a baseline, so please add any ideas to improve this!
"""

import glob
files = glob.glob("{}/targets/*".format(base_directory))
assert len(files) == 20
targets = [Image.open(f).resize((image_size, image_size)) for f in files]
targets = [transform(t).unsqueeze(0).to(device) for t in targets]

"""# 1.1. Optimization on the Z space
- Initialize latent vector: here, we simply initialize randomly, i.e., $z_0\sim \mathcal{N}(0,I)$.
- Initialize optimizer & loss function.
- Iteratively update $z$ using gradient descent.
- After optimization, evaluate using L2, LPIPS, and Cosine distance

$\mathcal{L} = \mathcal{L}_{MSE} + w_1\mathcal{L}_{LPIPS} + w_2\mathcal{L}_{cos}$
"""

# hyperparameters
lr = 0.01       # learning rate
iters = 1000     # number of iteration
viz_freq = 200  # visualize every 100 iterations

# initialize random latent
z = torch.randn(1, latent_dim, device=device)
with torch.no_grad():
    x = G([z])[0]
    x = stylegan_postprocess(x)
    x_init = process(x)
    plt.imshow(x_init)
    plt.title(r"Initial Image, $G(z_0)$", fontsize=20)

plt.savefig("hw3_v1.png")
plt.clf()

"""# Evaluation: L2, LPIPS, and Cosine Distance"""

dist_l2_list = []
dist_lpips_list = []
dist_cosine_list = []

from tqdm import tqdm

fig = plt.figure(figsize=(16,4*len(targets)))
for it, target in tqdm(enumerate(targets), total=len(targets)):
    # wrap in nn.Parameter() to allow optimization
    latent = nn.Parameter(z.clone())

    # initialize optimizer & loss function
    optimizer = optim.Adam([latent], lr=lr)  # Adam Optimizer
    criterion = nn.MSELoss()                 # MSE Loss (L2 distance)

    # for collecting image & latent sequence
    image_sequence = []

    # Perform optimization
    for i in range(iters):
        x = G([latent])[0]
        x = stylegan_postprocess(x)
        loss = criterion(x, target)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if i%viz_freq == 0 or i==iters-1:
            # save image sequence
            image_sequence.append(x)

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

plt.savefig("hw3_v2.png")
plt.clf()

print("L2: {:.4f}({:.4f})\nLPIPS: {:.4f}({:.4f})\nCosine: {:.4f}({:.4f})".format(
    np.mean([e.item() for e in dist_l2_list]), np.std([e.item() for e in dist_l2_list]),
    np.mean([e.item() for e in dist_lpips_list]), np.std([e.item() for e in dist_lpips_list]),
    np.mean([e.item() for e in dist_cosine_list]), np.std([e.item() for e in dist_cosine_list])))

"""# 1.2. Optimization on the W space
Here, we repeat the above experiment on the $\mathcal{W}$ space.
- Initialize latent vector: here, we simply initialize randomly, i.e., $w_0 = f(z_0), \text{ where } z_0\sim \mathcal{N}(0,I)$.
- NOTE: **input_is_latent = True** when using the $w$ latent vector
"""

# NOTE: here we reuse the z0 initialized in the first experiment
with torch.no_grad():
    w = G.style(z)  # w = f(z)
    
with torch.no_grad():
    x = G([w], input_is_latent=True)[0]
    x = stylegan_postprocess(x)
    x_init = process(x)
    plt.imshow(x_init)
    plt.title(r"Initial Image, $G(z_0)$", fontsize=20)

plt.savefig("hw3_v3.png")
plt.clf()

dist_l2_list = []
dist_lpips_list = []
dist_cosine_list = []

fig = plt.figure(figsize=(16,4*len(targets)))
for target in targets:
    # project z to W space through mapping function f
    with torch.no_grad():
        w = G.style(z)  # w = f(z)

    # wrap in nn.Parameter() to allow optimization
    latent = nn.Parameter(w)

    # initialize optimizer & loss function
    optimizer = optim.Adam([latent], lr=lr)  # Adam Optimizer
    criterion = nn.MSELoss()                 # MSE Loss (L2 distance)

    # for collecting image & latent sequence
    image_sequence = []

    # Perform optimization
    for i in range(iters):
        x = G([latent], input_is_latent=True)[0]  # NOTE: input_is_latent=True for w space
        x = stylegan_postprocess(x)
        loss = criterion(x, target)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if i%viz_freq == 0 or i==iters-1:
            # save image sequence
            image_sequence.append(x)

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

"""# Evaluation: L2, LPIPS, and Cosine Distance"""

plt.savefig("hw3_v4.png")
plt.clf()

print("L2: {:.4f}({:.4f})\nLPIPS: {:.4f}({:.4f})\nCosine: {:.4f}({:.4f})".format(
    np.mean([e.item() for e in dist_l2_list]), np.std([e.item() for e in dist_l2_list]),
    np.mean([e.item() for e in dist_lpips_list]), np.std([e.item() for e in dist_lpips_list]),
    np.mean([e.item() for e in dist_cosine_list]), np.std([e.item() for e in dist_cosine_list])))

"""# 1.3. Optimization on the W+ space
Here, we repeat the above experiment on the $\mathcal{W}^+$ space.
- Initialize latent vector: here, we simply initialize randomly, i.e., $w_0 = f(z_0), \text{ where } z_0\sim \mathcal{N}(0,I)$.
- Then, we repeat $w_0$ 14 times to create $w^+_0 = [w_0, ..., w_0]^T$
- NOTE: **input_is_latent = True** when using the $w^+$ latent vector
"""

# NOTE: here we reuse the z0 initialized in the first experiment
with torch.no_grad():
    w = G.style(z)  # w = f(z)
    wp = w.unsqueeze(1).repeat(1,14,1)  # shape: [1,14,512]
    
with torch.no_grad():
    x = G([wp], input_is_latent=True)[0]
    x = stylegan_postprocess(x)
    x_init = process(x)
    plt.imshow(x_init)
    plt.title(r"Initial Image, $G(z_0)$", fontsize=20)
    plt.savefig("hw3_v5.png")
    plt.clf()

dist_l2_list = []
dist_lpips_list = []
dist_cosine_list = []

fig = plt.figure(figsize=(16,4*len(targets)))
for target in targets:
    # project z to W space through mapping function f
    with torch.no_grad():
        w = G.style(z)  # w = f(z)
        wp = w.unsqueeze(1).repeat(1,14,1)  # shape: [1,14,512]

    # wrap in nn.Parameter() to allow optimization
    latent = nn.Parameter(wp)

    # initialize optimizer & loss function
    optimizer = optim.Adam([latent], lr=lr)  # Adam Optimizer
    criterion = nn.MSELoss()                 # MSE Loss (L2 distance)

    # for collecting image & latent sequence
    image_sequence = []

    # Perform optimization
    for i in range(iters):
        x = G([latent], input_is_latent=True)[0]  # NOTE: input_is_latent=True for w space
        x = stylegan_postprocess(x)
        loss = criterion(x, target)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if i%viz_freq == 0 or i==iters-1:
            # save image sequence
            image_sequence.append(x)

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

"""# Evaluation: L2, LPIPS, and Cosine Distance"""

plt.savefig("hw3_v6.png")
plt.clf()

print("L2: {:.4f}({:.4f})\nLPIPS: {:.4f}({:.4f})\nCosine: {:.4f}({:.4f})".format(
    np.mean([e.item() for e in dist_l2_list]), np.std([e.item() for e in dist_l2_list]),
    np.mean([e.item() for e in dist_lpips_list]), np.std([e.item() for e in dist_lpips_list]),
    np.mean([e.item() for e in dist_cosine_list]), np.std([e.item() for e in dist_cosine_list])))
