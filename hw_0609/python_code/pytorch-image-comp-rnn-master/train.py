import time
import os
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms

from pytorch_msssim import SSIM

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch-size', '-N', type=int, default=32, help='batch size')
parser.add_argument(
    '--train', '-f', required=True, type=str, help='folder of training images')
parser.add_argument(
    '--max-epochs', '-e', type=int, default=200, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
# parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument(
    '--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
parser.add_argument('--network', type=str, default='conv', help='type of network')
parser.add_argument('--metric', type=str, default='mse', help='type of loss function')
args = parser.parse_args()

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(SSIM_Loss, self).forward(img1, img2) )

## load 32x32 patches from images
import dataset

train_transform = transforms.Compose([
    transforms.RandomCrop((32, 32)),
    transforms.ToTensor(),
])

train_set = dataset.ImageFolder(root=args.train, transform=train_transform)

train_loader = data.DataLoader(
    dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

print('total images: {}; total batches: {}'.format(
    len(train_set), len(train_loader)))

## load networks on GPU

if args.network == 'conv':
    import conv_network

    encoder = conv_network.EncoderCell().cuda()
    binarizer = conv_network.Binarizer().cuda()
    decoder = conv_network.DecoderCell().cuda()

elif args.network =='lstm':
    import network

    encoder = network.EncoderCell().cuda()
    binarizer = network.Binarizer().cuda()
    decoder = network.DecoderCell().cuda()



solver = optim.Adam(
    [
        {
            'params': encoder.parameters()
        },
        {
            'params': binarizer.parameters()
        },
        {
            'params': decoder.parameters()
        },
    ],
    lr=args.lr)


def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    encoder.load_state_dict(
        torch.load('/home/hejung/2022_hws/hw_0609/python_code/pytorch-image-comp-rnn-master/checkpoint/encoder_{}_{:08d}.pth'.format(s, epoch)))
    binarizer.load_state_dict(
        torch.load('/home/hejung/2022_hws/hw_0609/python_code/pytorch-image-comp-rnn-master/checkpoint/binarizer_{}_{:08d}.pth'.format(s, epoch)))
    decoder.load_state_dict(
        torch.load('/home/hejung/2022_hws/hw_0609/python_code/pytorch-image-comp-rnn-master/checkpoint/decoder_{}_{:08d}.pth'.format(s, epoch)))


def save(index, epoch=True):
    if not os.path.exists('/home/hejung/2022_hws/hw_0609/python_code/pytorch-image-comp-rnn-master/checkpoint'):
        os.mkdir('/home/hejung/2022_hws/hw_0609/python_code/pytorch-image-comp-rnn-master/checkpoint')

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'

    torch.save(encoder.state_dict(), '/home/hejung/2022_hws/hw_0609/python_code/pytorch-image-comp-rnn-master/checkpoint/encoder_{}_{:08d}.pth'.format(
        s, index))

    torch.save(binarizer.state_dict(),
               '/home/hejung/2022_hws/hw_0609/python_code/pytorch-image-comp-rnn-master/checkpoint/binarizer_{}_{:08d}.pth'.format(s, index))

    torch.save(decoder.state_dict(), '/home/hejung/2022_hws/hw_0609/python_code/pytorch-image-comp-rnn-master/checkpoint/decoder_{}_{:08d}.pth'.format(
        s, index))

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

# resume()

scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)

last_epoch = 0
if args.checkpoint:
    resume(args.checkpoint)
    last_epoch = args.checkpoint
    scheduler.last_epoch = last_epoch - 1

ssim_loss_fn = SSIM_Loss(data_range=1.0, size_average=True, channel=3).to(get_device())

for epoch in range(last_epoch + 1, args.max_epochs + 1):

    scheduler.step()

    for batch, data in enumerate(train_loader):
        batch_t0 = time.time()

        ## init lstm state
        encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                       Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
        encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
        encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))

        decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))
        decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
        decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                       Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
        decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()),
                       Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()))

        patches = Variable(data.cuda())

        solver.zero_grad()

        losses = []

        res = patches - 0.5

        bp_t0 = time.time()

        for _ in range(args.iterations):
            if args.network == 'conv':
                encoded = encoder(res)
                codes = binarizer(encoded)
                output = decoder(codes)

            elif args.network == 'lstm':
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                    res, encoder_h_1, encoder_h_2, encoder_h_3)
                codes = binarizer(encoded)
                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                    codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                        

            if args.metric == 'ssim':
                loss = ssim_loss_fn(output+0.5,res+0.5)
                losses.append(loss)

            res = res - output

            if args.metric == 'mse':
                losses.append((res**2).mean())

            if args.metric == 'l1':
                losses.append(res.abs().mean())

        bp_t1 = time.time()

        loss = sum(losses) / args.iterations
        loss.backward()

        solver.step()

        batch_t1 = time.time()

        print(
            '[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.
            format(epoch, batch + 1,
                   len(train_loader), loss.data, bp_t1 - bp_t0, batch_t1 -
                   batch_t0))
        print(('{:.4f} ' * args.iterations +
               '\n').format(* [l.data for l in losses]))

        index = (epoch - 1) * len(train_loader) + batch

        ## save checkpoint every 500 training steps
        if index % 500 == 0:
            save(0, False)

    save(epoch)
