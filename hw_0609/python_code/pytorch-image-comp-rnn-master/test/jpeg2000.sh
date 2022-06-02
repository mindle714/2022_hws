#!/bin/bash

for i in {04,10,17,18,19}; do
  echo JPEG Encoding /content/drive/MyDrive/KODAK_5/kodim$i.png
  mkdir -p /content/drive/MyDrive/pytorch-image-comp-rnn-master/jpeg2000/kodim$i
  for j in {1..15..1}; do
    convert /content/drive/MyDrive/KODAK_5/kodim$i.png -quality $(($j*5)) -sampling-factor 4:2:0 /content/drive/MyDrive/pytorch-image-comp-rnn-master/jpeg2000/kodim$i/`printf "%02d" $j`.jp2
  done
done
