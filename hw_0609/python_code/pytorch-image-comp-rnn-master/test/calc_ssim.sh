#!/bin/bash

basedir="$(dirname -- $(readlink -f -- $0))"
LSTM=$basedir/../cal/lstm_ssim.csv
JPEG=$basedir/../cal/jpeg_ssim.csv

echo -n "" > $LSTM
#for i in {04,10,17,18,19}; do
for i in {01..24..1}; do
  echo Processing $basedir/../output/kodim$i
  for j in {00..15..1}; do
    echo -n `python3 $basedir/../metric.py -m ssim \
      -o $basedir/images/kodim$i.png \
      -c $basedir/decoded/conv_ckpt-decoder_epoch_00000200.pth/kodim$i/$j.png`', ' >> $LSTM
  done
  echo "" >> $LSTM
done

echo -n "" > $JPEG
#for i in {04,10,17,18,19}; do
for i in {01..24..1}; do
  echo Processing $basedir/../jpeg/kodim$i
  for j in {01..15..1}; do
    echo -n `python3 $basedir/../metric.py -m ssim \
      -o $basedir/images/kodim$i.png \
      -c $basedir/../jpeg/kodim$i/$j.jpg`', ' >> $JPEG
  done
  echo "" >> $JPEG
done

