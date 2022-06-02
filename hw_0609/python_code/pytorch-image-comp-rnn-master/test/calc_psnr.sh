#!/bin/bash

LSTM=/content/drive/MyDrive/pytorch-image-comp-rnn-master/cal/lstm_psnr.csv
JPEG=/content/drive/MyDrive/pytorch-image-comp-rnn-master/cal/jpeg_psnr.csv

echo -n "" > $LSTM
for i in {04,10,17,18,19}; do
  echo Processing /content/drive/MyDrive/pytorch-image-comp-rnn-master/output/kodim$i
  for j in {00..9..1}; do
    echo -n `python /content/drive/MyDrive/pytorch-image-comp-rnn-master/metric.py -m psnr -o /content/drive/MyDrive/KODAK_5/kodim$i.png -c /content/drive/MyDrive/pytorch-image-comp-rnn-master/output/kodim$i/$j.png`', ' >> $LSTM
  done
  echo "" >> $LSTM
done

echo -n "" > $JPEG
for i in {04,10,17,18,19}; do
  echo Processing /content/drive/MyDrive/pytorch-image-comp-rnn-master/jpeg/kodim$i
  for j in {01..15..1}; do
    echo -n `python /content/drive/MyDrive/pytorch-image-comp-rnn-master/metric.py -m psnr -o /content/drive/MyDrive/KODAK_5/kodim$i.png -c /content/drive/MyDrive/pytorch-image-comp-rnn-master/jpeg/kodim$i/$j.jpg`', ' >> $JPEG
  done
  echo "" >> $JPEG
done
