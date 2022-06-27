#!/bin/bash

basedir="$(dirname -- $(readlink -f -- $0))"

#for i in {04,10,17,18,19}; do
for i in {01..24..1}; do
  echo JPEG Encoding $basedir/../jpeg/kodim$i.png
  mkdir -p $basedir/../jpeg/kodim$i
  for j in {1..15..1}; do
    convert $basedir/images/kodim$i.png -quality $(($j*5)) \
      -sampling-factor 4:2:0 $basedir/../jpeg/kodim$i/$(printf "%02d" $j).jpg
  done
done
