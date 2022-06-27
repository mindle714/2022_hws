#!/usr/bin/bash

USAGE="Usage: $0 -e <encoder> -d <decoder>"
EXAMPLE="($0 -e conv_ckpt/encoder_epoch_00000200.pth -d conv_ckpt/decoder_epoch_00000200.pth)"

unset exps;
while getopts ":e:d:" opt; do
  case $opt in
    e) encoder="$OPTARG" ;;
    d) decoder="$OPTARG" ;;
    ?) echo "$USAGE"; echo "$EXAMPLE"; exit 2 ;;
  esac
done

if [ -z "$encoder" ] || [ -z "$decoder" ]; then  
  echo "$USAGE"; echo "$EXAMPLE"; exit 2
fi

iter=16
  
decdir="$(dirname $decoder | rev | cut -d/ -f1 | rev)"
decnet="$(cat $(dirname $decoder)/ARGS | tail -n1 | jq .network | tr -d \")"
decodedir="$decdir-$(basename $decoder)"

csv="test/decoded/$decodedir/ssim.csv"
mkdir -p test/decoded/$decodedir
echo -n "" > $csv

for i in {01..24..1}; do
  echo Encoding test/images/kodim$i.png

  encdir="$(dirname $encoder | rev | cut -d/ -f1 | rev)"
  encnet="$(cat $(dirname $encoder)/ARGS | tail -n1 | jq .network | tr -d \")"
  encodedir="$encdir-$(basename $encoder)"
  mkdir -p test/codes/$encodedir

  python3 encoder.py --model $encoder \
    --network $encnet \
    --input test/images/kodim$i.png \
    --output test/codes/$encodedir/kodim$i --iterations $iter

  echo Decoding test/codes/$encodedir/kodim$i.npz

  mkdir -p test/decoded/$decodedir/kodim$i
  python3 decoder.py --model $decoder \
    --network $decnet \
    --input test/codes/$encodedir/kodim$i.npz \
    --output test/decoded/$decodedir/kodim$i --iterations $iter

  echo Processing SSIM metric

  for j in $(seq $iter); do
    suffix="$(printf "%02d" $((j-1)))"
    echo -n "$(python3 metric.py -m ssim \
      -o test/images/kodim$i.png \
      -c test/decoded/$decodedir/kodim$i/$suffix.png)" >> $csv
    echo -n ", " >> $csv
  done
  echo "" >> $csv
done

python3 test/draw_rd.py --ref cal/jpeg_ssim.csv \
  --hyp $csv --output test/decoded/$decodedir/rd_curve.png
