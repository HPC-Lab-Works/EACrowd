#!/usr/bin/env bash
# Downloads the ETH/UCY leave-one-scene-out splits (Social-GAN protocol) as distributed
# in the Trajectron++ repository (positions in metres, 0.4 s per step).
set -euo pipefail
cd "$(dirname "$0")/.."
B=https://raw.githubusercontent.com/StanfordASL/Trajectron-plus-plus/master/experiments/pedestrians/raw
declare -A TEST=([eth]="biwi_eth" [hotel]="biwi_hotel" [univ]="students001 students003" [zara1]="crowds_zara01" [zara2]="crowds_zara02")
ALL="biwi_eth biwi_hotel crowds_zara01 crowds_zara02 crowds_zara03 students001 students003 uni_examples"
for d in eth hotel univ zara1 zara2; do
  mkdir -p data/raw/$d/{train,val,test}
  for f in ${TEST[$d]}; do curl -sSf -o data/raw/$d/test/$f.txt $B/$d/test/$f.txt; done
  for f in $ALL; do
    curl -sSf -o data/raw/$d/train/${f}_train.txt $B/$d/train/${f}_train.txt 2>/dev/null || rm -f data/raw/$d/train/${f}_train.txt
    curl -sSf -o data/raw/$d/val/${f}_val.txt $B/$d/val/${f}_val.txt 2>/dev/null || rm -f data/raw/$d/val/${f}_val.txt
  done
done
find data/raw -name "*.txt" | wc -l
