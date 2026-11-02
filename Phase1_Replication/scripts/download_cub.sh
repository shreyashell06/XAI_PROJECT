#!/usr/bin/env bash
# Downloads CUB-200-2011 (~1.1 GB) and splits it into data/CUB/{train,test} (5994 / 5794 images,
# RGB-only as in Label-free CBM: 5990 / 5790). Source: https://data.caltech.edu/records/65de6-vp158
set -e
mkdir -p data
cd data
if [ ! -d CUB ]; then
  [ -f CUB_200_2011.tgz ] || wget -q --show-progress -O CUB_200_2011.tgz "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
  tar xzf CUB_200_2011.tgz
  python ../scripts/split_cub.py
  rm -rf CUB_200_2011 CUB_200_2011.tgz attributes.txt
fi
echo "CUB ready at data/CUB"
