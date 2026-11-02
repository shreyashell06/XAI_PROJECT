#!/usr/bin/env bash
# Downloads CUB-200-2011 (~1.1 GB) and splits it into data/CUB/{train,test} (5994 / 5794 images,
# RGB-only as in Label-free CBM: 5990 / 5790). Source: https://data.caltech.edu/records/65de6-vp158
#
# Phase 2: the human annotations (part keypoints, bounding boxes, per-image attributes) are also kept,
# in data/CUB_meta/, because the faithfulness metrics (fvlc/faithfulness.py) need them. If data/CUB
# already exists from Phase 1 but data/CUB_meta does not, only the annotation files are extracted.
set -e
mkdir -p data
cd data

META_FILES="CUB_200_2011/images.txt CUB_200_2011/train_test_split.txt CUB_200_2011/bounding_boxes.txt \
CUB_200_2011/image_class_labels.txt CUB_200_2011/parts/parts.txt CUB_200_2011/parts/part_locs.txt \
CUB_200_2011/attributes/image_attribute_labels.txt CUB_200_2011/attributes/certainties.txt attributes.txt"

get_tgz() {
  [ -f CUB_200_2011.tgz ] || wget -q --show-progress -O CUB_200_2011.tgz \
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
}

copy_meta() {
  mkdir -p CUB_meta
  for f in $META_FILES; do cp "$f" CUB_meta/; done
}

if [ ! -d CUB ]; then
  get_tgz
  tar xzf CUB_200_2011.tgz
  python ../scripts/split_cub.py
  copy_meta
  rm -rf CUB_200_2011 CUB_200_2011.tgz attributes.txt
elif [ ! -f CUB_meta/part_locs.txt ]; then
  get_tgz
  tar xzf CUB_200_2011.tgz $META_FILES
  copy_meta
  rm -rf CUB_200_2011 CUB_200_2011.tgz attributes.txt
fi
echo "CUB ready at data/CUB (images) and data/CUB_meta (annotations)"
