#!/bin/bash
# A) Window size 2 with " subsampling
CORPUS=/home/shaohua/D/corpus/cleanwiki.txt

mkdir w2.sub
python hyperwords/corpus2pairs.py --win 2 --sub 1e-5 ${CORPUS} > w2.sub/pairs
scripts/pairs2counts.sh w2.sub/pairs > w2.sub/counts
python hyperwords/counts2vocab.py w2.sub/counts
# Calculate PMI matrices for each collection of pairs
python hyperwords/counts2pmi.py --cds 0.75 w2.sub/counts w2.sub/pmi
