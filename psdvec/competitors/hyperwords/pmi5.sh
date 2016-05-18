#!/bin/bash

# B) Window size 5 with dynamic contexts and "dirty" subsampling

CORPUS=/home/shaohua/D/corpus/cleanwiki.txt

mkdir w5.dyn.sub.del
python hyperwords/corpus2pairs.py --win 5 --dyn --sub 1e-5 --del ${CORPUS} > w5.dyn.sub.del/pairs
scripts/pairs2counts.sh w5.dyn.sub.del/pairs > w5.dyn.sub.del/counts
python hyperwords/counts2vocab.py w5.dyn.sub.del/counts

# Calculate PMI matrices for each collection of pairs
python hyperwords/counts2pmi.py --cds 0.75 w5.dyn.sub.del/counts w5.dyn.sub.del/pmi
