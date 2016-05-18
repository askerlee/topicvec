#!/bin/bash

# B) Window size 5 with dynamic contexts and "dirty" subsampling

CORPUS=/home/shaohua/D/corpus/rcv1clean.txt
DIR=w5.rcv1.dyn.sub.del
mkdir $DIR
python hyperwords/corpus2pairs.py --win 5 --dyn --sub 1e-5 --del ${CORPUS} > $DIR/pairs
scripts/pairs2counts.sh $DIR/pairs > $DIR/counts
python hyperwords/counts2vocab.py $DIR/counts

# Calculate PMI matrices for each collection of pairs
python hyperwords/counts2pmi.py --cds 0.75 $DIR/counts $DIR/pmi
