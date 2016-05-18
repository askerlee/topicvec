#!/bin/bash

# Create embeddings with SVD
DIR=w5.rcv1.dyn.sub.del
python hyperwords/pmi2svd.py --dim 50 --neg 5 $DIR/pmi $DIR/svd
cp $DIR/pmi.words.vocab $DIR/svd.words.vocab
cp $DIR/pmi.contexts.vocab $DIR/svd.contexts.vocab
