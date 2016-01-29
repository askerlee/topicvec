#!/bin/bash

# Create embeddings with SVD

CORPUS=/home/shaohua/D/corpus/cleanwiki.txt

python hyperwords/pmi2svd.py --dim 500 --neg 5 w2.sub/pmi w2.sub/svd
cp w2.sub/pmi.words.vocab w2.sub/svd.words.vocab
cp w2.sub/pmi.contexts.vocab w2.sub/svd.contexts.vocab
python hyperwords/pmi2svd.py --dim 500 --neg 5 w5.dyn.sub.del/pmi w5.dyn.sub.del/svd
cp w5.dyn.sub.del/pmi.words.vocab w5.dyn.sub.del/svd.words.vocab
cp w5.dyn.sub.del/pmi.contexts.vocab w5.dyn.sub.del/svd.contexts.vocab
