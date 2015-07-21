# topicvec
Source code for the manuscript "A Generative Word Embedding Model and its Low Rank Positive Semidefinite Solution".

####Update v0.4: 
Online block-wise factorization:
  
1. Obtain 70000 embeddings (25000 core + 45000 noncore), into 25000-70000-500-BLKEM.vec
    python factorize.py -b 25000 -o 45000 top2grams-wiki.txt
2. Incrementally learn other 60000 noncore embeddings, into 25000-130000-500-BLKEM.vec
    python factorize.py -v 25000-45000-500-BLKEM.vec -b 25000 -o 60000 top2grams-wiki.txt

####Update v0.3: 
Block-wise positive semidefinite approximation is implemented. Pretrained 100,000 embeddings and evaluation results are uploaded.

Testsets are by courtesy of Omer Levy (https://bitbucket.org/omerlevy/hyperwords/src).
