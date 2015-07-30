# PSDVec
Source code for "A Generative Word Embedding Model and its Low Rank Positive Semidefinite Solution" (accepted by EMNLP'15) and "PSDVec: Positive Semidefinte Word Embedding" (under review).

#### Update v0.4: Online block-wise factorization:
1. Obtain 25000 core embeddings, into _25000-500-EM.vec_:
    * ```python factorize.py -w 25000 top2grams-wiki.txt```  
2. Obtain 45000 noncore embeddings, totaling 70000 (25000 core + 45000 noncore), into _25000-70000-500-BLKEM.vec_:
    * ```python factorize.py -v 25000-500-EM.vec -o 45000 top2grams-wiki.txt```
3. Incrementally learn other 50000 noncore embeddings (based on 25000 core), into _25000-120000-500-BLKEM.vec_:
    * ```python factorize.py -v 25000-70000-500-BLKEM.vec -b 25000 -o 50000 top2grams-wiki.txt```
4. Repeat 3 a few times to get more embeddings of rarer words.

Pretrained 120,000 embeddings and evaluation results are uploaded.

#### Update v0.3: Block-wise factorization
Pretrained 100,000 embeddings and evaluation results are uploaded (_now replaced by an expanded set of 120,000 embeddings_).

Testsets are by courtesy of Omer Levy (https://bitbucket.org/omerlevy/hyperwords/src).
