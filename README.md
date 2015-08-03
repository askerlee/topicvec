# PSDVec
Source code for "A Generative Word Embedding Model and its Low Rank Positive Semidefinite Solution" (accepted by EMNLP'15) and "PSDVec: Positive Semidefinte Word Embedding" (about the use of this toolset, under review).

#### Update v0.42: Tikhonov Regularization (=Spherical Gaussian Prior) to embeddings in block-wise factorization:
* ```python factorize.py -v 25000-500-EM.vec -o 45000 -t0.5 top2grams-wiki.txt```
* It usually brings 1~2% boost of accuracy on the testsets.

#### Update v0.41: Gradient Descent (GD) solution:
* ```python factorize.py -G 500 -w 120000 top2grams-wiki.txt```
* GD is fast and scalable, but the performance is much worse (10~20% lower on the testsets). It's not recommended, unless initialized using unweighted Eigendecomposition (which is still not scalable).

#### Update v0.4: Online Block-wise Factorization:
1. Obtain 25000 core embeddings using Weighted PSD Approximation, into _25000-500-EM.vec_:
    * ```python factorize.py -w 25000 top2grams-wiki.txt```  
2. Obtain 45000 noncore embeddings using Weighted Least Squares, totaling 70000 (25000 cores + 45000 noncores), into _25000-70000-500-BLK-0.0.vec_:
    * ```python factorize.py -v 25000-500-EM.vec -o 45000 top2grams-wiki.txt```
3. Incrementally learn other 50000 noncore embeddings (based on 25000 cores), into _25000-120000-500-BLK-0.0.vec_:
    * ```python factorize.py -v 25000-70000-500-BLK-0.0.vec -b 25000 -o 50000 top2grams-wiki.txt```
4. Repeat 3 a few times to get more embeddings of rarer words.

Pretrained 120,000 embeddings and evaluation results are uploaded.

#### Update v0.3: Block-wise Factorization
Pretrained 100,000 embeddings and evaluation results are uploaded (_now replaced by an expanded set of 120,000 embeddings_).

Testsets are by courtesy of Omer Levy (https://bitbucket.org/omerlevy/hyperwords/src).

The Gradient Descent algorithm was based on the suggestion of Peilin Zhao (not included as a part of the papers).
