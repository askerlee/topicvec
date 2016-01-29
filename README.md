# PSDVec
Source code for "A Generative Word Embedding Model and its Low Rank Positive Semidefinite Solution" (accepted by EMNLP'15) and "PSDVec: Positive Semidefinte Word Embedding" (about the use of this toolset, under review).

#### Update v0.5: Code for Topic Embedding:
Two algorithms: 
1. topicvecDir.py: uses a Dirichlet prior for topic mixting proportions.
2. topicvecMLE.py: No prior. Uses MLE to estimate topic mixting proportions.

#### Update v0.42: Tikhonov Regularization (=Spherical Gaussian Prior) to embeddings in block-wise factorization:
1. Obtain 25000 core embeddings using Weighted PSD Approximation, into _25000-500-EM.vec_:
    * ```python factorize.py -w 25000 top2grams-wiki.txt```  
2. Obtain 45000 noncore embeddings using Weighted Least Squares, totaling 80000 (25000 cores + 55000 noncores), into _25000-80000-500-BLK-2.0.vec_:
    * ```python factorize.py -v 25000-500-EM.vec -o 55000 -t2 top2grams-wiki.txt```
3. Incrementally learn other 50000 noncore embeddings (based on 25000 cores), into _25000-130000-500-BLK-4.0.vec_:
    * ```python factorize.py -v 25000-80000-500-BLK-2.0.vec -b 25000 -o 50000 -t4 top2grams-wiki.txt```
4. Repeat 3 again, with Tikhonov coeff = 8 to get more embeddings of rarer words, into _25000-180000-500-BLK-8.0.vec_:
    * ```python factorize.py -v 25000-130000-500-BLK-4.0.vec -b 25000 -o 50000 -t8 top2grams-wiki.txt```

Pretrained 180,000 embeddings and evaluation results are uploaded. Now the performance is systematically better than other methods.

#### Update v0.41: Gradient Descent (GD) solution:
* ```python factorize.py -G 500 -w 120000 top2grams-wiki.txt```
* GD is fast and scalable, but the performance is much worse (~10% lower on the testsets). It's not recommended, unless initialized using unweighted Eigendecomposition (which is still not scalable).

#### Update v0.4: Online Block-wise Factorization

#### Update v0.3: Block-wise Factorization

Testsets are by courtesy of Omer Levy (https://bitbucket.org/omerlevy/hyperwords/src).

The Gradient Descent algorithm was based on the suggestion of Peilin Zhao (not included as a part of the papers).
