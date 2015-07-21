#topicvec
Source code for the manuscript "A Generative Word Embedding Model and its Low Rank Positive Semidefinite Solution".

####Update v0.4: 
Online block-wise factorization:

1. Obtain 25000 core embeddings, into 25000-500-EM.vec:

```python factorize.py -w 25000 top2grams-wiki.txt```  
2. Obtain 45000 noncore embeddings, totaling 70000 (25000 core + 45000 noncore), into 25000-70000-500-BLKEM.vec:

```python factorize.py -v 25000-500-EM.vec -o 45000 top2grams-wiki.txt```
3. 3. Incrementally learn other 50000 noncore embeddings (based on 25000 core), into 25000-120000-500-BLKEM.vec:

```python factorize.py -v 25000-70000-500-BLKEM.vec -b 25000 -o 50000 top2grams-wiki.txt```
4. Repeat 3 a few times to get more embeddings of rarer words.

####Update v0.3: 
Block-wise positive semidefinite approximation is implemented. Pretrained 100,000 embeddings and evaluation results are uploaded.

Testsets are by courtesy of Omer Levy (https://bitbucket.org/omerlevy/hyperwords/src).
