# TopicVec
TopicVec is the source code for "Generative Topic Embedding: a Continuous Representation of Documents" (ACL 2016).

PSDVec (in folder 'psdvec') is the source code for "A Generative Word Embedding Model and its Low Rank Positive Semidefinite Solution" (EMNLP 2015).

#### Update v0.7: 
The topic inference is now 6 times faster.

#### Update v0.6:
##### Algorithm update: 
topicvecDir.py: uses exact inference instead a second-order approximation in the M-step.

#### Update v0.5:
##### Main algorithm: 
topicvecDir.py: uses a Dirichlet prior for topic mixting proportions.

####Required files on Dropbox:
https://www.dropbox.com/sh/lqbk3iioobegbp8/AACc8Kfr1KZIkKl9bGaIrOjfa?dl=0

1. Pretrained 180000 embeddings (25000 cores) in 3 archives. For faster loading into Python, 25000-180000-500-BLK-8.0.vec.npy can be used;
2. Unigram files top1grams-wiki.txt & top1grams-reuters.txt;
3. RCV1 cleansed corpus ( before downloading, please apply for permission from NIST according to: http://trec.nist.gov/data/reuters/reuters.html ).

If you are in China, you can also download the above files from baidu netdisk without the hassle of "climbing over the wall":
https://pan.baidu.com/s/1gVmRhK1HA2XwVWZbZHHLZQ#list/path=%2F
