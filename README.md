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
