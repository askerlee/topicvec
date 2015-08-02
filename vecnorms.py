# this simple script is to find patterns of the norms (L1) of the learned embeddings
from utils import *
import sys
import operator

if len(sys.argv) == 1:
    print "Usage: vecnorms.py embedding_filename"
    sys.exit(1)
    
embeddingFile = sys.argv[1]
V, vocab, word2id, skippedWords = load_embeddings(embeddingFile)
warning("\nCompute norms...")

word2norm = {}
for i in xrange( len(V) ):
    word2norm[ vocab[i] ] = norm1( V[i] )
warning("Done\nSorting words ascendingly by norm...")

# sort ascendingly by the norm length
sorted_wordnorms = sorted( word2norm.items(), key=operator.itemgetter(1) )
warning("Done\n")

for word_norm in sorted_wordnorms:
    word, norm = word_norm
    print "%i %s: %.2f" %( word2id[word], word, norm )
    