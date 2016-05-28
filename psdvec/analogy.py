import sys
import re
from utils import *

def pred_ana( model, a, a2, b, maxcands = 10 ):
    questWordIndices = [ model.word2id[x] for x in (a,a2,b) ]
    # b2 is effectively iterating through the vocab. The row is all the cosine values
    b2a2 = model.sim_row(a2)
    b2a  = model.sim_row(a)
    b2b  = model.sim_row(b)

    mulsims = ( b2a2 + 1 ) * ( b2b + 1 ) / ( b2a + 1.001 )
    mulsims[questWordIndices] = -10000
    b2s = []
    for i in xrange(maxcands):
        imul = np.nanargmax(mulsims)
        b2mul  = model.vocab[imul]
        b2s.append( [ b2mul, mulsims[imul] ] ) 
        mulsims[imul] = -10000
        
    return b2s
    
embedding_npyfile = "25000-180000-500-BLK-8.0.vec.npy"
embedding_arrays = np.load(embedding_npyfile)
V, vocab, word2ID, skippedWords_whatever = embedding_arrays
print "%d words loaded from '%s'" %(len(vocab), embedding_npyfile)
model = VecModel(V, vocab, word2ID, vecNormalize=True)
print "Model initialized. Ready for input:"

while True:
    line = raw_input()
    line = line.strip()
    words = re.split("\s+", line)
    if len(words) != 3:
        print "Only 3 words are allowed"
        continue
    
    oov = 0
    for w in words:
        if w not in model:
            print "'%s' not in vocab" %w
            oov += 1
    if oov > 0:
        continue
        
    a, a2, b = words
    b2s = pred_ana( model, a, a2, b )
    for word, sim in b2s:
        print word, sim
    print
        
