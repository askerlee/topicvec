import numpy as np
import sys
import pdb
from utils import *

topic_vec_file = sys.argv[1]
T = load_matrix_from_text( topic_vec_file, "topic" )
K = T.shape[0]
cosine_mat = []
for x in xrange(K):
    for y in xrange(x):
        if normF(T[x]) < 1e-6 or normF(T[y]) < 1e-6:
            continue
        cosine = np.dot( T[x], T[y] ) / normF(T[x]) / normF(T[y])
        cosine_mat.append( [ cosine, x, y ] )

cosine_sum = 0
for i in xrange( len(cosine_mat) ):
    cosine_sum += cosine_mat[i][0]

print "Avg: %.5f" %( cosine_sum / len(cosine_mat) )
cosine_sorted = sorted( cosine_mat, key=lambda cosine_tuple: cosine_tuple[0], reverse=True )
for i in xrange(10):
    cosine, x, y = cosine_sorted[i]
    print "%d,%d: %.5f" %( x, y, cosine )
    print "%d: %s" %( x, T[x][:10] )
    print "%d: %s" %( y, T[y][:10] )
    print
            