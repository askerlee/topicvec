#!/usr/bin/env python
# kmeans.py using any of the 20-odd metrics in scipy.spatial.distance
# kmeanssample 2 pass, first sample sqrt(N)

from __future__ import division
import random
import numpy as np
from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py
    # http://docs.scipy.org/doc/scipy/reference/spatial.html
from scipy.sparse import issparse  # $scipy/sparse/csr.py

__date__ = "2011-11-17 Nov denis"
    # X sparse, any cdist metric: real app ?
    # centres get dense rapidly, metrics in high dim hit distance whiteout
    # vs unsupervised / semi-supervised svm

#...............................................................................
def kmeans( X, centres, delta=.001, maxiter=10, metric="euclidean", p=2, verbose=1 ):
    """ centres, Xtocentre, distances = kmeans( X, initial centres ... )
    in:
        X N x dim  may be sparse
        centres k x dim: initial centres, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centres
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    see also: kmeanssample below, class Kmeans below.
    """

    if verbose:
        print "kmeans: X %s  centres %s  delta=%.2g  maxiter=%d  metric=%s" % (
            X.shape, centres.shape, delta, maxiter, metric)
    allx = np.arange(N)
    prevdist = 0
    for jiter in range( 1, maxiter+1 ):
        D = cdist( X, centres, metric=metric, p=p )  # |X| x |centres|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx,xtoc]
        avdist = distances.mean()  # median ?
        if verbose >= 2:
            print "kmeans: av |X - nearest centre| = %.4g" % avdist
        if (1 - delta) * prevdist <= avdist <= prevdist \
        or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:
                centres[jc] = X[c].mean( axis=0 )
    if verbose:
        print "kmeans: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc)
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[ xtoc == j ]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print "kmeans: cluster 50 % radius", r50.astype(int)
        print "kmeans: cluster 90 % radius", r90.astype(int)
            # scale L1 / dim, L2 / sqrt(dim) ?
    return centres, xtoc, distances

def randomsample( X, n ):
    """ random.sample of the rows of X
        X may be sparse -- best csr
    """
    sampleix = random.sample( xrange( X.shape[0] ), int(n) )
    return X[sampleix]

if __name__ == "__main__":
    import random
    import sys
    from time import time

    N = 10000
    dim = 10
    ncluster = 10
    kmdelta = .001
    kmiter = 10
    metric = "cosine"
    seed = 1

    np.set_printoptions( 1, threshold=200, edgeitems=5, suppress=True )
    np.random.seed(seed)
    random.seed(seed)

    unigramFilename = "top1grams-wiki.txt"
    word_vec_file = "25000-180000-500-BLK-8.0.vec"
                
    vocab_dict = loadUnigramFile(unigramFilename)
    V, vocab, word2ID, skippedWords_whatever = load_embeddings(word_vec_file)
    # map of word -> id of all words with embeddings
    vocab_dict2 = {}
    
    if normalize_vecs:
        Vnorm = np.array( [ normF(x) for x in V ] )
        for i,w in enumerate(vocab):
            if Vnorm[i] == 0:
                print "WARN: %s norm is 0" %w
                # set to 1 to avoid "divided by 0 exception"
                Vnorm[i] = 1
            
        V /= Vnorm[:, None]
        
    # dimensionality of topic/word embeddings
    N0 = V.shape[1]

    customStopwordList = re.split( "\s+", self.customStopwords )
    for stop_w in customStopwordList:
        stopwordDict[stop_w] = 1
    print "Custom stopwords: %s" %( ", ".join(customStopwordList) )
                
    print "N %d  dim %d  ncluster %d  metric %s" % (N, dim, ncluster, metric)
    t0 = time()

    randomcentres = randomsample( X, ncluster )
    centres, xtoc, dist = kmeans( X, randomcentres,
        delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    print "%.0f msec" % ((time() - t0) * 1000)
    