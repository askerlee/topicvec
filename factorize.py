import sys
import numpy as np
from scipy import linalg
import getopt
import timeit
import pdb
from utils import *
import os
import glob

# factorization weighted by unigram probs
# MAXITERS is not used. Only for API conformity
def uniwe_factorize(G, u, N0, MAXITERS=0, testenv=None):
    timer = Timer( "uniwe_factorize()" )
    print "Begin factorization weighted by unigrams"

#    if BigramWeight is not None:
#        maxwe = np.max(BigramWeight) * 1.0
#        # normalize to [0,1]
#        BigramWeight = BigramWeight / maxwe

    Gsym  = sym(G)

    power = 0.5
    Utrans = np.power(u, power)
    Weight = np.outer( Utrans, Utrans )

    # es: eigenvalues. vs: right eigenvectors
    # Weight * Gsym = vs . diag(es) . vs.T 
    es, vs = np.linalg.eigh( Weight * Gsym )

    # find the cut point of positive eigenvalues
    es2 = sorted(es, reverse=True)
    
    d = len(es) - 1
    while es2[d] <= 0 and d >= 0:
        d -= 1
    posEigenSum = np.sum( es2[:d+1] )
    print "%d positive eigenvalues, sum: %.3f" %( d+1, posEigenSum )

    d = N0 - 1
    while es2[d] < 0 and d >= 0:
        d -= 1
    if d == -1:
        print "All eigenvalues are negative. Abnormal"
        sys.exit(2)

    print "Eigenvalues cut at the %d-th largest value, between %.3f-%.3f" %( d+1, es2[d], es2[d+1] )
    cutoff = es2[d]
    # keep the top N eigenvalues, set others to 0
    es_N = map( lambda x: x if x >= cutoff else 0, es )
    es_N = np.array(es_N, dtype=np.float32)
    print "All eigen norm: %.3f, Kept sum: %.3f" %( norm1(es), norm1(es_N) )

    es_N_sqrt = np.diag( np.sqrt(es_N) )
    # remove all-zero columns, leaving N columns
    es_N_sqrt = es_N_sqrt[ :, np.flatnonzero( es_N > 0 ) ]

    # vs.T / Utrans, is dividing vs.T row by row, = vs.T . diag(Utrans^-1)
    # (vs.T / Utrans).T, is dividing vs column by column. 
    V = np.dot( (vs.T / Utrans).T, es_N_sqrt )
    VV = np.dot( V, V.T )

    print "No Weight V: %.3f, VV: %.3f, G-VV: %.3f" %( norm1(V), norm1(VV), norm1(G - VV) )
    print "Uni Weighted VV: %.3f, G-VV: %.3f" %( norm1(VV, Weight), norm1(G - VV, Weight) )

#    if BigramWeight is not None:
#        print "Freq Weighted:"
#        print "VV: %.3f, G-VV: %.3f" %( norm1(VV, BigramWeight), norm1(G - VV, BigramWeight) )

    if testenv:
        model = vecModel( V, testenv['vocab'], testenv['word2dim'], vecNormalize=True )
        evaluate_sim( model, testenv['simTestsets'], testenv['simTestsetNames'] )
        evaluate_ana( model, testenv['anaTestsets'], testenv['anaTestsetNames'] )

    return V, VV

# Factorization without weighting
# N: dimension of factorization
def nowe_factorize(G, N):
    timer = Timer( "nowe_factorize()" )
    print "Begin unweighted factorization"

    Gsym = sym(G)

    # es: eigenvalues. vs: right eigenvectors
    # Gsym = vs . diag(es) . vs.T 
    es, vs = np.linalg.eigh(Gsym)

    # es2: sorted original eigenvalues of Gsym
    es2 = sorted(es, reverse=True)

    d = len(es) - 1
    # d points at the smallest nonnegative eigenvalue
    while es2[d] <= 0 and d >= 0:
        d -= 1
    posEigenSum = np.sum( es2[:d+1] )
    print "%d positive eigenvalues, sum: %.3f" %( d+1, posEigenSum )

    d = N - 1
    while es2[d] < 0 and d >= 0:
        d -= 1
    if d == -1:
        print "All eigenvalues are negative. Weird"
        sys.exit(2)

    #print "Eigenvalues:\n%s" %(es2)

    print "Eigenvalues cut at the %d-th, %.3f ~ %.3f" %( d+1, es2[d], es2[d+1] )
    cutoff = es2[d]
    # keep the top N eigenvalues, set others to 0
    es_N = map( lambda x: x if x >= cutoff else 0, es )
    es_N = np.array( es_N, dtype=np.float32 )
    print "All eigen sum: %.3f, Kept eigen sum: %.3f" %( norm1(es), norm1(es_N) )

    es_N_sqrt = np.diag( np.sqrt(es_N) )
    # remove all-zero columns
    es_N_sqrt = es_N_sqrt[ :, np.flatnonzero( es_N > 0 ) ]

    V = np.dot( vs, es_N_sqrt )
    VV = np.dot( V, V.T )

    print "No Weight V: %.3f, VV: %.3f, G-VV: %.3f" %( norm1(V), norm1(VV), norm1(G - VV) )
    print "No Weight Gsym: %.3f, Gsym-VV: %.3f" %( norm1(Gsym), norm1(Gsym - VV) )

    return V, VV

# Weighted factorization by bigram freqs, optimized using Gradient Descent algorithm
# Weight: nonnegative weight matrix. Assume already normalized
# N0: desired rank of V
def we_factorize_GD(G, Weight, N0, MAXITERS=5000, testenv=None):
    timer1 = Timer( "we_factorize_GD()" )
    D = len(Weight)

    # initialize V to unweighted low rank approximation
    #V, VV = nowe_factorize(G, N0)
    # In this function, A is defined as VV-G, i.e. minus the returned A
    #A = -A

    V = np.random.rand( N0, D )

    VV = np.dot( V, V.T )
    A = VV - G

    matSizes1 = matSizes( norm1, [ VV, A ], Weight ) #matSizes( norm1, [VV, Gsym - VV, A], WeightSym ) + \

#    print "L1 SymWeighted: VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizes1[0:3] )
    print "L1 Weighted: VV: %.3f, G-VV: %.3f" %tuple( matSizes1 )

#    matSizesF = matSizes( normF, [ VV, A ], Weight ) #matSizes( normF, [VV, Gsym - VV, A], WeightSym ) + \

#    print "L2 SymWeighted: VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizesF[0:3] )
#    print "L2 Weighted: VV: %.3f, G-VV: %.3f" %tuple( matSizesF )

    print "\nBegin Gradient Descent of weighted factorization by bigram freqs"

    #pdb.set_trace()

    for it in xrange(MAXITERS):
        timer2 = Timer( "GD iter %d" %(it+1) )
        print "\nGD Iter %d:" %( it + 1 )
        
        # step size
        gamma = 1.0 / ( it + 2 )
        Grad = np.dot( (A * Weight), V.T )

        if it < 50:
            r = norm1(V)/norm1(Grad)
        else:
            r = 1

        print "Rate: %f" %(r*gamma)

        Vnew = V - r * gamma * Grad
        VV = np.dot( Vnew, Vnew.T )
        A = VV - G

        matSizes1 = matSizes( norm1, [VV, A], Weight ) #matSizes( norm1, [VV, Gsym - VV, A], WeightSym ) + \

#       print "L1 SymWeighted: VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizes1[0:3] )
        print "L1 Weighted: VV: %.3f, G-VV: %.3f" %tuple( matSizes1 )

#        matSizesF = matSizes( normF, [VV, A], Weight ) #matSizes( normF, [VV, Gsym - VV, A], WeightSym ) + \

#       print "L2 SymWeighted: VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizesF[0:3] )
#        print "L2 Weighted: VV: %.3f, G-VV: %.3f" %tuple( matSizesF )

        V = Vnew

        if testenv:
            model = vecModel( V, testenv['vocab'], testenv['word2dim'], vecNormalize=True )
            evaluate_sim( model, testenv['simTestsets'], testenv['simTestsetNames'] )
            evaluate_ana( model, testenv['anaTestsets'], testenv['anaTestsetNames'] )
        
# Weighted factorization by bigram freqs, optimized using EM algorithm
# if MAXITERS==1, it's identical to nowe_factorize()
# Weight: nonnegative weight matrix. Assume already normalized
# N0: desired rank
def we_factorize_EM(G, Weight, N0, MAXITERS=5, testenv=None):

    timer1 = Timer( "we_factorize_EM()" )
    
    print "Begin EM of weighted factorization by bigram freqs"
    Gsym  = sym(G)

    alpha = 0.5
    # X: low rank rep in the M step
    X = alpha * G

    N = min( N0 + MAXITERS - 1, len(G) )

    #pdb.set_trace()

    for it in xrange(MAXITERS):
        timer2 = Timer( "EM iter %d" %(it+1) )
        print "\nEM Iter %d:" %(it+1)
        
        Gi = Weight * G + (1 - Weight) * X
        V, VV = nowe_factorize(Gi, N)

        # reduce the rank by one in every iteration
        if N > N0:
            N -= 1

        X = VV

        print "L1 Weighted: Gi: %.3f, VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizes( norm1, [Gi, VV, Gsym - VV, G - VV], Weight ) )
#        print "L2 Weighted: Gi: %.3f, VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizes( normF, [Gi, VV, Gsym - VV, G - VV], Weight ) )

        #X = Xnew

        if testenv:
            model = vecModel( V, testenv['vocab'], testenv['word2dim'], vecNormalize=True )
            evaluate_sim( model, testenv['simTestsets'], testenv['simTestsetNames'] )
            evaluate_ana( model, testenv['anaTestsets'], testenv['anaTestsetNames'] )

    return V, VV

# Weighted factorization by bigram freqs, optimized using Frank-Wolfe algorithm
# Weight: nonnegative weight matrix. Assume already normalized
# N0: desired rank of V
def we_factorize_FW(G, Weight, N0, MAXITERS=6, testenv=None):

    timer1 = Timer( "we_factorize_FW()" )
    
    D = len(Weight)

    Gsym  = sym(G)

    # initial X in the SVD set
    X = 0
    #V, VV = nowe_factorize(G, N0)
    #X = VV

    print "Begin Frank-Wolfe of weighted factorization by bigram freqs"

#    e, v = np.linalg.eigh( Gsym )
#    print "Eigen sum of Gsym: %.3f" %( norm1(e) )
#
#    e, v = np.linalg.eigh( sym(G * Weight) )
#    print "Eigen sum of GWsym: %.3f" %( norm1(e) )
#    t = np.abs(e[0]) * 5

    t = 10

    print "t=%.3f\n" %(t)

    #pdb.set_trace()
    use_power_iter = False

    TRUNC_CYCLE = 3
    # maximum number of largest negative eigenvalues/eigenvectors used for update
    # "-1" means all
    maxUpdVecNum = -1

    for it in xrange(MAXITERS):
        timer2 = Timer( "FW iter %d" %( it + 1 ) )
        print "\nFW Iter %d:" %( it + 1 )
        # step size
        gamma = 1.0 / ( it + 2 )
        #Grad = ( X - Gsym ) * WeightSym
        # symmetrized original grad performs much better than symmetric grad ( X - Gsym ) * WeightSym
        Grad = sym( ( X - G ) * Weight )

#        if use_power_iter:
#        #if True:
#            e1p, v1p = power_iter(Grad)
#            if e1p > 0:
#                print "Warn: Principal eigenvalue %.3f > 0" %e1p
                #break
        #else:

        # largest k negative eigenvalues / eigenvectors
        negEs = []
        negVs = []

        if True:
            #eigenPos = -1
            # eigenvalues are in ascending order in es
            # largest (magnitude) negative eigenvalue is at the beginning of es
            es, vs = np.linalg.eigh(Grad)

            for i in xrange( len(es) ):
                if es[i] < -0.001 and ( maxUpdVecNum == -1 or len(negEs) < maxUpdVecNum ):
                    negEs.append( es[i] )
                    negVs.append( vs[:, i].astype(np.float64) )
                if es[i] >= 0:
                    break

            if len(negEs) == 0:
                print "All eigenvalues are positive. Stop"
                break

            greaterPositiveCount = 0
            for i in xrange( len(es) - 1, -1, -1 ):
                if es[i] >= abs( negEs[0] ):
                    greaterPositiveCount += 1
                else:
                    break

            # at least 1, at most N0
            # try to keep S low rank
            usedEigenNum = min( max( len(negEs)/3, 1 ), N0/3 )

            print "%d -eigen used. %.3f ~ %.3f" %( usedEigenNum, negEs[0], negEs[ usedEigenNum - 1 ] )
            if greaterPositiveCount > 0:
                print "Warn: Principal eigen %.3f > 0, %d +eigen are larger" %( es[-1], greaterPositiveCount )

        #e1 = e1d
        #v1 = v1d

        # efft: effective t
        efft = t / np.power(gamma, 0.7)
        print "eff t: %.3f" %efft

        S = np.zeros((D, D), dtype=np.float64)

        # S is always PSD
        for i in xrange( usedEigenNum ):
            S += efft * np.outer( negVs[i], negVs[i] )

        # since X is PSD, Xnew is also PSD
        Xnew = X + gamma * ( S.astype(np.float32) - X )

        doTrunc = False

        # Truncate insignificant eigenvectors
        if it%TRUNC_CYCLE == TRUNC_CYCLE - 1:
            # print the old matrix sizes first for contrast to the new sizes
            matSizes1old = matSizes( norm1, [Xnew, Gsym - Xnew, G - Xnew], Weight ) #matSizes( norm1, [Xnew, Gsym - X, G - X], WeightSym ) + \

#           print "L1 SymWeighted: VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizes1old[0:3] )
            print "L1 Weighted: VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizes1old )

#            matSizesFold = matSizes( normF, [Xnew, Gsym - Xnew, G - Xnew], Weight ) #matSizes( normF, [Xnew, Gsym - X, G - X], WeightSym ) + \

#           print "L2 SymWeighted: VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizesFold[0:3] )
#            print "L2 Weighted: VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizesFold )

            es, vs = np.linalg.eigh(Xnew)

            es = map( lambda x: x if x >= 0 else 0, es )
            es = np.array( es, dtype = np.float32 )     
                       
            E_sqrt = np.diag( np.sqrt(es) )
            V = vs.dot(E_sqrt)

            if testenv:
                model = vecModel( V, testenv['vocab'], testenv['word2dim'], vecNormalize=True )
                evaluate_sim( model, testenv['simTestsets'], testenv['simTestsetNames'] )
                evaluate_ana( model, testenv['anaTestsets'], testenv['anaTestsetNames'] )

            print "Clear %d insignificant eigenvectors: %.3f ~ %.3f" %( D - N0, es[ D - N0 - 1], es[0] )

            es[ 0 : D - N0 ] = 0
            Xnew = vs.dot( np.diag(es).dot( vs.T ) )
            
            E_sqrt = np.diag( np.sqrt(es) )
            V = vs.dot(E_sqrt)
            
            doTrunc = True

        X = Xnew

        matSizes1 = matSizes( norm1, [Xnew, Gsym - X, G - X], Weight ) #matSizes( norm1, [Xnew, Gsym - X, G - X], WeightSym ) + \

#       print "L1 SymWeighted: VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizes1[0:3] )
        print "L1 Weighted: VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizes1 )

#        matSizesF = matSizes( normF, [Xnew, Gsym - X, G - X], Weight ) #matSizes( normF, [Xnew, Gsym - X, G - X], WeightSym ) + \

#       print "L2 SymWeighted: VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizesF[0:3] )
#        print "L2 Weighted: VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizesF )

        if doTrunc:
            matSizes1old = np.array( matSizes1old, dtype=np.float32 )
#            matSizesFold = np.array( matSizesFold, dtype=np.float32 )
            matSizes1    = np.array( matSizes1, dtype=np.float32 )
#            matSizesF    = np.array( matSizesF, dtype=np.float32 )
            ratio1 = matSizes1 / matSizes1old
            print "Trunc max ratio in norm1: %.3f" %max(ratio1)
#            ratioF = matSizesF / matSizesFold
#            print "Trunc max ratio: norm1 %.3f, normF %.3f" %( max(ratio1), max(ratioF) )

            if testenv:
                model = vecModel( V, testenv['vocab'], testenv['word2dim'], vecNormalize=True )
                evaluate_sim( model, testenv['simTestsets'], testenv['simTestsetNames'] )
                evaluate_ana( model, testenv['anaTestsets'], testenv['anaTestsetNames'] )

#        if it > 10 and norm1(G_VV, WeightSym) / norm1(G_VV_old, WeightSym) > 1.1:
#                print "Sudden surge of G-VV norm: %.3f -> %.3f. Stop" %( norm1(G_VV_old, WeightSym), norm1(G_VV, WeightSym) )
#                X, Gsym_VV, G_VV = X_old, Gsym_VV_old, G_VV_old
#                break

    V, VV, vs, es = lowrank_fact(X, N0)

    print "Eigenvalue range: %f ~ %f\n" %(es[-1], es[0])
    print "End of Frank-Wolfe"
    print

    #pdb.set_trace()
    return V, VV

def normalizeWeight( RawCounts, cutQuantile=0.0004, zero_weight_diagonal=True ):
    np.sqrt(RawCounts, RawCounts)
    Weight = RawCounts
    idealCutPoint = getQuantileCut( Weight, cutQuantile )
    totalElemCount = Weight.shape[0] * Weight.shape[1]
    
    if do_weight_cutoff:
        cutEntryCount = np.sum( Weight > idealCutPoint )
        Weight[ Weight > idealCutPoint ] = idealCutPoint
        print "%d (%.3f%%) elements in Weight cut off at %.2f" %(cutEntryCount, 
                                                        cutEntryCount * 100.0 / totalElemCount, idealCutPoint)

    if zero_weight_diagonal:
        for i in xrange(len(Weight)):
            Weight[i,i]=0
    
    maxwe = np.max(Weight) * 1.0
    # normalize to [0,1]
    Weight = Weight / maxwe
    return Weight
    
def factorize(alg, algName, G, Weight, N, MAX_ITERS, testenv):
    V, VV = alg( G, Weight, N, MAX_ITERS, testenv )
    A = G - VV
    print
    save_embeddings( "%d-%d-%s.vec" %(vocab_size, N, algName), vocab, V, "V" )
    save_embeddings( "%d-%d-%s.residue" %(vocab_size, N, algName), vocab, A, "A" )
    print
    
def main():
    kappa = 0.01
    # vector dimensionality
    N = 500
    # default -1 means to read all words
    topWordNum = -1
    vocab_size = -1
    do_smoothing = True
    do_weight_cutoff = True
    extraWordFile = None
    do_UniWeight = False
    MAX_EM_ITERS = 0
    MAX_FW_ITERS = 0
    # EM iters of the core words
    MAX_CORE_EM_ITERS = 5
    block_factorize = False
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"k:n:t:e:UE:F:Cv:b")
        if len(args) != 1:
            raise getopt.GetoptError("")
        bigram_filename = args[0]
        for opt, arg in opts:
            if opt == '-k':
                kappa = float(arg) / 100
            if opt == '-n':
                N = int(arg)
            if opt == '-t':
                topWordNum = int(arg)
            if opt == '-v':
                vocab_size = int(arg)
            if opt == '-e':
                extraWordFile = arg
            if opt == '-C':
                do_weight_cutoff = False
            if opt == '-U':
                do_UniWeight = True
            if opt == '-E':
                MAX_EM_ITERS = int(arg)
            if opt == '-F':
                MAX_FW_ITERS = int(arg)
            if opt == '-b':
                block_factorize = True
                    
    except getopt.GetoptError:
        print 'Usage: factorize.py [ -k smooth_k -n vec_dim -t topword_num -v vocab_size -e extra_word_file -U -E -F ] bigram_file'
        sys.exit(2)

    # load testsets
    simTestsetDir = "D:/Dropbox/doc2vec/code/testsets/ws/"
    simTestsetNames = [ "ws353_similarity", "ws353_relatedness", "bruni_men", "radinsky_mturk", "luong_rare", "simlex_999a" ]
    anaTestsetDir = "D:/Dropbox/doc2vec/code/testsets/analogy/"
    anaTestsetNames = [ "google", "msr" ]

    simTestsets = loadTestsets(loadSimTestset, simTestsetDir, simTestsetNames)
    anaTestsets = loadTestsets(loadAnaTestset, anaTestsetDir, anaTestsetNames)
    print
    
    testenv = { 'vocab': vocab, 'word2dim': word2dim, 'simTestsets': simTestsets, 'simTestsetNames': simTestsetNames,
                 'anaTestsets': anaTestsets, 'anaTestsetNames': anaTestsetNames }

    if block_factorize:
        if topWordNum == -1:
            print "-t has to be specified when doing blockwise factorization"
            sys.exit(2)
        if extraWordFile:
            print "Extra word file is unnecessary when doing blockwise factorization"
            sys.exit(2)
            
        vocab, word2dim, G, F, u = loadBigramFileBlock( bigram_filename, topWordNum, kappa, vocab_size )
        vocab_size = len(vocab)
        G11, G12, G21 = G
        F11, F12, F21 = F
        
        # Weight11 modifies F11 in place. Memory copy is avoided
        Weight11 = normalizeWeight(F11)
        V11, VV11 = we_factorize_EM( G11, Weight11, N, MAX_CORE_EM_ITERS, testenv )
        
        
    else:        
        extraWords = {}
        if extraWordFile:
            with open(extraWordFile) as f:
                for line in f:
                    w, wid = line.strip().split('\t')
                    extraWords[w] = 1
                    
        vocab, word2dim, G, F, u = loadBigramFile(bigram_filename, topWordNum, extraWords, kappa)
        vocab_size = len(vocab)
    
        # Weight modifies F in place. Memory copy is avoided
        Weight = normalizeWeight(F)

        #we_factorize_GD( G, Weight, N, testenv )
        
        # factorize(alg, algName, G, Weight, N, MAX_ITERS, testenv)
        if do_UniWeight:
            factorize(uniwe_factorize, "UNI", G, u, N, 0, testenv)
            
        if MAX_EM_ITERS > 0:
            factorize(we_factorize_EM, "EM", G, Weight, N, MAX_EM_ITERS, testenv)
    
        if MAX_FW_ITERS > 0:
            factorize(we_factorize_FW, "FW", G, Weight, N, MAX_EM_ITERS, testenv)


if __name__ == '__main__':
    main()
