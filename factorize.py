import sys
import numpy as np
from scipy import linalg
import getopt
import timeit
import pdb
from utils import *
import os
import glob
import time

# factorization weighted by unigram probs
# MAXITERS is not used. Only for API conformity
# tikhonovCoeff: coefficient of Tikhonov regularization on the unigram prob-weighted least squares
# Not implemented yet
def uniwe_factorize(G, u, N0, MAXITERS=0, tikhonovCoeff=0, testenv=None):
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
    # keep the top N0 eigenvalues, set others to 0
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

    #print "No Weight V: %.3f, VV: %.3f, G-VV: %.3f" %( norm1(V), norm1(VV), norm1(G - VV) )
    print "L1 Uni Weighted VV: %.3f, G-VV: %.3f" %( norm1(VV, Weight), norm1(G - VV, Weight) )

#    if BigramWeight is not None:
#        print "Freq Weighted:"
#        print "VV: %.3f, G-VV: %.3f" %( norm1(VV, BigramWeight), norm1(G - VV, BigramWeight) )

    if testenv:
        model = VecModel( V, testenv['vocab'], testenv['word2id'], vecNormalize=True, precompute_gramian=True )
        evaluate_sim( model, testenv['simTestsets'], testenv['simTestsetNames'] )
        evaluate_ana( model, testenv['anaTestsets'], testenv['anaTestsetNames'] )

    timer.printElapseTime()

    return V, VV

# Factorization without weighting
# N: dimension of factorization
# tikhonovCoeff: coefficient of Tikhonov regularization on the unweighted least squares
def nowe_factorize(G, N, tikhonovCoeff=0):
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

    if tikhonovCoeff > 0:
        scale = 1.0 / ( 1 + tikhonovCoeff )
        es_N *= scale
        print "Regularized kept eigen sum: %.3f" %( norm1(es_N) )
        
    es_N_sqrt = np.diag( np.sqrt(es_N) )
    # remove all-zero columns
    es_N_sqrt = es_N_sqrt[ :, np.flatnonzero( es_N > 0 ) ]

    V = np.dot( vs, es_N_sqrt )
    VV = np.dot( V, V.T )

#    print "No Weight V: %.3f, VV: %.3f, G-VV: %.3f" %( norm1(V), norm1(VV), norm1(G - VV) )
#    print "No Weight Gsym: %.3f, Gsym-VV: %.3f" %( norm1(Gsym), norm1(Gsym - VV) )

    timer.printElapseTime()

    return V, VV

# Weighted factorization by bigram freqs, optimized using Gradient Descent algorithm
# Weight: nonnegative weight matrix. Assume already normalized
# N0: desired rank of V
# tikhonovCoeff: coefficient of Tikhonov regularization. Not implemented yet
def we_factorize_GD(G, Weight, N0, MAXITERS=5000, tikhonovCoeff=0, testenv=None):
    timer1 = Timer( "we_factorize_GD()" )
    # D is the number of words in the vocab (W in the paper)
    D = len(G)

    isEnoughGramian, installedMemGB, requiredMemGB = isMemEnoughGramian(D, 5)
    init_with_nowe = True

    if init_with_nowe:
        isEnoughEigen, installedMemGB, requiredMemGB = isMemEnoughEigen(D)
        # enough mem, initialize with unweighted low rank approximation
        if isEnoughEigen >= 1:
            # initialize V to unweighted low rank approximation
            V, VV = nowe_factorize(G, N0)

            if testenv:
                model = VecModel( V, testenv['vocab'], testenv['word2id'], vecNormalize=True, precompute_gramian=isEnoughGramian )
                evaluate_sim( model, testenv['simTestsets'], testenv['simTestsetNames'] )
                if isEnoughGramian:
                    evaluate_ana( model, testenv['anaTestsets'], testenv['anaTestsetNames'] )
                del model
                
        else:
            print "Not enough RAM: %.1fGB mem detected, %.1fGB mem required." %( installedMemGB, requiredMemGB )
            print "Initialize V randomly"
            V = np.random.rand( D, N0 )
            VV = np.dot( V, V.T )

    else:
        print "Initialize V randomly"
        V = np.random.rand( D, N0 )
        VV = np.dot( V, V.T )

    # in-place operations to reduce the space of a matrix
    G *= Weight
    G_Weight = G
    VV *= Weight
    norm1_VV = norm1(VV)
    VV -= G_Weight
    # In this function, A is defined as VV-G. 
    # A * Weight = ( VV - G ) * Weight = VV * Weight - G * Weight
    A_Weight = VV

    print "L1 Weighted: VV: %.3f, G-VV: %.3f" %( norm1_VV, norm1(A_Weight) )
    print "\nBegin Gradient Descent of weighted factorization by bigram freqs"

    #pdb.set_trace()

    for it in xrange(MAXITERS):
        timer2 = Timer( "GD iter %d" %(it+1) )
        print "\nGD Iter %d:" %( it + 1 )

        # step size
        gamma = 1.0 / ( it + 2 )
        # Grad is D*N0, takes little memory
        Grad = np.dot( A_Weight, V )

        # limit the norm of the step size to no less than the norm of V, times gamma
        # the gradient still converges to zero, although r is fluctuating
        r = norm1(V)/norm1(Grad)
        if r > 1.0:
            r = 1.0

        print "Rate: %f" %( r * gamma )

        V -= r * gamma * Grad
        VV = np.dot( V, V.T )
        VV *= Weight
        norm1_VV = norm1(VV)
        VV -= G_Weight
        A_Weight = VV

        print "L1 Weighted: VV: %.3f, G-VV: %.3f" %( norm1_VV, norm1(A_Weight) )

        if testenv and it % 5 == 4:
            model = VecModel( V, testenv['vocab'], testenv['word2id'], vecNormalize=True, precompute_gramian=isEnoughGramian )
            evaluate_sim( model, testenv['simTestsets'], testenv['simTestsetNames'] )
            # evaluate_ana is very slow when we couldn't precomputeGramian()
            # so in this case, only call it every 100 iterations to save training time
            if isEnoughGramian or it % 100 == 99:
                evaluate_ana( model, testenv['anaTestsets'], testenv['anaTestsetNames'] )
            del model
            
        timer2.printElapseTime()

    timer1.printElapseTime()

    VV = np.dot( V, V.T )
    return V, VV

# Weighted factorization by bigram freqs, optimized using EM algorithm
# if MAXITERS==1, it's identical to nowe_factorize()
# Weight: nonnegative weight matrix. Assume it's already normalized
# N0: desired rank
# tikhonovCoeff: coefficient of Tikhonov regularization on the weighted least squares using EM
def we_factorize_EM(G, Weight, N0, MAXITERS=5, tikhonovCoeff=0, testenv=None):

    timer1 = Timer( "we_factorize_EM()" )

    D = len(G)

    isEnoughEigen, installedMemGB, requiredMemGB = isMemEnoughEigen(D)
    # Not enough RAM
    if isEnoughEigen == 0:
        print "%.1fG RAM is required by eigendecomposition, but only %.1fG is installed" %(requiredMemGB, installedMemGB)
        print "Proceeding may hang your computer. Please reduce the factorized vocab size (-w)"
        print "You have 10 seconds to stop execution using Ctrl-C:"
        for i in xrange(10):
            time.sleep(1)
            print "\r%d\r" %i,
        print "Timeout. Proceed anyway"
        
    isEnoughGramian, installedMemGB, requiredMemGB = isMemEnoughGramian(D, 5)

    print "Begin EM of weighted factorization by bigram freqs"
    Gsym  = sym(G)

    alpha = 0.5
    # X: low rank rep in the M step
    X = alpha * G

    # N is N0 plus the iteration number. Decrease 1 after each iteration
    N = min( N0 + MAXITERS - 1, len(G) )

    #pdb.set_trace()

    for it in xrange(MAXITERS):
        timer2 = Timer( "EM iter %d" %(it+1) )
        print "\nEM Iter %d:" %(it+1)

        Gi = Weight * G + (1 - Weight) * X
        V, VV = nowe_factorize( Gi, N, tikhonovCoeff )

        # reduce the rank by one in every iteration
        if N > N0:
            N -= 1

        X = VV

        print "L1 Weighted: Gi: %.3f, VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizes( norm1, [Gi, VV, Gsym - VV, G - VV], Weight ) )
#        print "L2 Weighted: Gi: %.3f, VV: %.3f, Gsym-VV: %.3f, G-VV: %.3f" %tuple( matSizes( normF, [Gi, VV, Gsym - VV, G - VV], Weight ) )

        #X = Xnew

        if testenv:
            model = VecModel( V, testenv['vocab'], testenv['word2id'], vecNormalize=True, precompute_gramian=isEnoughGramian )
            evaluate_sim( model, testenv['simTestsets'], testenv['simTestsetNames'] )
            evaluate_ana( model, testenv['anaTestsets'], testenv['anaTestsetNames'] )

        timer2.printElapseTime()

    timer1.printElapseTime()

    return V, VV

# Weighted factorization by bigram freqs, optimized using Frank-Wolfe algorithm
# Weight: nonnegative weight matrix. Assume already normalized
# N0: desired rank of V
# tikhonovCoeff: coefficient of Tikhonov regularization. Not implemented yet
def we_factorize_FW(G, Weight, N0, MAXITERS=6, tikhonovCoeff=0, testenv=None):

    timer1 = Timer( "we_factorize_FW()" )

    D = len(G)

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
                model = VecModel( V, testenv['vocab'], testenv['word2id'], vecNormalize=True, precompute_gramian=True )
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
                model = VecModel( V, testenv['vocab'], testenv['word2id'], vecNormalize=True, precompute_gramian=True )
                evaluate_sim( model, testenv['simTestsets'], testenv['simTestsetNames'] )
                evaluate_ana( model, testenv['anaTestsets'], testenv['anaTestsetNames'] )

#        if it > 10 and norm1(G_VV, WeightSym) / norm1(G_VV_old, WeightSym) > 1.1:
#                print "Sudden surge of G-VV norm: %.3f -> %.3f. Stop" %( norm1(G_VV_old, WeightSym), norm1(G_VV, WeightSym) )
#                X, Gsym_VV, G_VV = X_old, Gsym_VV_old, G_VV_old
#                break

        timer2.printElapseTime()

    V, VV, vs, es = lowrank_fact(X, N0)

    print "Eigenvalue range: %f ~ %f\n" %(es[-1], es[0])
    print "End of Frank-Wolfe"
    print

    timer1.printElapseTime()

    #pdb.set_trace()
    return V, VV

# RawCounts is a list of numpy arrays. It may contain only one array
def normalizeWeight( RawCounts, do_weight_cutoff, cutQuantile=0.0002, zero_diagonal=True ):
    for Weight in RawCounts:
        np.sqrt(Weight, Weight)

    idealCutPoint = getQuantileCut( RawCounts[0], cutQuantile )

    maxwe = 0

    for i, Weight in enumerate(RawCounts):
        totalElemCount = Weight.shape[0] * Weight.shape[1]

        if do_weight_cutoff:
            cutEntryCount = np.sum( Weight > idealCutPoint )
            Weight[ Weight > idealCutPoint ] = idealCutPoint
            print "%d (%.3f%%) elements in Weight-%d cut off at %.2f" %( cutEntryCount,
                                               cutEntryCount * 100.0 / totalElemCount, i+1, idealCutPoint )

        if zero_diagonal:
            for i in xrange( min(Weight.shape) ):
                Weight[i,i]=0

        maxwe1 = np.max(Weight) * 1.0
        if maxwe1 > maxwe:
            maxwe = maxwe1

    for Weight in RawCounts:
        # normalize to [0,1]
        Weight /= maxwe

    print

    if len(RawCounts) == 1:
        return RawCounts[0]
    else:
        return RawCounts

# tikhonovCoeff: coefficient of Tikhonov regularization on the weighted least squares
# Set to 0 to disable regularization
# https://en.wikipedia.org/wiki/Tikhonov_regularization#Generalized_Tikhonov_regularization
def block_factorize( G, F, V1, N0, tikhonovCoeff, do_weight_cutoff ):
    # new G12, G21, F12, F21, allocated in loadBigramFileInBlock() and passed in
    # core_size * noncore_size, noncore_size * core_size
    G12, G21 = G
    F12, F21 = F

    noncore_size = len(G21)

    Weight12, Weight21 = normalizeWeight( [ F12, F21 ], do_weight_cutoff, zero_diagonal=False)

    # new WGsum: noncore_size * core_size
    WGsum = ( Weight12 * G12 ).T + ( Weight21 * G21 )

    memLogger.debug( "del G1, G21" )
    del G12, G21
    # G1, G21 should be released

    # Save memory, reuse Weight21
    Weight21 += Weight12.T
    Wsum = Weight21
    Wsum[ np.isclose(Wsum,0) ] = 0.001
    WGsum /= Wsum
    Gwmean = WGsum

    # only Wsum(Weight21) and Gwmean are used later
    del F12, Weight12

    # embeddings of noncore words
    # new V2: noncore_size * N0
    V2 = np.zeros( ( noncore_size, N0 ), dtype=np.float32 )
    Tikhonov = np.identity(N0) * tikhonovCoeff

    timer = Timer()

    print "Begin finding embeddings of non-core words"

    # Find each noncore word's embedding
    for i in xrange(noncore_size):
        # core_size
        wi = Wsum[i]
        # new VW: core_size * N0
        VW = V1.T * wi
        # new VWV: N0 * N0
        VWV = VW.dot(V1)
        VWV_Tik = VWV + Tikhonov
        V2[i] = np.linalg.inv(VWV_Tik).dot( VW.dot(Gwmean[i]) )
        if i >= 0 and i % 100 == 99:
            print "\r%d / %d." %(i+1,noncore_size),
            print timer.getElapseTime(), "\r",

    print

    # F21 = Weight21 = Wsum, should be released
    # Gwmean = WGsum should be released
    # VW should be released
    # VWV is N0*N0, small, ignored
    memLogger.debug( "del F21, WGsum, VW" )
    del F21, Weight21, Wsum, Gwmean, WGsum, VW

    return V2


def factorize( alg, algName, G, Weight, N0, MAX_ITERS, tikhonovCoeff, vocab, testenv, save_residuals=False ):
    print "%d iterations of %s" %( MAX_ITERS, algName )
    V, VV = alg( G, Weight, N0, MAX_ITERS, tikhonovCoeff, testenv )
    print

    vocab_size = len(vocab)

    save_embeddings( "%d-%d-%s.vec" %(vocab_size, N0, algName), vocab, V, "V" )
    print

    if save_residuals:
        A = G - VV
        save_embeddings( "%d-%d-%s.residual" %(vocab_size, N0, algName), vocab, A, "A" )
        print

def usage():
    print """factorize.py [ -n vec_dim -b num_core_words -v core_pre_vec_file ... ] bigram_file
Options:
  -n:  Dimensionality of the generated embeddings. Default: 500
  -b:  Do block-wise decomposition, and specify the number of core words
  -v:  Existing embedding file of core words, for online learning of noncore
       embeddings
  -o:  Number of noncore words to generate embeddings. Default: -1 (all)
  -w:  Number of words in bigram_file to generate embeddings. Default: -1 (all)
  -e:  A file containing extra words to generate embeddings, even if beyond
       the top words specified by -w
  -k:  Kappa of Jelinek-Mercer Smoothing (in percent). Default: 2 (=0.02)
  -t:  Specify a Tikhonov regularization coefficient. Default: 0 (disable)
  -c:  Disable weight cutoff
  -z:  Set G's elements to 0 whose corresponding weights are 0. 
       At the same time, set the default magnitude of vectors. Default: 8
       Diagonal of G will be set to the square of the default magnitude
  -E:  Number of iterations of the EM procedure. Default: 4
  -F:  Use Frank-Wolfe procedure, and specify the number of iterations
  -U:  Use PSD approximation with Unigram weighting
  -G:  Use Gradient Descent, and specify the number of iterations
"""

def main():
    # degree of smoothing
    kappa = 0.02
    # tikhonovCoeff: coefficient of Tikhonov regularization on the weighted least squares in block_factorize()
    # default is to disable it
    tikhonovCoeff = 0
    # vector dimensionality
    N0 = 500
    default_vec_len = 8
    
    # default -1 means to read all words
    vocab_size = -1
    core_size = -1
    noncore_size = -1
    pre_vec_file = None
    test_block = False

    do_smoothing = True
    do_weight_cutoff = True
    zero_G_elem_at_weight_0 = False
    
    extraWordFile = None
    do_UniWeight = False
    MAX_EM_ITERS = 4
    MAX_FW_ITERS = 0
    MAX_GD_ITERS = 0
    # EM iters of the core words
    MAX_CORE_EM_ITERS = 4
    do_block_factorize = False
    save_residuals = False

    try:
        opts, args = getopt.getopt(sys.argv[1:],"n:b:v:o:w:e:k:t:cz:E:F:UG:hr")
        if len(args) != 1:
            raise getopt.GetoptError("")
        bigram_filename = args[0]
        for opt, arg in opts:
            if opt == '-n':
                N0 = int(arg)
            if opt == '-b':
                do_block_factorize = True
                core_size = int(arg)
            if opt == '-v':
                do_block_factorize = True
                pre_vec_file = arg
            if opt == '-o':
                do_block_factorize = True
                noncore_size = int(arg)
            if opt == '-w':
                vocab_size = int(arg)
            if opt == '-e':
                extraWordFile = arg
            if opt == '-k':
                kappa = float(arg) / 100
            if opt == '-t':
                tikhonovCoeff = float(arg)
                print "Using Tikhonov regularization with coeff: %.1f" %tikhonovCoeff
            if opt == '-c':
                do_weight_cutoff = False
            if opt == '-z':
                zero_G_elem_at_weight_0 = True
                default_vec_len = float(arg)
            if opt == '-E':
                MAX_EM_ITERS = int(arg)
                MAX_CORE_EM_ITERS = int(arg)
            if opt == '-F':
                MAX_FW_ITERS = int(arg)
            # uniweight needs one iteration only. so no arg
            if opt == '-U':
                do_UniWeight = True
            if opt == '-G':
                MAX_GD_ITERS = int(arg)
            if opt == '-r':
                save_residuals = True
            if opt == '-h':
                usage()
                sys.exit(0)

    except getopt.GetoptError:
        usage()
        sys.exit(2)

    # load testsets
    simTestsetDir = "./testsets/ws/"
    simTestsetNames = [ "ws353_similarity", "ws353_relatedness", "bruni_men", "radinsky_mturk", "luong_rare", "simlex_999a" ]
    anaTestsetDir = "./testsets/analogy/"
    anaTestsetNames = [ "google", "msr" ]

    simTestsets = loadTestsets(loadSimTestset, simTestsetDir, simTestsetNames)
    anaTestsets = loadTestsets(loadAnaTestset, anaTestsetDir, anaTestsetNames)
    print

    testenv = { 'simTestsets': simTestsets, 'simTestsetNames': simTestsetNames,
                 'anaTestsets': anaTestsets, 'anaTestsetNames': anaTestsetNames }

    if do_block_factorize:
        if extraWordFile:
            print "Extra word file is unnecessary when doing blockwise factorization"
            sys.exit(2)

        if vocab_size > 0 and core_size > 0:
            # in case -o, -w, -b are all specified, check their consistency
            if noncore_size > 0 and noncore_size != vocab_size - core_size:
                print "noncore_size %d + core_size %d != vocab_size %d " %( noncore_size, core_size, vocab_size )
                sys.exit(2)
            else:
                noncore_size = vocab_size - core_size

        if not pre_vec_file:
            # do_block_factorize might be accidentally enabled by -o. But without -v or -b we couldn't proceed
            if core_size < 0:
                print "Neither -v nor -b is specified. Unable to determine core words"
                sys.exit(2)

            # passed-in word2preID_core={}
            vocab_all, word2id_all, word2id_core, coreword_preIDs, G, F, u \
                                                          = loadBigramFileInBlock( bigram_filename, core_size,
                                                              noncore_size, {}, {}, kappa )
            # returned coreword_preIDs = []

            # Usually in the bigram file there are many more words than -b core_size
            # so the actual core words read should always be core_size words. No update needed

            G11 = G.pop(0)
            F11 = F.pop(0)

            # Weight normalization is in place. F11 couldn't be released prior to Weight11
            Weight11 = normalizeWeight( [ F11 ], do_weight_cutoff)

            # since pre_vec_file is not specified, according to a previous condition, we know core_size > 0
            testenv['vocab'] = vocab_all[:core_size]
            testenv['word2id'] = word2id_core

            # obtain embeddings of core words
            # new V1, VV1 in we_factorize_EM()
            # core_size * N0, core_size * core_size
            V1, VV1 = we_factorize_EM( G11, Weight11, N0, MAX_CORE_EM_ITERS, testenv )
            print "\nEmbeddings of %d core words have been solved" %core_size
            memLogger.debug( "del G11, F11, VV1" )
            # Weight11 is a reference to F11. Has to be deleted too to release F11
            del G11, F11, Weight11, VV1
            vocab_joint = vocab_all
            V_pre_skipped = []
            word2id_joint = word2id_all

        else:
            if core_size > 0:
                print "Embeddings of top %d words in '%s' will be loaded as core" %( core_size, pre_vec_file )
            else:
                print "Embeddings of all words in '%s' will be loaded as core" %(pre_vec_file)

            # here we don't skip words yet
            # skippedWords_whatever is empty and we don't care about it
            V_pre, vocab_pre, word2preID, skippedWords_whatever = load_embeddings(pre_vec_file)
            N0 = V_pre.shape[1]

            prewords_skipped = {}
            vocab_skipped = []

            # recompute core_size, initialize vocab_core & word2preID_core
            # If there are less than core_size words in pre_vec_file, then the whole returned vocab is used as vocab_core
            # the actual core_size is smaller than specified core_size
            if core_size == -1 or core_size > len(vocab_pre):
                core_size = len(vocab_pre)
                vocab_core = vocab_pre
                word2preID_core = word2preID
            # 0 < core_size <= len(vocab_core)
            # the first core_size words in vocab_pre are vocab_core
            else:
                vocab_core = vocab_pre[:core_size]
                vocab_skipped = vocab_pre[core_size:]
                for w in vocab_skipped:
                    prewords_skipped[w] = 1
                word2preID_core = {}
                for w in vocab_core:
                    word2preID_core[w] = word2preID[w]

            if noncore_size > 0:
                print "2 blocks of %d core words and %d noncore words will be loaded. Skip %d words" \
                                %( core_size, noncore_size, len(prewords_skipped) )
            else:
                print "2 blocks of %d core words and all noncore words will be loaded. Skip %d words" \
                                %( core_size, len(prewords_skipped) )

            vocab_all, word2id_all, word2id_core, coreword_preIDs, G, F, u = \
                                         loadBigramFileInBlock( bigram_filename, core_size, noncore_size,
                                           word2preID_core, prewords_skipped, kappa )
            # the actual skipped vocab might be larger than vocab_skipped above.
            # Some pre core words might not exist in this bigram file, thus not in coreword_preIDs,
            # so they should be added to vocab_skipped

            # update core_size to the num of pre core words existing in the bigram file
            core_size = len(word2id_core)
            noncore_size = len(word2id_all) - core_size
            # select corresponding rows into V1 and V_pre_skipped.
            # V1 is some rows within the 0:core_size rows of V_pre, but row order might change
            # other rows are into V_pre_skipped
            V1 = V_pre[coreword_preIDs]
            skippedPreIDMask = np.ones( len(vocab_pre), np.bool )
            skippedPreIDMask[coreword_preIDs] = 0
            V_pre_skipped = V_pre[skippedPreIDMask]
            skippedPreIDs = set( xrange( len(vocab_pre) ) ) - set(coreword_preIDs)
            #pdb.set_trace()
            # update vocab_skipped, add words in the pre vec file but not in the bigram file
            vocab_skipped = [ vocab_pre[i] for i in skippedPreIDs ]
            vocab_joint = vocab_all[:core_size] + vocab_skipped + vocab_all[core_size:]
            #vocab_joint = vocab_all[:core_size] + vocab_all[core_size:]
            # updated word to ID mapping
            word2id_joint = { w:i for (i, w) in enumerate(vocab_joint) }

        # block_factorize( G, F, V1, N0, do_weight_cutoff ):
        V2 = block_factorize( G, F, V1, N0, tikhonovCoeff, True )

        # concatenate vocab's and V's.
        # A use case:
        # On an original vocab of [ 1, ..., i1, ..., i2, ..., i3, ... ]
        # 1. Factorize [ 1, ..., i1 ] as cores, and block factorize [ i1+1, ..., i2 ]
        # Save to a vec file.
        # vocab_all = vocab_joint = [ 1, ..., i1, ..., i2, ..., i3 ]
        # vocab_skipped = []
        # 2. load the vec file, use [ 1, ..., i1 ] as cores, and block factorize [ i2+1, ..., i3 ]
        # So in the second run of block factorization, skipped words should be placed inbetween
        # vocab_all = [ 1, ..., i1, i2+1, ..., i3 ]
        # vocab_skipped = [ i1+1, ..., i2 ]
        # vocab_joint = [ 1, ..., i1, i1+1, ..., i2, i2+1, ..., i3 ]

        vocab_jointsize = len(vocab_joint)
        #V_joint = np.concatenate( ( V1, V2 ) )
        V_joint = np.concatenate( ( V1, V_pre_skipped, V2 ) )
        save_embeddings( "%d-%d-%d-%s-%.1f.vec" %(core_size, vocab_jointsize, N0, "BLK", tikhonovCoeff), vocab_joint, V_joint, "V" )

        """
        if test_block:
            print "Test EM on the complete matrix\n"
            VV = np.dot( V, V.T )
            G0 = G[3]
            Weight0 = normalizeWeight( [ F[3] ], do_weight_cutoff )
            print "L1 Weighted VV: %.3f, G-VV: %.3f" %( norm1(VV, Weight0), norm1(G0 - VV, Weight0) )

            testenv['word2id'] = testenv['word2id_all']
            we_factorize_EM( G0, Weight0, N0, MAXITERS, testenv )
            print
        """

        if testenv:
            testenv['vocab'] = vocab_joint
            testenv['word2id'] = word2id_joint
            print "Test embeddings derived from block factorization\n"
            # An array of vocab_size * vocab_size is created here. Watch the amount of memory
            model = VecModel( V_joint, testenv['vocab'], testenv['word2id'], vecNormalize=True, precompute_gramian=True )
            evaluate_sim( model, testenv['simTestsets'], testenv['simTestsetNames'] )
            evaluate_ana( model, testenv['anaTestsets'], testenv['anaTestsetNames'] )

        # never save residuals when doing block factorization

    else:
        extraWords = {}
        if extraWordFile:
            with open(extraWordFile) as f:
                for line in f:
                    w, wid = line.strip().split('\t')
                    extraWords[w] = 1

        vocab, word2id, G, F, u = loadBigramFile( bigram_filename, vocab_size, extraWords, kappa )
        vocab_size = len(vocab)
        testenv['vocab'] = vocab
        testenv['word2id'] = word2id

        # Weight modifies F in place. Memory copy is avoided
        Weight = normalizeWeight( [ F ], do_weight_cutoff=do_weight_cutoff )

        if zero_G_elem_at_weight_0:
            NonzeroFilter = Weight != 0
            G *= NonzeroFilter
            nonzeroCount = np.count_nonzero(NonzeroFilter)
            totalCount = G.shape[0] * G.shape[1]
            zeroCount = totalCount - nonzeroCount
            print "%d (%.1f%%) nonzero elements in G are set to 0, %d left" \
                            %( zeroCount, zeroCount * 100.0 / totalCount, nonzeroCount )
            
            default_vec_len_sqr = default_vec_len * default_vec_len
            for i in xrange( G.shape[0] ):
                G[i,i] = default_vec_len_sqr
            print "Diagnoal elements of G are set to %.1f" %( default_vec_len_sqr )
            
        #we_factorize_GD( G, Weight, N0, testenv )

        # factorize( alg, algName, G, Weight, N, MAX_ITERS, tikhonovCoeff, vocab, testenv )
        #if do_UniWeight:
        #    factorize( uniwe_factorize, "UNI", G, u, N0, 0, vocab, 0, testenv, save_residuals )

        #if MAX_FW_ITERS > 0:
        #    factorize( we_factorize_FW, "FW", G, Weight, N0, MAX_FW_ITERS, 0, vocab, testenv, save_residuals )

        #if MAX_GD_ITERS > 0:
        #    factorize( we_factorize_GD, "GD", G, Weight, N0, MAX_GD_ITERS, 0, vocab, testenv, save_residuals )

        if MAX_EM_ITERS > 0:
            factorize( we_factorize_EM, "EM", G, Weight, N0, MAX_EM_ITERS, tikhonovCoeff, vocab, testenv, save_residuals )



if __name__ == '__main__':
    memLogger = initConsoleLogger("Mem")
    main()
