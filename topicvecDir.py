import numpy as np
import scipy.linalg
from scipy.special import *
import getopt
import sys
from utils import *
import topicvecUtil as tutil
import pdb
import time

unigramFilename = "top1grams-wiki.txt" 
vec_file = "25000-500-EM.vec"
doc_filename = None
K = 30
N0 = 500
max_l = 3
init_l = 1
max_grad = 0
topD = 12
alpha0 = 5
alpha1 = 1
delta = 0.1
MAX_EM_ITERS = 500
topicDiff_tolerance = 2e-3
zero_topic0 = True
smoothing_context = 0
context_weight = 0.5
appendLogfile = False
remove_stop = True
seed = 0

def usage():
    print """topicvecDir.py [ -v vec_file -a alpha ... ] doc_file
Options:
  -k:  Number of topic embeddings to extract. Default: 20
  -v:  Existing embedding file of all words.
  -r:  Existing residual file of core words.
  -a:  Hyperparameter alpha. Default: 0.1.
  -i:  Number of iterations of the EM procedure. Default: 100
  -u:  Unigram file, to obtain unigram probs.
  -l:  Magnitude of topic embeddings.  
  -A:  Append to the old log file.
  -s:  Seed the random number generator to x. Used to repeat experiments
"""
    
def getOptions():  
    global unigramFilename, vec_file, doc_filename, K, N0, max_l, init_l, topD, alpha0, alpha1, delta
    global MAX_EM_ITERS, topicDiff_tolerance, zero_topic0, appendLogfile, seed
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"k:v:i:u:l:s:Ah")
        if len(args) != 1:
            raise getopt.GetoptError("")
        doc_filename = args[0]
        for opt, arg in opts:
            if opt == '-k':
                K = int(arg)
            if opt == '-v':
                vec_file = arg
            if opt == '-a':
                alpha1 = float(opt)
            if opt == '-i':
                MAX_EM_ITERS = int(arg)
            if opt == '-u':
                unigramFilename = arg
            if opt == '-l':
                max_l = int(arg)
            if opt == '-s':
                seed = int(arg)
            if opt == '-A':
                appendLogfile = True
            if opt == '-h':
                usage()
                sys.exit(0)

    except getopt.GetoptError:
        usage()
        sys.exit(2)
        
# V: W x N0
# T: K x N0
# VT: W x K
# u: W x 1
# r: K x 1
# Pi: L x K
# sum_pi_v: K x N0
    
def calcLoglikelihood(alpha, theta, Pi, sum_pi_v, T, r):
    theta0 = np.sum(theta)
    #pdb.set_trace()
    entropy = np.sum( gammaln(theta) ) - gammaln(theta0)
    entropy += (theta0 - K) * psi(theta0) - np.sum( (theta-1) * psi(theta) )
    entropy -= np.sum( Pi * np.log(Pi) )
    # Em[k] = sum_j Pi[j][k]
    Em = np.sum( Pi, axis=0 )
    
    tutil.fileLogger.debug("Em:")
    tutil.fileLogger.debug(Em)
    
    Em_Ephi = ( Em + alpha - 1 ) * ( psi(theta) - psi(theta0) )
    sum_r_pi = np.dot( Em, r )
    loglike = entropy + np.sum(Em_Ephi) + np.trace( np.dot( T, sum_pi_v.T ) ) + sum_r_pi
    return loglike

def updateTheta(Pi, alpha):
    theta = np.sum( Pi, axis=0 ) + alpha
    return theta

# vocab is for debugging purpose only
def updatePi(theta, T, r, V, wids, vocab):
    psiTheta = psi(theta)
    L = len(wids)
    Pi = np.zeros( (L, K) )
    
    for i,wid in enumerate(wids):
        v = V[wid]
        # smooth the current vector using context (preceding) vectors
        if smoothing_context and i > 0:
            j = max( 0, i - smoothing_context )
            totalWeight = 1
            for x in xrange(j, i):
                v += context_weight * V[ wids[x] ]
                totalWeight += context_weight
            v /= totalWeight
            
        Tv = np.dot( T, v )
        Pi[i] = np.exp( psiTheta + Tv + r )
    
    Pi = normalize(Pi)    
#    pdb.set_trace()
    return Pi

def updateTopicEmbeddings(V, u, X, EV, Pi, sum_pi_v, T, r, delta):
    Em = np.sum( Pi, axis=0 )
    
    Em_expR = Em * np.exp(r)
    #pdb.set_trace()
    EV_XT = EV + np.dot( T, X  )
    diffMat = sum_pi_v - (EV_XT.T * Em_expR).T
    diffMat *= delta
    for k in xrange( len(diffMat) ):
        if max_grad > 0 and np.linalg.norm(diffMat[k]) > max_grad:
            diffMat[k] *= max_grad / np.linalg.norm(diffMat[k])
    T2 = T + diffMat

    maxTStep = np.max( np.linalg.norm( diffMat, axis=1 ) )
    
#    magT = np.linalg.norm( T, axis=1 )
#    magT2 = np.linalg.norm( T2, axis=1 )
#    magDiff = np.linalg.norm( delta * diffMat, axis=1 )
#    
#    tutil.fileLogger.debug( "Magnitudes:" )
#    tutil.fileLogger.debug( "T    : %s" %(magT) )
#    tutil.fileLogger.debug( "T2   : %s" %(magT2) )
#    tutil.fileLogger.debug( "Diff : %s" %(magDiff) )
    
    # max_l == 0: do not do normalization
    if max_l > 0:
        for k in xrange( len(T2) ):
            # do normalization only if the magnitude > max_l
            if np.linalg.norm(T2[k]) > max_l:
                T2[k] = max_l * normalizeF(T2[k])
    
    if zero_topic0:
        T2[0] = np.zeros(N0)
     
    r2 = tutil.calcTopicResiduals(T2, V, u)
    topicDiff = np.linalg.norm( T - T2 )        
    return T2, r2, topicDiff, maxTStep
            
def main():
    global unigramFilename, vec_file, doc_filename, K, N0, max_l, init_l, topD, alpha0, alpha1
    global delta, MAX_EM_ITERS, topicDiff_tolerance, zero_topic0, remove_stop

    alpha = np.array( [ alpha1 ] * K )

    vocab_dict = loadUnigramFile(unigramFilename)
    V, vocab, word2ID, skippedWords_whatever = load_embeddings(vec_file)
    # map of word -> id of all words with embeddings
    vocab_dict2 = {}
        
    # dimensionality of topic/word embeddings
    N0 = V.shape[1]
    # number of all words
    vocab_size = V.shape[0]
    
    # set unigram probs
    u = np.zeros(vocab_size)
    
    #pdb.set_trace()
    
    for wid,w in enumerate(vocab):
        u[wid] = vocab_dict[w][2]
        vocab_dict2[w] = wid
        
    u = normalize(u)
    vocab_dict = vocab_dict2
        
    with open(doc_filename) as DOC:
        doc = DOC.readlines()
        doc = "".join(doc)
        
    wordsInSentences, wc = extractSentenceWords(doc, 2)
    print "%d words extracted" %wc
    
    widInSentences = []
    wids = []
    countedWC = 0
    outvocWC = 0
    stopwordWC = 0
    wid2freq = {}
    
    for sentence in wordsInSentences:
        currentSentWids = []
        
        for w in sentence:
            w = w.lower()
            if remove_stop and w in stopwordDict:
                stopwordWC += 1
                continue
                
            if w in vocab_dict:
                wid = vocab_dict[w]
                wids.append( wid )
                currentSentWids.append(wid)
                
                if wid not in wid2freq:
                    wid2freq[wid] = 1
                else:
                    wid2freq[wid] += 1
                countedWC += 1
            else:
                outvocWC += 1

        if len(currentSentWids) >= 1:
            widInSentences.append(currentSentWids)
            
    print "%d words kept, %d stop words, %d out voc" %( countedWC, stopwordWC, outvocWC )
    wid_freqs = sorted( wid2freq.items(), key=lambda kv: kv[1], reverse=True )
    tutil.screen_log_output("Top words:")
    for wid, freq in wid_freqs[:30]:
        print "%s(%d): %d" %( vocab[wid], wid, freq ),
        tutil.fileLogger.debug( "%s(%d): %d" %( vocab[wid], wid, freq ) )
    print
        
    T = np.zeros ( (K, N0) )
    
    if seed != 0:
        np.random.seed(seed)
        tutil.screen_log_output("Seed: %d" %seed )
        
    for i in xrange(0, K):
        T[i] = np.random.multivariate_normal( np.zeros(N0), np.eye(N0) )
        if init_l > 0:
            T[i] = init_l * normalizeF(T[i])
    
    if zero_topic0:
        T[0] = np.zeros(N0)
        alpha[0] = alpha0
            
#    sum_v = np.zeros(N0)
#    for wid in wids:
#        sum_v += V[wid]
#    
#    T[0] = max_l * normalizeF(sum_v)
    #tutil.fileLogger.debug("avg_v:")
    #tutil.fileLogger.debug(T[0])
    
    r = tutil.calcTopicResiduals(T, V, u)
    theta = np.ones(K)
    Pi = updatePi(theta, T, r, V, wids, vocab)
    theta = updateTheta(Pi, alpha)
    
    sum_pi_v = tutil.calcSum_pi_v(Pi, V, wids)
    loglike = calcLoglikelihood( alpha, theta, Pi, sum_pi_v, T, r )
    loglike2 = 0

    print "Precompute Ev and Evv...",
    
    Ev = np.dot(u, V)
    Evv = np.zeros( (N0, N0) )
    for wid in xrange(vocab_size):
        Evv += u[wid] * np.outer( V[wid], V[wid] )
    #X1 = np.linalg.inv(Evv)
    EV = np.tile( Ev, (K, 1) )
    
    print "Done."
    
    it = 0
    print "Iter %d Loglike: %.2f" %(it, loglike)
    tutil.fileLogger.debug( "Iter %d Loglike: %.2f" %(it, loglike) )
    
    while it == 0 or ( it < MAX_EM_ITERS and topicDiff > topicDiff_tolerance ):
    #while it == 0 or ( it < MAX_EM_ITERS and abs(loglike - loglike2) > loglike_tolerance ):
        tutil.fileLogger.debug( "EM Iter %d:" %it )
        
        T, r, topicDiff, maxTStep = updateTopicEmbeddings( V, u, Evv, EV, Pi, sum_pi_v, T, r, delta / ( it + 1 ) )
        theta = updateTheta(Pi, alpha)
        Pi = updatePi( theta, T, r, V, wids, vocab )
        sum_pi_v = tutil.calcSum_pi_v( Pi, V, wids )

        loglike2 = loglike
        loglike = calcLoglikelihood( alpha, theta, Pi, sum_pi_v, T, r )
        it += 1
        print "Iter %d: loglike %.2f, topicDiff %.4f, maxTStep %.3f" %(it, loglike, topicDiff, maxTStep)
        tutil.fileLogger.debug( "Iter %d: loglike %.2f, topicDiff %.4f, maxTStep %.3f" %(it, loglike, topicDiff, maxTStep) )
        
        if it % 5 == 0:
            Em = np.sum( Pi, axis=0 )
            principalK = np.argmax(Em)
            tutil.fileLogger.debug( "Principal T: %d" %principalK )
    
            tutil.fileLogger.debug( "T[:,%d]:" %topD )
            tutil.fileLogger.debug(T[:,:topD])
        
        if it % 20 == 0:
            tutil.printTopWordsInTopic( wids, vocab, Pi, V, T, wid2freq, False, topD )

    Em = np.sum( Pi, axis=0 )
    print "Em:\n%s\n" %(Em)
        
    tutil.printTopWordsInTopic( wids, vocab, Pi, V, T, wid2freq, True, topD )
    
    tutil.fileLogger.debug( "End at %s" %time.ctime() )
        
if __name__ == '__main__':
    np.seterr(all="raise")
    np.set_printoptions(threshold=np.nan)
    getOptions()
    tutil.fileLogger = initFileLogger( __file__, appendLogfile )
    tutil.fileLogger.debug( "Begin at %s" %time.ctime() )
    main()
