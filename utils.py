import numpy as np
import scipy.linalg
from scipy.stats.stats import spearmanr
import time
import re
import pdb
import sys
import os
import glob

class Timer(object):
    def __init__(self, name=None):
        self.name = name
        self.tstart = time.time()
        self.tlast = self.tstart
        self.firstCall = True
        
    def getElapseTime(self, isStr=True):
        totalElapsed = time.time() - self.tstart
        # elapsed time since last call
        interElapsed = time.time() - self.tlast
        self.tlast = time.time()
        
        firstCall = self.firstCall
        self.firstCall = False
        
        if isStr:
            if self.name:
                if firstCall:
                    return '%s elapsed: %.2f' % ( self.name, totalElapsed )
                return '%s elapsed: %.2f/%.2f' % ( self.name, totalElapsed, interElapsed )
            else:
                if firstCall:
                    return 'Elapsed: %.2f' % ( totalElapsed )
                return 'Elapsed: %.2f/%.2f' % ( totalElapsed, interElapsed )
        else:
            return totalElapsed, interElapsed
                        
    def printElapseTime(self):
        print self.getElapseTime()
           
# Weight: nonnegative real matrix. If not specified, return the unweighted norm
def norm1(M, Weight=None):
    if Weight is not None:
        s = np.sum( np.abs( M * Weight ) )
    else:
        s = np.sum( np.abs(M) )

    return s

def normF(M, Weight=None):
    if Weight is not None:
        # M*M: element-wise square
        s = np.sum( M * M * Weight )
    else:
        s = np.sum( M * M )

    return np.sqrt(s)

# given a list of matrices, return a list of their norms
def matSizes( norm, Ms, Weight=None ):
    sizes = []
    for M in Ms:
        sizes.append( norm(M, Weight) )

    return sizes

def sym(M):
    return ( M + M.T ) / 2.0

def skew(M):
    return ( M - M.T ) / 2.0

# Assume A has been approximately sorted by rows, and in each row, sorted by columns
# matrix F returned from loadBigramFile satisfies this
# print the number of elements >= A[0,0]/2^n
# return the idea cut point above which there are at least "fraction" of the elements
# these elements will be cut off to this upper limit
def getQuantileCut(A, fraction):
    totalElemCount = A.shape[0] * A.shape[1]
    maxElem = A[0,0]
    cutPoint = maxElem
    idealCutPoint = cutPoint
    idealFound = False

    while cutPoint >= 10:
        aboveElemCount = np.sum( A >= cutPoint )
        print "Cut point %.0f: %d/%.3f%%" %( cutPoint, aboveElemCount, aboveElemCount * 100.0 / totalElemCount )
        if not idealFound and aboveElemCount >= totalElemCount * fraction:
            idealCutPoint = cutPoint
            idealFound = True
        cutPoint /= 2.0

    return idealCutPoint

# find the principal eigenvalue/eigenvector: e1 & v1.
# if e1 < 0, then the left principal singular vector is -v1, and the right is v1.
# much faster than numpy.linalg.eig / scipy.linalg.eigh
def power_iter(M):
    MAXITER = 100
    epsilon = 1e-6
    vec = np.random.rand(len(M))
    old_vec = vec

    for i in xrange(MAXITER):
        vec2 = np.dot( M, vec )
        magnitude = np.linalg.norm(vec2)
        vec2 /= magnitude
        vec = vec2

        if i%2 == 1:
            error = np.linalg.norm( vec2 - old_vec )
            #print "%d: %f, %f" %( i+1, magnitude, error )
            if error < epsilon:
                break
            old_vec = vec2

    vec2 = np.dot( M, vec )
    if np.sum(vec2)/np.sum(vec) > 0:
        eigen = magnitude
    else:
        eigen = -magnitude

    return eigen, vec

# each column of vs is an eigenvector
def lowrank_fact(VV, N0):
    timer1 = Timer( "lowrank_fact()" )

    es, vs = np.linalg.eigh(VV)
    es = es[-N0:]
    vs = vs[ :, -N0: ]
    E_sqrt = np.diag( np.sqrt(es) )
    V = vs.dot(E_sqrt.T)
    VV = V.dot(V.T)

    return V, VV, vs,es

def save_embeddings( filename, vocab, V, matrixName ):
    FMAT = open(filename, "wb")
    print "Save matrix '%s' into %s" %(matrixName, filename)

    vocab_size = len(vocab)
    N = len(V[0])

    #pdb.set_trace()

    FMAT.write( "%d %d\n" %(vocab_size, N) )
    for i in xrange(vocab_size):
        line = vocab[i]
        for j in xrange(N):
            line += " %.5f" %V[i,j]
        FMAT.write("%s\n" %line)

    FMAT.close()

# for computational convenience, each row is an embedding vector
def load_embeddings( filename, maxWordCount=-1, extraWords={}, precision=np.float32 ):
    FMAT = open(filename)
    print "Load embedding text file '%s'" %(filename)
    V = []
    word2dim = {}
    vocab = []

    try:
        header = FMAT.readline()
        lineno = 1
        match = re.match( r"(\d+) (\d+)", header)
        if not match:
            raise ValueError(lineno, header)

        vocab_size = int(match.group(1))
        N = int(match.group(2))

        if maxWordCount > 0:
            maxWordCount = min(maxWordCount, vocab_size)
        else:
            maxWordCount = vocab_size

        print "%d extra words" %(len(extraWords))

        # maxWordCount + len(extraWords) is the maximum num of words.
        # V may contain extra rows that will be removed at the end
        V = np.zeros( (maxWordCount + len(extraWords), N), dtype=precision )
        wid = 0
        orig_wid = 0
        
        for line in FMAT:
            lineno += 1
            line = line.strip()
            # end of file
            if not line:
                if orig_wid != vocab_size:
                    raise ValueError( lineno, "%d words declared in header, but %d read" %(vocab_size, len(V)) )
                break

            orig_wid += 1
            fields = line.split(' ')
            fields = filter( lambda x: x, fields )
            w = fields[0]

            if orig_wid % 1000 == 0:
                print "\r%d    %d    \r" %( orig_wid, len(extraWords) ),
                
            if orig_wid >= maxWordCount and len(extraWords) == 0:
                break
            if orig_wid >= maxWordCount and w not in extraWords:
                continue

            V[wid] = np.array( [ float(x) for x in fields[1:] ], dtype=precision )
            word2dim[w] = wid
            vocab.append(w)
            wid += 1
            if w in extraWords:
                del extraWords[w]

    except ValueError, e:
        if len( e.args ) == 2:
            print "Unknown line %d:\n%s" %( e.args[0], e.args[1] )
        else:
            exc_type, exc_obj, tb = sys.exc_info()
            print "Source line %d - %s on File line %d:\n%s" %( tb.tb_lineno, e, lineno, line )
        exit(2)

    FMAT.close()
    print "%d embeddings read, %d kept" %(orig_wid, wid)

    if wid < len(V):
        V = V[0:wid]
    
    # V: embeddings, vocab: array of words, word2dim: dict of word to index in V        
    return V, vocab, word2dim

# borrowed from gensim.models.word2vec
def load_embeddings_bin( filename, maxWordCount=-1, extraWords={}, precision=np.float32 ):
    print "Load embedding binary file '%s'" %(filename)
    word2dim = {}
    vocab = []
    #origWord2dim = {}
    #origVocab = []

    with open(filename, "rb") as fin:
        header = fin.readline()
        vocab_size, N = map(int, header.split())

        if maxWordCount > 0:
            maxWordCount = min(maxWordCount, vocab_size)
        else:
            maxWordCount = vocab_size

        print "max %d words, %d extra words" %( maxWordCount, len(extraWords) )
        # maxWordCount + len(extraWords) is the maximum num of words.
        # V may contain extra rows that will be removed at the end
        V = np.zeros( (maxWordCount + len(extraWords), N), dtype=precision )

        full_binvec_len = np.dtype(precision).itemsize * N

        #pdb.set_trace()
        orig_wid = 0
        wid = 0
        while True:
            # mixed text and binary: read text first, then binary
            word = []
            while True:
                ch = fin.read(1)
                if ch == ' ':
                    break
                if ch != '\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                    word.append(ch)
            word = b''.join(word)

            if word[0].isupper():
                word2 = word.lower()
                # if the lowercased word hasn't been read, treat the embedding as the lowercased word's
                # otherwise, add the capitalized word to V
                if word2 not in word2dim:
                    word = word2

            #origWord2dim[word] = orig_wid
            #origVocab.append(word)

            orig_wid += 1
            if orig_wid % 1000 == 0:
                print "\r%d    %d    \r" %( orig_wid, len(extraWords) ),
            if orig_wid >= vocab_size:
                break

            if orig_wid >= maxWordCount and len(extraWords) == 0:
                break

            if orig_wid >= maxWordCount and word not in extraWords:
                fin.read(full_binvec_len)
                continue

            word2dim[word] = wid
            vocab.append(word)
            V[wid] = np.fromstring( fin.read(full_binvec_len), dtype=precision )
            wid += 1
            if word in extraWords:
                del extraWords[word]

    if wid < len(V):
        V = V[0:wid]
    print "%d embeddings read, %d embeddings kept" %(orig_wid, wid)
    
    # V: embeddings, vocab: array of words, word2dim: dict of word to index in V        
    return V, vocab, word2dim

def loadBigramFile(bigram_filename, topWordNum, extraWords, kappa=0.01):
    print "Loading bigram file '%s':" %bigram_filename
    BIGRAM = open(bigram_filename)
    lineno = 0
    vocab = []
    word2dim = {}
    # 1: headers, 2: bigrams. for error msg printing
    stage = 1
    # must be True. Otherwise some entries in logb_j will be log of 0
    # After smoothing, entries in b_j are always positive, thus logb_j is fine
    # To reduce code modifications, this flag is not removed
    do_smoothing=True
    timer1 = Timer( "loadBigramFile()" )

    try:
        header = BIGRAM.readline()
        lineno += 1
        match = re.match( r"# (\d+) words, \d+ occurrences", header )
        if not match:
            raise ValueError(lineno, header)

        wholeVocabSize = int(match.group(1))
        print "Totally %d words"  %wholeVocabSize
        # If topWordNum < 0, read all focus words
        if topWordNum < 0:
            topWordNum = wholeVocabSize

        # skip params
        header = BIGRAM.readline()
        header = BIGRAM.readline()
        lineno += 2

        match = re.match( r"# (\d+) bigram occurrences", header)
        if not match:
            raise ValueError(lineno, header)

        header = BIGRAM.readline()
        lineno += 1

        if header[0:6] != "Words:":
            raise ValueError(lineno, header)

        # vector log_u, unigram log-probs
        log_u = []

        i = 0
        wc = 0
        # Read the word list, build the word2dim mapping
        # Keep first topWordNum words and words in extraWords, if any
        while True:
            header = BIGRAM.readline()
            lineno += 1
            header = header.rstrip()

            # "Words" field ends
            if not header:
                break

            words = header.split("\t")
            for word in words:
                w, freq, log_ui = word.split(",")
                if i < topWordNum or w in extraWords:
                    word2dim[w] = i
                    log_u.append(float(log_ui))
                    vocab.append(w)
                    i += 1
                wc += 1

        # Usually these two should match, unless the bigram file is corrupted
        if wc != wholeVocabSize:
            raise ValueError( "%d words declared in header, but %d seen" %(wholeVocabSize, wc) )

        vocab_size = len(vocab)
        print "%d words seen, top %d & %d extra to keep. %d kept" %( wholeVocabSize, topWordNum, len(extraWords), vocab_size )

        log_u = np.array(log_u)
        u = np.exp(log_u)
        # renormalize unigram probs
        if topWordNum < wholeVocabSize:
            u = u / np.sum(u)
            log_u = np.log(u)

        k_u = kappa * u
        # original B, without smoothing
        #B = []
        G = np.zeros( (vocab_size, vocab_size), dtype=np.float32 )
        F = np.zeros( (vocab_size, vocab_size), dtype=np.float32 )

        header = BIGRAM.readline()
        lineno += 1

        if header[0:8] != "Bigrams:":
            raise ValueError(lineno, header)

        print "Read bigrams:"
        stage = 2

        line = BIGRAM.readline()
        lineno += 1
        wid = 0

        while True:
            line = line.strip()
            # end of file
            if not line:
                break

            # If we have read the bigrams of all the wanted words
            if wid == vocab_size:
                # if some words in extraWords are not read, there is bug
                break

            orig_wid, w, neighborCount, freq, cutoffFreq = line.split(",")
            orig_wid = int(orig_wid)

            if orig_wid % 500 == 0:
                print "\r%d\r" %orig_wid,

            if orig_wid <= topWordNum or w in extraWords:
                readNeighbors = True
                # remove it from the extra list, as a double-check measure
                # when all wanted words are read, the extra list should be empty
                if w in extraWords:
                    del extraWords[w]
            else:
                readNeighbors = False

            # x_{.j}
            x_j = np.zeros(vocab_size, dtype=np.float32)
            skipRemainingNeighbors = False
            
            while True:
                line = BIGRAM.readline()
                lineno += 1

                # Empty line. Should be end of file
                if not line:
                    break

                # A comment. Just in case of future extension
                # Currently only the last line in the file is a comment
                if line[0] == '#':
                    continue

                # beginning of the next word. Continue at the outer loop
                # Neighbor lines always start with '\t'
                if line[0] != '\t':
                    break

                # if the current context word is not wanted, skip these lines
                if not readNeighbors or skipRemainingNeighbors:
                    continue

                line = line.strip()
                neighbors = line.split("\t")
                for neighbor in neighbors:
                    w2, freq2, log_bij = neighbor.split(",")
                    if w2 in word2dim:
                        i = word2dim[w2]
                        x_j[i] = int(freq2)
                    # when meeting the first focus word not in vocab, all following focus words are not in vocab
                    # since neighbors are sorted ascendingly by ID
                    # So they are skipped to speed up reading
                    else:
                        skipRemainingNeighbors = True
                        break
                        
            # B stores original probs
            #B.append( x_j / np.sum(x_j) )

            # only push to F & G when this word is wanted
            if readNeighbors:
                # append a copy of x_j by * 1
                # otherwise only append a pointer. The contents may be changed accidentally elsewhere
                # the freqs are transformed and used as weights

                # smoothing using ( total freq of w )^0.7
                if do_smoothing:
                    x_j_norm1 = norm1(x_j)
                    utrans = x_j_norm1 * k_u
                    x_j += utrans
                    #x_j_norm2 = norm1(x_j)
                    #smooth_norm = norm1(utrans)
                    #if wid % 50 == 0:
                    #    print "%d,%d: smoothing %.5f/%.5f. %d -> %d" %( orig_wid, wid+1, smooth_norm, smooth_norm/x_j_norm1,
                    #                                                        x_j_norm1, x_j_norm2 )

                F[wid] = x_j

                # normalization
                b_j = x_j / np.sum(x_j)

                logb_j = np.log(b_j)
                G[wid] = logb_j - log_u
                wid += 1

    except ValueError, e:
        if len( e.args ) == 2:
            print "Unknown line %d:\n%s" %( e.args[0], e.args[1] )
        else:
            exc_type, exc_obj, tb = sys.exc_info()
            print "Source line %d: %s" %(tb.tb_lineno, e)
            if stage == 1:
                print header
            else:
                print line
        exit(0)

    print
    BIGRAM.close()

    return vocab, word2dim, G, F, u

# If core_size == -1, the core words are all demanded vocab words
# If vocab_size == -1, the demanded vocab words are all words in the bigram file
# If test_alg=True, load three blocks, as well as the whole matrix in G[3]&F[3]
def loadBigramFileInBlock(bigram_filename, core_size, vocab_size=-1, kappa=0.01, test_alg=False):
    print "Loading bigram file '%s' into 3 blocks:" %bigram_filename
    BIGRAM = open(bigram_filename)
    lineno = 0
    vocab = []
    word2dim_all = {}
    word2dim_core = {}
    
    # 1: headers, 2: bigrams. for error msg printing
    stage = 1
    # must be True. Otherwise some entries in logb_j will be log of 0
    # After smoothing, entries in b_j are always positive, thus logb_j is fine
    # To reduce code modifications, this flag is not removed
    do_smoothing=True
    timer1 = Timer( "loadBigramFileInBlock()" )

    #pdb.set_trace()
    
    try:
        header = BIGRAM.readline()
        lineno += 1
        match = re.match( r"# (\d+) words, \d+ occurrences", header )
        if not match:
            raise ValueError(lineno, header)

        wholeVocabSize = int(match.group(1))
        print "Totally %d words"  %wholeVocabSize
        
        if vocab_size > wholeVocabSize:
            raise ValueError( "%d words demanded, but only %d declared in header" %(vocab_size, wholeVocabSize) )
        # all the words are included in the vocab
        if vocab_size < 0:
            vocab_size = wholeVocabSize
                
        # If core_size < 0, the core block is the whole matrix
        # the returned upperright, lowerleft blocks will be empty
        if core_size < 0:
            core_size = vocab_size
        elif core_size > vocab_size:
            raise ValueError( "%d core words > %d total demanded words" %(core_size, vocab_size) )
        
        # skip params
        header = BIGRAM.readline()
        header = BIGRAM.readline()
        lineno += 2

        match = re.match( r"# (\d+) bigram occurrences", header )
        if not match:
            raise ValueError(lineno, header)

        header = BIGRAM.readline()
        lineno += 1

        if header[0:6] != "Words:":
            raise ValueError(lineno, header)

        # vector log_u, log-probs of all unigrams (at most vocab_size unigrams)
        log_u0 = []

        wc = 0
        # Read the focus list, build the word2dim_all / word2dim_core mapping
        # Read all context words of the core_size words
        # Read top core_size context words of remaining words
        while True:
            header = BIGRAM.readline()
            lineno += 1
            header = header.rstrip()

            # "Words" field ends
            if not header:
                break

            if wc < vocab_size:
                words = header.split("\t")
                for word in words:
                    w, freq, log_ui = word.split(",")
                    word2dim_all[w] = wc
                    if wc < core_size:
                        word2dim_core[w] = wc
                        
                    log_u0.append(float(log_ui))
                    vocab.append(w)
                    wc += 1
                    if vocab_size == wc:
                        break
                    
        # Usually these two should match, unless the bigram file is corrupted
        if wc != vocab_size:
            raise ValueError( "%d words demanded, but only %d read" %(vocab_size, wc) )

        print "%d words in file, top %d to read into vocab, top %d are core" %( wholeVocabSize, vocab_size, core_size )

        log_u0 = np.array(log_u0)
        u0 = np.exp(log_u0)
        # unigram prob & logprob of the top core_size words
        u1 = u0[0:core_size]
        u1 = u1 / np.sum(u1)
        log_u1 = np.log(u1)
        
        k_u0 = kappa * u0
        k_u1 = kappa * u1
        remainNum = vocab_size - core_size
        # G1/F1: the upper block (later split into upperleft and upperright blocks)
        # G21/F21: the lowerleft block
        # the lower right block (biggest) G22/F22 is ignored
        G1 = np.zeros( (core_size, vocab_size), dtype=np.float32 )
        G21 = np.zeros( (remainNum, core_size), dtype=np.float32 )
        F1 = np.zeros( (core_size, vocab_size), dtype=np.float32 )
        F21 = np.zeros( (remainNum, core_size), dtype=np.float32 )
        if test_alg:
            G0 = np.zeros( (vocab_size, vocab_size), dtype=np.float32 )
            F0 = np.zeros( (vocab_size, vocab_size), dtype=np.float32 )
            
        header = BIGRAM.readline()
        lineno += 1

        if header[0:8] != "Bigrams:":
            raise ValueError(lineno, header)

        print "Read bigrams:"
        stage = 2

        line = BIGRAM.readline()
        lineno += 1
        contextWID = 0
        # focusIDLimit: the limit of neighbor wid
        # Currently read all neighbors
        focusIDLimit = vocab_size
        # wid offset of context words to store in F/G
        contextIDOffset = 0
        
        G = G1
        F = F1
        k_u = k_u0
        log_u = log_u0
        
        #pdb.set_trace()
        
        while True:
            line = line.strip()
            # end of file
            if not line:
                break

            # We have read the bigrams of all the wanted words
            if contextWID == vocab_size:
                break

            orig_wid, w, neighborCount, freq, cutoffFreq = line.split(",")
            orig_wid = int(orig_wid)

            if orig_wid % 300 == 0:
                print "\r%d\r" %orig_wid,

            if contextWID == core_size:
                focusIDLimit = core_size
                contextIDOffset = core_size
                G = G21
                F = F21
                k_u = k_u1
                log_u = log_u1
                print "%d core words are all read. Read remaining %d words" %(core_size, remainNum)
                
            # x_{.j}
            x_j = np.zeros(focusIDLimit, dtype=np.float32)
            if test_alg:
                x0_j = np.zeros(vocab_size, dtype=np.float32)
            
            skipRemainingNeighbors = False
            
            while True:
                line = BIGRAM.readline()
                lineno += 1

                # Empty line. Should be end of file
                if not line:
                    break

                # A comment. Just in case of future extension
                # Currently only the last line in the file is a comment
                if line[0] == '#':
                    continue

                # beginning of the next word. Continue at the outer loop
                # Neighbor lines always start with '\t'
                if line[0] != '\t':
                    break

                if skipRemainingNeighbors:
                    continue
                    
                line = line.strip()
                neighbors = line.split("\t")
                for neighbor in neighbors:
                    w2, freq2, log_bij = neighbor.split(",")

                    # when meeting the first focus word not in vocab, all following focus words are not in vocab
                    # since neighbors are sorted ascendingly by ID
                    # So they are skipped to speed up reading
                    if w2 not in word2dim_all:
                        skipRemainingNeighbors = True
                        break

                    i = word2dim_all[w2]
                    freq2 = int(freq2)
                    if i < focusIDLimit:
                        x_j[i] = freq2
                    if test_alg:
                        x0_j[i] = freq2
                    elif i >= focusIDLimit:
                        skipRemainingNeighbors = True
                        break
                    
            if do_smoothing:
                x_j_norm1 = norm1(x_j)
                utrans = x_j_norm1 * k_u
                x_j += utrans
                
                if test_alg:
                    x0_j_norm1 = norm1(x0_j)
                    u0trans = x0_j_norm1 * k_u0
                    x0_j += u0trans
                    
            F[ contextWID - contextIDOffset ] = x_j

            # normalization
            b_j = x_j / np.sum(x_j)
            logb_j = np.log(b_j)
            G[ contextWID - contextIDOffset ] = logb_j - log_u
            
            if test_alg:
                F0[contextWID] = x0_j
                b0_j = x0_j / np.sum(x0_j)
                logb0_j = np.log(b0_j)
                G0[contextWID] = logb0_j - log_u0
                
            contextWID += 1
                
    except ValueError, e:
        if len( e.args ) == 2:
            print "Unknown line %d:\n%s" %( e.args[0], e.args[1] )
        else:
            exc_type, exc_obj, tb = sys.exc_info()
            print "Source line %d: %s" %(tb.tb_lineno, e)
            if stage == 1:
                print header
            else:
                print line
        exit(0)

    print
    BIGRAM.close()

    G11 = G1[ :, :core_size ]
    G12 = G1[ :, core_size: ]
    F11 = F1[ :, :core_size ]
    F12 = F1[ :, core_size: ]
    
    if test_alg:
        return vocab, word2dim_all, word2dim_core, [ G11, G12, G21, G0 ], [ F11, F12, F21, F0 ], u0
    else:
        return vocab, word2dim_all, word2dim_core, [ G11, G12, G21 ], [ F11, F12, F21 ], u0
        
def loadUnigramFile(filename):
    UNI = open(filename)
    vocab_dict = {}
    i = 1
    for line in UNI:
        line = line.strip()
        if line[0] == '#':
            continue
        fields = line.split("\t")
                             # id, freq, log prob
        vocab_dict[ fields[0] ] = ( i, int(fields[1]), np.exp(float(fields[2])) )
        i += 1

    return vocab_dict

def loadExtraWordFile(filename):
    extraWords = {}
    with open(filename) as f:
        for line in f:
            w, wid = line.strip().split('\t')
            extraWords[w] = 1

    return extraWords

# borrowed from Omer Levy's code
# extraArgs is not used, only for API conformity
def loadSimTestset(path, extraArgs=None):
    testset = []
    print "Read sim testset " + path
    with open(path) as f:
        for line in f:
            x, y, sim = line.strip().lower().split()
            testset.append( [ x, y, float(sim) ] )
    return testset

def loadAnaTestset(path, extraArgs=None):
    testset = []
    print "Read analogy testset " + path
    
    if extraArgs is not None and 'skipPossessive' in extraArgs:
        skipPossessive = True
        possessive = 0
    else:
        skipPossessive = False
           
    with open(path) as f:
        for line in f:
            # skip possessive forms
            if skipPossessive and line.find("'") >= 0:
                possessive += 1
                continue
            a, a2, b, b2 = line.strip().lower().split()
            testset.append( [ a, a2, b, b2 ] )
    
    if skipPossessive:
        print "%d possessive pairs skipped" %possessive
        
    return testset

# available loaders: loadSimTestset, loadAnaTestset
def loadTestsets(loader, testsetDir, testsetNames, extraArgs=None):
    # always use unix style path
    testsetDir = testsetDir.replace("\\", "/")
    if testsetDir[-1] != '/':
        testsetDir += '/'

    if not os.path.isdir(testsetDir):
        print "ERR: Test set dir does not exist or is not a dir:\n" + testsetDir
        sys.exit(2)

    testsets = []
    if len(testsetNames) == 0:
        testsetNames = glob.glob( testsetDir + '*.txt' )
        if len(testsetNames) == 0:
            print "No testset ended with '.txt' is found in " + testsetDir
            sys.exit(2)
        testsetNames = map( lambda x: os.path.basename(x)[:-4], testsetNames )

    for testsetName in testsetNames:
        testset = loader( testsetDir + testsetName + ".txt", extraArgs )
        testsets.append(testset)

    return testsets

# "model" in methods below has to support two methods:
# model[w]: return the embedding of w
# model.similarity(x, y): return the cosine similarity between the embeddings of x and y
# realb2 is passed in only for debugging purpose
def predict_ana( model, a, a2, b, realb2 ):
    questWordIndices = [ model.word2dim[x] for x in (a,a2,b) ]
    # b2 is effectively iterating through the vocab. The row is all the cosine values
    b2a2 = model.sim_row(a2)
    b2a  = model.sim_row(a)
    b2b  = model.sim_row(b)
    addsims = b2a2 - b2a + b2b

    addsims[questWordIndices] = -10000

    iadd = np.nanargmax(addsims)
    b2add  = model.vocab[iadd]

    # For debugging purposes
    ia = model.word2dim[a]
    ia2 = model.word2dim[a2]
    ib = model.word2dim[b]
    ib2 = model.word2dim[realb2]
    realaddsim = addsims[ib2]

    mulsims = ( b2a2 + 1 ) * ( b2b + 1 ) / ( b2a + 1.001 )
    mulsims[questWordIndices] = -10000
    imul = np.nanargmax(mulsims)
    b2mul  = model.vocab[imul]

    return b2add, b2mul

''' baa2 = model[b] - model[a] + model[a2]
    baa2 = baa2/normF(baa2)
    sims2 = model.V.dot(baa2)
    dists1 = np.abs( model.V - baa2 ).dot( np.ones( model.V.shape[1] ) )

    sims2[questWordIndices] = -10000
    dists1[questWordIndices] = 10000

    i2 = np.nanargmax(sims2)
    b22 = model.vocab[i2]
    i1 = np.nanargmin(dists1)
    b21 = model.vocab[i1]

    realsim2 = sims2[ib2]
    realdist1 = dists1[ib2]

    # F-norm (L2)
    topIDs2 = sims2.argsort()[-5:][::-1]
    topwords2 = [ model.vocab[i] for i in topIDs2 ]
    topsims2 = sims2[topIDs2]

    # Manhattan distance (L1)
    topIDs1 = sims1.argsort()[-5:][::-1]
    topwords1 = [ model.vocab[i] for i in topIDs1 ]
    topsims1 = sims1[topIDs1]

    if b22 != realb2:
        print "%s,%s\t%s,[%s]" %(a,a2,b,realb2)
        print "%s,%f\t%s\t%s" %(b21,realsim1, str(topsims1), str(topwords1))
        print "%s,%f\t%s\t%s" %(b22,realsim2, str(topsims2), str(topwords2))
        print
        #pdb.set_trace()
    return b2add, b2mul, b21, b22
    '''

# vocab_dict is a vocabulary dict, usually bigger than model.vocab, loaded from a unigram file
# its purpose is to find absent words in the model
def evaluate_sim(model, testsets, testsetNames, getAbsentWords=False, vocab_dict=None, cutPoint=0 ):

    # words in the vocab but not in the model
    absentModelID2Word = {}
    # words not in the vocab (of coz not in the model)
    absentVocabWords = {}
    # words in the vocab but below the cutPoint (id > cutPoint), may be in or out of the model
    cutVocabWords = {}
    # a set of spearman coeffs, in the same order as in testsets
    spearmanCoeff = []

    for i,testset in enumerate(testsets):
        modelResults = []
        groundtruth = []

        for x, y, sim in testset:
            if vocab_dict and x in vocab_dict:
                xid = vocab_dict[x][0]
                if cutPoint > 0 and xid > cutPoint:
                    cutVocabWords[x] = 1

            if vocab_dict and y in vocab_dict:
                yid = vocab_dict[y][0]
                if cutPoint > 0 and yid > cutPoint:
                    cutVocabWords[y] = 1

            if x not in model:
                if getAbsentWords and x in vocab_dict:
                    absentModelID2Word[xid] = x
                else:
                    absentVocabWords[x] = 1
            elif y not in model:
                if getAbsentWords and y in vocab_dict:
                    absentModelID2Word[yid] = y
                else:
                    absentVocabWords[y] = 1
            else:
                modelResults.append( model.similarity(x, y) )
                groundtruth.append(sim)
                #print "%s %s: %.3f %.3f" %(x, y, modelResults[-1], sim)
        print "%s: %d test pairs, %d valid" %( testsetNames[i], len(testset), len(modelResults) ),
        spearmanCoeff.append( spearmanr(modelResults, groundtruth)[0] )
        print ", %.5f" %spearmanCoeff[-1]

    # return hashes directly, for ease of merge
    return spearmanCoeff, absentModelID2Word, absentVocabWords, cutVocabWords

# vocab_dict is a vocabulary dict, usually bigger than model.vocab, loaded from a unigram file
# its purpose is to find absent words in the model
def evaluate_ana(model, testsets, testsetNames, getAbsentWords=False, vocab_dict=None, cutPoint=0 ):
    # for words in the vocab but not in the model. mapping from words to IDs
    absentModelID2Word = {}
    # words not in the vocab (of coz not in the model)
    absentVocabWords = {}
    # words in the vocab but below the cutPoint (id > cutPoint), may be in or out of the model
    cutVocabWords = {}
    # a set of scores, in the same order as in testsets
    # each is a tuple (add_score, mul_score)
    anaScores = []
    
    #pdb.set_trace()

    for i,testset in enumerate(testsets):
        modelResults = []
        groundtruth = []

        correct_add = 0.0
        correct_mul = 0.0
        validPairNum = 0
        currentScores = np.array( [ 0.0, 0.0 ] )
        
        for j,analogy in enumerate(testset):

            allWordsPresent = True
            watchWhenWrong = False

            # check presence of all words in this test pair
            for x in analogy:
                if vocab_dict and x in vocab_dict:
                    xid = vocab_dict[x][0]
                    if cutPoint > 0 and xid > cutPoint:
                        cutVocabWords[x] = 1
                        watchWhenWrong = True

                if x not in model:
                    if vocab_dict and x in vocab_dict:
                        absentModelID2Word[ vocab_dict[x][0] ] = x
                    else:
                        absentVocabWords[x] = 1
                    allWordsPresent = False

            if allWordsPresent:
                a, a2, b, b2 = analogy
                b2add, b2mul = predict_ana( model, a, a2, b, b2 )
                validPairNum += 1
                if b2add == b2:
                    correct_add += 1
                elif watchWhenWrong:
                    print "%s~%s = %s~%s,%.3f (%s,%3f)" %( a, a2, b, b2, model.similarity(b,b2), b2_add, model.similarity(b,b2_add) )

                if b2mul == b2:
                    correct_mul += 1

                """if b2_mul == b2:
                    correct_mul += 1
                if b2_L1 == b2:
                    correct_L1 += 1
                if b2_L2 == b2:
                    correct_L2 += 1
                currentScores = np.array([ correct_add, correct_mul, correct_L1, correct_L2 ]) / validPairNum
                """

                # latest cumulative scores
                currentScores = np.array( [ correct_add, correct_mul ] ) / validPairNum

            if j % 500 == 499:
                print "\r%i/%i/%i: Add %.5f, Mul %.5f\r" %( j + 1, validPairNum, len(testset), 
                                                            currentScores[0], currentScores[1] ),

        print "\n%s: %d analogies, %d valid" %( testsetNames[i], len(testset), validPairNum ),
        anaScores.append(currentScores)
        print ". Add Score: %.5f, Mul Score: %.5f" %( currentScores[0], currentScores[1] )

    return anaScores, absentModelID2Word, absentVocabWords, cutVocabWords

def bench(func, N, topEigenNum=0):
    print "Begin to factorize a %dx%d matrix" %(N,N)
    a = np.random.randn(N, N)
    a = (a+a.T)/2
    tic = time.clock()
    func(a)
    toc = time.clock()
    diff = toc - tic
    print "Elapsed time is %.3f" %diff
    return diff

class VecModel:
    def __init__(self, V, vocab, word2dim, vecNormalize=True):
        self.Vorig = V
        self.V = np.array([ x/normF(x) for x in self.Vorig ])
        self.word2dim = word2dim
        self.vecNormalize = vecNormalize
        self.vocab = vocab
        self.iterIndex = 0
        self.cosTable = None

    def __contains__(self, w):
        return w in self.word2dim

    def __getitem__(self, w):
        if w not in self:
            return None
        else:
            if self.vecNormalize:
                return self.V[ self.word2dim[w] ]
            else:
                return self.Vorig[ self.word2dim[w] ]

    def orig(self, w):
        if w not in self:
            return None
        else:
            return self.Vorig[ self.word2dim[w] ]

    def precompute_cosine(self):
        print "Precompute cosine matrix...",
        self.cosTable = np.dot( self.V, self.V.T )
        print "Done."

    def similarity(self, x, y):
        if x not in self or y not in self:
            return 0

        if self.vecNormalize:
            if self.cosTable is not None:
                ix = self.word2dim[x]
                iy = self.word2dim[y]
                return self.cosTable[ix,iy]
            return np.dot( self[x], self[y] )

        # when vectors are not normalized, return the raw dot product
        vx = self[x]
        vy = self[y]
        # vector too short. the similarity doesn't make sense
        if normF(vx) <= 1e-6 or normF(vy) <= 1e-6:
            return 0

        return np.dot( self[x], self[y] )

    def sim_row(self, x):
        if x not in self:
            return 0

        if self.vecNormalize:
            if self.cosTable is None:
                self.precompute_cosine()
            ix = self.word2dim[x]
            return self.cosTable[ix]

        vx = self[x]
        # vector too short. the dot product similarity doesn't make sense
        if normF(vx) <= 1e-6:
            return np.zeros( len(self.vocab) )

        return self.V.dot(vx)

