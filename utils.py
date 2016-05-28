# -*- coding=GBK -*-

import numpy as np
import scipy.linalg
from scipy.stats.stats import spearmanr
import time
import re
import pdb
import sys
import os
import glob
import logging
from psutil import virtual_memory
import os.path
import random
import unicodedata
import sys

unicode_punc_tbl = dict.fromkeys( i for i in xrange(128, sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P') )

logging.basicConfig( level=logging.DEBUG )
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

def str2dict(s):
    wordlist = re.split( "\s+", s )
    return dict.fromkeys(wordlist, 1)

stopwordStr = '''a about above across after again against all almost alone along also
although always am among an and another any anybody anyone anything
apart are around as  at away be because been before behind being below
besides between beyond both but by can cannot could  did do does doing done
down  during each either else enough etc  ever every everybody
everyone except far few for  from get gets got had has have having
he her here herself him himself his how however if in indeed instead into
is it its itself just kept me maybe might  more most mostly much must
my myself  neither  no nobody none nor not nothing  of off often on one
only onto or other others ought our ours out  own  please
pp quite rather really said seem  shall she should since so
some somebody somewhat still such than that the their theirs them themselves
then there therefore these they this thorough thoroughly those through thus to
together too toward towards until up upon was we well were what
whatever when whenever where whether which while who whom whose why will with
within would yet you your yourself
re d ll m ve t s'''

stopwordDict = str2dict(stopwordStr)

np.seterr(all="raise")
np.set_printoptions(suppress=True, threshold=np.nan, precision=3)

def initConsoleLogger(loggerName):
    consoleLogger = logging.getLogger(loggerName)
    streamHandler = logging.StreamHandler()
    consoleLogger.addHandler(streamHandler)  
    return consoleLogger
    
def initFileLogger(loggerName, isAppending=False):
    loggerName = os.path.splitext(loggerName)[0]
    currDate = timeToStr( time.time(), "%m.%d" )
    filename = "%s-%s.log" %( loggerName, currDate )
    sn = 0
    while os.path.isfile(filename):
        sn += 1
        filename = "%s-%s-%d.log" %( loggerName, currDate, sn )
  
    fileLogger = logging.getLogger(loggerName)
    if isAppending:
        mode = 'a'
    else:
        mode = 'w'
        
    fileHandler = logging.FileHandler(filename, mode=mode)
    fileLogger.addHandler(fileHandler)
    return fileLogger
    
def warning(*objs):
    sys.stderr.write(*objs)
    
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

def timeToStr(timeNum, fmt="%H:%M:%S"):
    timeStr = time.strftime(fmt, time.localtime(timeNum))
    return timeStr

# Weight: nonnegative real matrix. If not specified, return the unweighted norm
def norm1(M, Weight=None):
    if len(M.shape) == 1:
        if Weight is not None:
            return np.sum( np.abs( M * Weight ) )
        else:
            return np.sum( np.abs(M) )

    s = 0
    
    if Weight is not None:
        for i in xrange( len(M) ):
            # row by row calculation. 
            # If doing matrix multiplication, a big temporary matrix will be generated, consuming a lot RAM
            row = np.abs( M[i] * Weight[i] )
            s += np.sum(row)
    else:
        for i in xrange( len(M) ):
            row = np.abs( M[i] )
            s += np.sum(row)

    return s

def normF(M, Weight=None):
    if len(M.shape) == 1:
        if Weight is not None:
            # M*M makes all elems positive, and all elems of Weight are nonnegative. So no need to take abs()
            return np.sqrt( np.sum( M * M * Weight ) )
        else:
            return np.sqrt( np.sum( M * M ) )
    
    s = 0
    
    if Weight is not None:
        for i in xrange( len(M) ):
            # row by row calculation. 
            # If doing matrix multiplication, a big temporary matrix will be generated, consuming a lot RAM
            row = M[i] * M[i] * Weight[i]
            s += np.sum(row)
    else:
        for i in xrange( len(M) ):
            row = M[i] * M[i]
            s += np.sum(row)

    return np.sqrt(s)

# normalize a 1-d or 2-d array of nonnegative numbers: 
# keep the original array intact, return a copy of normalized array
# when array is 2d:
# axis=0: normalize columns. axis=1: normalize rows (default)
def normalize(data, axis=1):
    if np.min(data) < 0:
        raise RuntimeError("Negative element in data passed to normalize()")
    if data.ndim == 1:
        return data / np.sum(data)
    if axis == 0:
        s = np.sum(data, axis=0)
        return data / s
    elif axis == 1:
        ss = np.sum(data, axis=1)
        return data / np.tile(ss, (data.shape[1],1)).T
    else:
        raise RuntimeError('function normalize: axis must be 0/1')

# normalize a 1-d or 2-d array of numbers, w.r.t. F-norm
def normalizeF(data, axis=1):
    if data.ndim == 1:
        return data / normF(data) 
    
    data2 = np.copy(data)
    if axis == 0:
        for i in xrange(data2.shape[1]):
            if normF(data2[:,i]) > 0:
                data2[:,i] /= normF(data2[:,i])
        return data2
        
    elif axis == 1:
        for i in xrange(data2.shape[0]):
            if normF(data2[i]) > 0:
                data2[i] /= normF(data2[i])
        return data2                
    else:
        raise RuntimeError('function normalize: axis must be 0/1')

def cosine(x, y):
    x2 = normalizeF(x)
    y2 = normalizeF(y)
    return np.dot(x2, y2)

# Given a list of matrices, return a list of their norms
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
    totalNonzeroElemCount = np.sum( A > 0 ) #A.shape[0] * A.shape[1]
    maxElem = A[0,0]
    cutPoint = maxElem
    idealCutPoint = cutPoint
    idealFound = False
    
    while cutPoint >= 10:
        aboveElemCount = np.sum( A >= cutPoint )
        print "Cut point %.0f: %d/%.3f%%" %( cutPoint, aboveElemCount, aboveElemCount * 100.0 / totalNonzeroElemCount )
        if not idealFound and aboveElemCount >= totalNonzeroElemCount * fraction:
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
# It's a prerequisite that all eigenvalues are already nonnegative
# This requirement is guaranteed after nowe_factorize()
def lowrank_fact(VV, N0):
    timer1 = Timer( "lowrank_fact()" )

    es, vs = np.linalg.eigh(VV)
    es = es[-N0:]
    vs = vs[ :, -N0: ]
    E_sqrt = np.diag( np.sqrt(es) )
    V = vs.dot(E_sqrt)
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

def save_matrix_as_text( filename, rowTypeName, T, *extraCols, **kwargs ):
    FMAT = open(filename, "wb")
    print "Save %s matrix into '%s'" %(rowTypeName, filename)
    colSep = kwargs.get("colSep", " ")
    
    K, N = T.shape

    #pdb.set_trace()
    extraColNum = len(extraCols)
    
    FMAT.write( "%d %d %d\n" %( K, N, extraColNum ) )
    for i in xrange(K):
        # if rowNames is provided, print the corresponding row name at the beginning of each line
        line = ""
        for j in xrange(extraColNum):
            col = str( extraCols[j][i] )
            line += col + colSep
        line += "%.5f" %T[i,0]
            
        for j in xrange(1, N):
            line += " %.5f" %T[i,j]
        FMAT.write("%s\n" %line)

    FMAT.close()
    print "%d rows of %s(s) (%d-d each) saved" %( K, rowTypeName, N )

def load_matrix_from_text( filename, rowTypeName, colSep=" " ):
    FMAT = open(filename)
    print "Load %s matrix from '%s'" %(rowTypeName, filename)
    precision = np.float64
    extraCols = []
    extraColNum = 0
    lineno = 0
    
    try:
        header = FMAT.readline()
        rowID = 0
        lineno += 1
        match = re.match( r"(\d+) (\d+) (\d+)", header)
        if not match:
            raise ValueError(lineno, header)

        K = int(match.group(1))
        N = int(match.group(2))
        extraColNum = int(match.group(3))
        print "'%s': %dx%d, %d extra columns" %( filename, K, N, extraColNum )
        
        for i in xrange(extraColNum):
            extraCols.append([])
        
        M = np.zeros( (K, N), dtype=precision )

        for line in FMAT:
            line = line.strip()
            # end of file
            if not line:
                if rowID != K:
                    raise ValueError( lineno, "%d rows declared in header, but %d read" %( K, rowID ) )
                break

            row_extraCols = []
            fields = line.split(colSep)
            for i in xrange(extraColNum):
                extraCols[i].append(fields[i])
                
            matFields = fields[extraColNum:]
            # matrix values are always concatenated by " "
            # if colSep is not " ", matrix values should take one column
            if colSep != " ":
                if len(matFields) > 1:
                    raise ValueError( lineno, "%d columns of matrix values when colSep is not space" %( len(matFields) ) )
                else:
                    matFields = matFields[0].split(" ")
                    
            M[rowID] = np.array( [ float(x) for x in matFields ], dtype=precision )
            rowID += 1
                
    except ValueError, e:
        if len( e.args ) == 2:
            warning( "Unknown line %d:\n%s\n" %( e.args[0], e.args[1] ) )
        else:
            exc_type, exc_obj, tb = sys.exc_info()
            warning( "Source line %d - %s on File line %d:\n%s\n" %( tb.tb_lineno, e, lineno, line ) )
        exit(2)

    FMAT.close()
    warning( "%dx%d %s matrix loaded from '%s'\n" %(K, N, rowTypeName, filename) )

    if len(extraCols) == 0:
        return M
    else:
        return M, extraCols
        
# load top maxWordCount words, plus extraWords
def load_embeddings( filename, maxWordCount=-1, extraWords={}, record_skipped=False ):
    FMAT = open(filename)
    warning( "Load embedding text file '%s'\n" %(filename) )
    
    V = []
    word2id = {}
    skippedWords = {}

    vocab = []
    precision = np.float32

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

        warning( "Will load embeddings of %d words" %maxWordCount )
        if len(extraWords) > 0:
            warning( ", plus %d extra words" %(len(extraWords)) )
        warning("\n")

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
                    raise ValueError( lineno, "%d words declared in header, but %d read" %( vocab_size, orig_wid ) )
                break

            fields = line.split(' ')
            # remove empty fields
            fields = filter( lambda x: x, fields )
            w = fields[0]

            if w in extraWords:
                del extraWords[w]
                isInterested = True
            elif orig_wid < maxWordCount:
                isInterested = True
            elif record_skipped:
                isInterested = False
                skippedWords[w] = 1
            else:
                break
							
            orig_wid += 1

            if isInterested:
                V[wid] = np.array( [ float(x) for x in fields[1:] ], dtype=precision )
                word2id[w] = wid
                vocab.append(w)
                wid += 1

            if orig_wid % 1000 == 0:
                warning( "\r%d    %d    %d    \r" %( orig_wid, wid, len(extraWords) ) )

            if orig_wid > vocab_size:
                raise ValueError( "%d words declared in header, but more are read" %(vocab_size) )

    except ValueError, e:
        if len( e.args ) == 2:
            warning( "Unknown line %d:\n%s\n" %( e.args[0], e.args[1] ) )
        else:
            exc_type, exc_obj, tb = sys.exc_info()
            warning( "Source line %d - %s on File line %d:\n%s\n" %( tb.tb_lineno, e, lineno, line ) )
        exit(2)

    FMAT.close()
    warning( "\n%d embeddings read, %d kept\n" %(orig_wid, wid) )

    #pdb.set_trace()

    if wid < len(V):
        V = V[:wid]

    # V: embeddings, vocab: array of words, word2id: dict of word to index in V
    return V, vocab, word2id, skippedWords

# borrowed from gensim.models.word2vec
# load top maxWordCount words, plus extraWords
def load_embeddings_bin( filename, maxWordCount=-1, extraWords={}, record_skipped=False ):
    print "Load embedding binary file '%s'" %(filename)
    word2id = {}
    skippedWords = {}
    vocab = []
    #origWord2id = {}
    #origVocab = []
    precision = np.float32

    with open(filename, "rb") as fin:
        header = fin.readline()
        vocab_size, N = map(int, header.split())

        if maxWordCount > 0:
            maxWordCount = min(maxWordCount, vocab_size)
        else:
            maxWordCount = vocab_size

        print "Will load embeddings of %d words" %maxWordCount,
        if len(extraWords) > 0:
            print "\b, plus %d extra words" %(len(extraWords))
        else:
            print

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
                if word2 not in word2id:
                    word = word2

            #origWord2id[word] = orig_wid
            #origVocab.append(word)

            if w in extraWords:
                del extraWords[w]
                isInterested = True
            elif orig_wid < maxWordCount:
                isInterested = True
            elif record_skipped:
                isInterested = False
                skippedWords[w] = 1
            else:
                break

            orig_wid += 1

            if isInterested:
                word2id[word] = wid
                vocab.append(word)
                V[wid] = np.fromstring( fin.read(full_binvec_len), dtype=precision )
                wid += 1
            else:
                fin.read(full_binvec_len)

            if orig_wid % 1000 == 0:
                print "\r%d    %d    %d    \r" %( orig_wid, wid, len(extraWords) ),

            if orig_wid > vocab_size:
                raise ValueError( "%d words declared in header, but more are read" %(vocab_size) )

    if wid < len(V):
        V = V[:wid]
    print "\n%d embeddings read, %d embeddings kept" %(orig_wid, wid)

    # V: embeddings, vocab: array of words, word2id: dict of word to index in V
    return V, vocab, word2id, skippedWords

# load Hyperwords embeddings
def load_embeddings_hyper(modelPath, vecType):
    sys.path.append('./hyperwords/hyperwords')
    from representations.explicit import PositiveExplicit
    from representations.embedding import SVDEmbedding
    print "Load Hyperwords(%s) embedding file '%s'" %(vecType, modelPath)
    if vecType == 'PPMI':
        base = PositiveExplicit
    else:
        base = SVDEmbedding
        
    class HyperEmbed(base):
        def __contains__(self, w):
            return w in self.wi
    
    model = HyperEmbed(modelPath, True)

    print "Done."
    return model
    
# load residuals
# the dict word2id is to ensure the same word is mapped to the same id as in the embedding file
# in other words, the embedding file and residual file had to be generated in the same batch
def load_residuals( filename, word2id={}, maxRowCount=-1, maxColCount=-1 ):
    FMAT = open(filename)
    warning( "Load residual file '%s'\n" %(filename) )
    
    precision = np.float32

    try:
        header = FMAT.readline()
        lineno = 1
        match = re.match( r"(\d+) (\d+)", header)
        if not match:
            raise ValueError(lineno, header)

        vocab_size = int(match.group(1))
        vocab_size2 = int(match.group(2))

        if maxRowCount > 0:
            maxRowCount = min(maxRowCount, vocab_size)
        else:
            maxRowCount = vocab_size

        if maxColCount > 0:
            maxColCount = min(maxColCount, vocab_size2)
        else:
            maxColCount = vocab_size2

        warning( "Will load residuals of %dx%d words" %( maxRowCount, maxColCount ) )

        A = np.zeros( (maxRowCount, maxColCount), dtype=precision )

        for line in FMAT:
            lineno += 1
            line = line.strip()
            # end of file
            if not line:
                if lineno != vocab_size:
                    raise ValueError( lineno, "%d rows declared in header, but %d read" %( vocab_size, lineno ) )
                break

            fields = line.split(' ')
            fields = filter( lambda x: x, fields )
            w = fields[0]

            if len(word2id) > 0:
                if w not in word2id or word2id[w] != lineno - 1:
                    raise ValueError("ID of '%s' is inconsistent between '%s' and the loaded embeddings. "
                                     "Make sure they were generated in the same batch" %(w, filename) )
            A[ lineno - 1 ] = np.array( [ float(x) for x in fields[1:] ], dtype=precision )

            if lineno % 1000 == 0:
                warning( "\r%d\r" %lineno )

            if lineno >= vocab_size:
                raise ValueError( "%d words declared in header, but more are read" %(vocab_size) )

    except ValueError, e:
        if len( e.args ) == 2:
            warning( "Unknown line %d:\n%s\n" %( e.args[0], e.args[1] ) )
        else:
            exc_type, exc_obj, tb = sys.exc_info()
            warning( "Source line %d - %s on File line %d:\n%s\n" %( tb.tb_lineno, e, lineno, line ) )
        exit(2)

    FMAT.close()
    warning( "\n%d rows read, each row %d words\n" %(lineno, vocab_size2) )

    #pdb.set_trace()

    return A

def loadBigramFile( bigram_filename, topWordNum, extraWords, kappa=0.01 ):
    print "Loading bigram file '%s':" %bigram_filename
    BIGRAM = open(bigram_filename)
    lineno = 0
    vocab = []
    word2id = {}
    # 1: headers, 2: bigrams. for error msg printing
    stage = 1
    # In order to disable smoothing, just set kappa to 0.
    # But when smoothing is disabled, some entries in logb_i will be log of 0
    # After smoothing, entries in b_i are always positive, thus logb_i is fine
    # do_smoothing=True

    timer1 = Timer( "loadBigramFile()" )

    try:
        header = BIGRAM.readline()
        lineno += 1
        match = re.match( r"#\s+(\d+) words,\s+\d+ occurrences", header )
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

        match = re.match( r"#\s+(\d+) bigram occurrences", header)
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
        # Read the word list, build the word2id mapping
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
                    word2id[w] = i
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
        contextWID = 0

        #pdb.set_trace()

        while True:
            line = line.strip()
            # end of file
            if not line:
                break

            # If we have read the bigrams of all the wanted words
            if contextWID == vocab_size:
                # if some words in extraWords are not read, there is bug
                break

            # word ID, word, number of distinct neighbors, sum of freqs of all neighbors, cut off freq
            orig_wid, w, neighborCount, neighborTotalOccur, cutoffFreq = line.split(",")
            orig_wid = int(orig_wid)
            neighborTotalOccur = float(neighborTotalOccur)

            if orig_wid % 200 == 0:
                print "\r%d\r" %orig_wid,

            if orig_wid <= topWordNum or w in extraWords:
                recordCurrWord = True
                # remove it from the extra list, as a double-check measure
                # when all wanted words are read, the extra list should be empty
                if w in extraWords:
                    del extraWords[w]
            else:
                recordCurrWord = False

            # x_{.j}
            x_i = np.zeros(vocab_size, dtype=np.float32)
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
                if not recordCurrWord or skipRemainingNeighbors:
                    continue

                line = line.strip()
                neighbors = line.split("\t")
                for neighbor in neighbors:
                    w2, freq2, log_bij = neighbor.split(",")
                    if w2 in word2id:
                        i = word2id[w2]
                        x_i[i] = int(freq2)
                    # when meeting the first focus word not in vocab, all following focus words are not in vocab
                    # since neighbors are sorted ascendingly by ID
                    # So they are skipped to speed up reading
                    else:
                        skipRemainingNeighbors = True
                        break

            # only save in F & G when this word is wanted
            if recordCurrWord:
                # Question: whether set F to the original freq or smoothed freq (assign F before or after smoothing)?
                F[contextWID] = x_i

                """
                x_i_norm1 = np.sum(x_i)
                utrans = x_i_norm1 * k_u
                x_i = x_i * (1 - kappa) + utrans

                # the smoothing shoudn't change the norm1 of x_i
                # i.e. x_i_norm1 = np.sum(x_i)
                # After normalization, b_i = ( normalized x_i )*( 1 - kappa ) + u * kappa
                b_i = x_i / np.sum(x_i)
                """

                x_i /= neighborTotalOccur
                b_i = x_i *( 1 - kappa ) + k_u
                g_i = np.log(b_i) - log_u
                G[contextWID] = g_i
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

    return vocab, word2id, G, F, u

# If noncore_size == -1, all noncore words are loaded into the upperright and lowerleft blocks
# word2preID_core are the IDs of words in the pretrained embedding file
# If vocab_core and word2preID_core are specified, core words are limited to words in them
# Otherwise the top core_size words are core words
def loadBigramFileInBlock( bigram_filename, core_size, noncore_size=-1, word2preID_core={}, prewords_skipped={}, kappa=0.02 ):

    # corewords_specified means the list of core words are specified in word2preID_core
    
    if len(word2preID_core) > 0:
        corewords_specified = True
        # recordUpperleft is always the negation of corewords_specified. But sometimes is semantically clearer
        recordUpperleft = False
        # this core_size is used in comparsion with the total word count in the header
        # this size might be inaccurate, as some words in word2preID_core might be missing from this bigram file
        core_size = len(word2preID_core)
    else:
        corewords_specified = False
        recordUpperleft = True
        # if core words are not specified, core_size should always > 0,
        # otherwise I don't know how many words are core words
        if core_size < 0:
            raise ValueError( "Argument error: core_size = %d < 0 when word2preID_core is not specified" %core_size )
        if len(prewords_skipped) > 0:
            raise ValueError( "Argument error: word2preID_core is empty but prewords_skipped is not" )

    # if corewords_specified, return a list of coreword IDs in the pretrained mapping
    # otherwise, return empty list (just for return value conformity)
    coreword_preIDs = []

    if not recordUpperleft:
        # do not record G11/F11
        print "Loading bigram file '%s' into 2 blocks. Will skip %d words" \
                                    %( bigram_filename, len(prewords_skipped) )
    else:
        print "Loading bigram file '%s' into 3 blocks." %bigram_filename

    BIGRAM = open(bigram_filename)

    lineno = 0
    vocab_all = []
    vocab_core = []
    vocab_noncore = []

    word2id_all = {}
    # origID is the original ID in this bigram file
    # preID is the ID in the pretrained vec file
    word2origID_all = {}
    word2id_noncore = {}
    word2id_core = {}

    # stage 1: header and unigrams, stage 2: bigrams. for error msg printing
    stage = 1
    # do_smoothing must be True. Otherwise some entries in logb_i will be log of 0
    # After smoothing, entries in b_i are always positive, thus logb_i is fine
    # To reduce code modifications, this flag is not removed
    timer1 = Timer( "loadBigramFileInBlock()" )

    #pdb.set_trace()

    try:
        header = BIGRAM.readline()
        lineno += 1
        match = re.match( r"#\s+(\d+) words,\s+\d+ occurrences", header )
        if not match:
            raise ValueError(lineno, header)

        wholeVocabSize = int(match.group(1))
        print "Totally %d words"  %wholeVocabSize

        if core_size >= wholeVocabSize:
            raise ValueError( "%d core words, but vocabulary only declares %d words in header" %( core_size, wholeVocabSize ) )

        # at least consider one noncore word
        min_vocab_size = core_size + max( noncore_size, 1 )
        if min_vocab_size > wholeVocabSize:
            warning( "%d (%d + %d) words demanded, but only %d declared in header" %( min_vocab_size, core_size,
                                                                                             max( noncore_size, 1 ), wholeVocabSize) )
            min_vocab_size = wholeVocabSize
            noncore_size2 = wholeVocabSize - core_size
            warning( "Noncore words adjusted from %d to %d" %( noncore_size, noncore_size2 ) )
            noncore_size = noncore_size2
            
        # all the words are included in the vocab_all
        # in this case, vocab_size needs to be initialized
        # otherwise corewords_specified, noncore_size & vocab_size will be computed later, 
        # needn't to be initialized
        if not corewords_specified:
            if noncore_size < 0:
                vocab_size = wholeVocabSize
                noncore_size = vocab_size - core_size
            else:
                # core_size will be updated later
                # some core words in word2preID_core may not be present in this bigram file
                vocab_size = core_size + noncore_size

        # skip params
        header = BIGRAM.readline()
        header = BIGRAM.readline()
        lineno += 2

        match = re.match( r"#\s+(\d+) bigram occurrences", header )
        if not match:
            raise ValueError(lineno, header)

        header = BIGRAM.readline()
        lineno += 1

        if header[0:6] != "Words:":
            raise ValueError(lineno, header)

        # vector log_u, log-probs of all unigrams (at most vocab_size unigrams)
        log_u0 = []
        log_u0_core = []
        log_u0_noncore = []
        wc = 0
        core_wc = 0
        noncore_wc = 0
        skipped_wc = 0
        # maximum ID in the original order of core words
        max_core_origID = 0

        # Read the focus list, build the word2id_all / word2id_core mapping
        # Read all context words of the core_size words
        # Read top core_size context words of remaining words
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

                # load core words only in word2preID_core, other words as noncore
                # core words may not be consecutive, they may be interspersed by noncore words
                # So put them into two sets of arrays
                if corewords_specified:
                    if w in word2preID_core:
                        word2id_core[w] = core_wc
                        core_wc += 1
                        coreword_preIDs.append( word2preID_core[w] )
                        log_u0_core.append(float(log_ui))
                        vocab_core.append(w)

                        if wc > max_core_origID:
                            max_core_origID = wc

                    elif w in prewords_skipped:
                        skipped_wc += 1
                    elif noncore_size < 0 or noncore_wc < noncore_size :
                        word2id_noncore[w] = noncore_wc
                        noncore_wc += 1
                        log_u0_noncore.append(float(log_ui))
                        vocab_noncore.append(w)
                        
                    word2origID_all[w] = wc
                    wc += 1

                # load fresh core words
                else:
                    if wc == vocab_size:
                        break

                    word2id_all[w] = wc
                    if wc < core_size:
                        word2id_core[w] = wc
                    else:
                        word2id_noncore[w] = wc - core_size

                    log_u0.append(float(log_ui))
                    vocab_all.append(w)
                    wc += 1

        if corewords_specified:
            # some core words may be missing from the bigram file. So recompute core_size
            # If these two don't match, then some specified core words don't appear in the unigram list
            if core_size != core_wc:
                print "WARN: %d core words demanded, but only %d read" %(core_size, core_wc)
                core_size = core_wc

            noncore_size = noncore_wc
            vocab_size = core_size + noncore_size
            log_u0 = log_u0_core + log_u0_noncore
            vocab_all = vocab_core + vocab_noncore

            word2id_all = word2id_core.copy()
            # insert noncore words into word2id_all
            # core ID \in [ 0, coresize - 1 ]
            # noncore ID \in [ core_size, ... ]
            for w in word2id_noncore:
                word2id_all[w] = word2id_noncore[w] + core_size

        else:
            # core words are consecutive. So orig ID = id
            max_core_origID = core_size
            word2origID_all = word2id_all
            if vocab_size > 0 and wc < vocab_size:
                print "WARN: %d words demanded, but only %d read" %(vocab_size, wc)
                vocab_size = wc

        print "%d words in file, top %d to read into vocab (%d core, %d noncore), %d skipped" \
               %( wholeVocabSize, vocab_size, core_size, noncore_size, skipped_wc )

        # unigram prob & logprob of all the words
        log_u0 = np.array(log_u0)
        u0 = np.exp(log_u0)
        # re-normalization is needed, as some unigrams may be out of vocab_all
        u0 /= np.sum(u0)

        log_u0 = np.log(u0)
        log_u_core    = log_u0[:core_size]
        log_u_noncore = log_u0[core_size:]

        k_u0 = kappa * u0
        k_u_core    = k_u0[:core_size]
        k_u_noncore = k_u0[core_size:]

        ### Reading bigrams begins ###
        header = BIGRAM.readline()
        lineno += 1

        if header[0:8] != "Bigrams:":
            raise ValueError(lineno, header)

        print "Read bigrams:"
        stage = 2

        line = BIGRAM.readline()
        lineno += 1

        contextWID = 0

        # new G12, F12
        # G12/F12: the upperright block
        G12 = np.zeros( (core_size, noncore_size), dtype=np.float32 )
        F12 = np.zeros( (core_size, noncore_size), dtype=np.float32 )
        # new G21, F21
        # G21/F21: the lowerleft block
        # the lower right block (biggest) G22/F22 is ignored
        G21 = np.zeros( (noncore_size, core_size), dtype=np.float32 )
        F21 = np.zeros( (noncore_size, core_size), dtype=np.float32 )

        # when reading core words, keep the upperleft block, the whole vocab
        if recordUpperleft:
            # new G11, F11
            # G11/F11: the upperleft block
            G11 = np.zeros( (core_size, core_size), dtype=np.float32 )
            F11 = np.zeros( (core_size, core_size), dtype=np.float32 )
            rowLength = vocab_size
            k_u = k_u0
            log_u = log_u0
        # when reading core words, only keep the upperright block, noncore_size words
        else:
            rowLength = noncore_size
            k_u = k_u_noncore
            log_u = log_u_noncore

        # focusIDLimit: the limit of neighbor wid
        # In the beginning, read all neighbors
        focusIDLimit = vocab_size

        # vars are initialized like those of a core word,
        # So pretend the previous nonexistent word is a core word
        lastContextIsCore = True

        #pdb.set_trace()
        core_readcount = 0
        noncore_readcount = 0
        coreMsg_printed = False

        while True:
            line = line.strip()
            # end of file
            if not line:
                break

            # We have read the bigrams of all the wanted words
            if contextWID == vocab_size:
                break

            # word ID, word, number of distinct neighbors, sum of freqs of all neighbors, cut off freq
            orig_wid, w, neighborCount, neighborTotalOccur, cutoffFreq = line.split(",")
            orig_wid = int(orig_wid)
            neighborTotalOccur = float(neighborTotalOccur)

            if w in word2id_core:
                # the current context word is a core word
                contextIsCore = True
                wid = word2id_core[w]
                # context type switches from noncore to core, update relevant variables
                # if context type doesn't change (last context is also core), then just keep vars unchanged
                # context type switching should only happen very few times
                if not lastContextIsCore:
                    focusIDLimit = vocab_size

                    if recordUpperleft:
                        rowLength = vocab_size
                        k_u = k_u0
                        log_u = log_u0
                    # when reading core words, only keep the upperright block, noncore_size words
                    else:
                        rowLength = noncore_size
                        k_u = k_u_noncore
                        log_u = log_u_noncore

                    lastContextIsCore = True

            elif w in word2id_noncore:
                contextIsCore = False
                wid = word2id_noncore[w]
                # context type switches from core to noncore, update relevant variables
                # if context type doesn't change (last context is also noncore), then just keep vars unchanged
                # context type switching should only happen very few times
                if lastContextIsCore:
                    # in a row of noncore (context) word, only freqs of core (focus) words are recorded
                    focusIDLimit = max_core_origID
                    rowLength = core_size

                    k_u = k_u_core
                    log_u = log_u_core

                    lastContextIsCore = False

            # x_{i.}
            x_i = np.zeros(rowLength, dtype=np.float32)

            if w in prewords_skipped:
                saveCurrRow = False
                skipRemainingNeighbors = True
            else:
                saveCurrRow = True
                skipRemainingNeighbors = False

            while True:
                line = BIGRAM.readline()
                lineno += 1

                # Empty line. Should be end of file
                if not line:
                    break

                # Encounter a comment. Just in case of future extension
                # Currently only the last line in the file is a comment after the header
                if line[0] == '#':
                    continue

                # Beginning of the next word. Continue at the outer loop
                # Neighbor lines always start with '\t'
                if line[0] != '\t':
                    break

                if skipRemainingNeighbors:
                    continue

                line = line.strip()
                neighbors = line.split("\t")

                for neighbor in neighbors:
                    w2, freq2, log_bij = neighbor.split(",")

                    # w2 in skip list, and surely not in word2id_all
                    # so check here to avoid setting skipRemainingNeighbors
                    if w2 in prewords_skipped:
                        continue
                        
                    # when meeting the first focus word out of vocab_all, all following focus words are not in vocab_all
                    # since neighbors are sorted ascendingly by ID
                    # So they are skipped to speed up reading
                    if w2 not in word2id_all:
                        skipRemainingNeighbors = True
                        break

                    origID = word2origID_all[w2]
                    # On a noncore row. Should have focus word orig ID <= max_core_origID
                    # origIDs of core words may be interspersed by origIDs of noncore words
                    # but IDs of core words are consecutive, and preceding IDs of noncore words
                    if not contextIsCore and origID > max_core_origID:
                        skipRemainingNeighbors = True
                        break

                    freq2 = int(freq2)
                    # On a core (context) row.
                    # If recordUpperleft, use the map from whole vocab to IDs;
                    # otherwise use the map from core words to IDs
                    if contextIsCore:
                        if recordUpperleft:
                            # w2id: id of w2
                            w2id = word2id_all[w2]
                            x_i[w2id] = freq2
                        # don't keep upperleft block. core (focus) words are discarded
                        # w2id \in [ 0, noncore_size - 1 ]
                        elif w2 in word2id_noncore:
                            w2id = word2id_noncore[w2]
                            x_i[w2id] = freq2
                    # On a noncore (context) row. Only record core (focus) words
                    elif w2 in word2id_core:
                        w2id = word2id_core[w2]
                        x_i[w2id] = freq2

            if not saveCurrRow:
                continue
                
            # Question: whether set F to the original freq or smoothed freq (assign F before or after smoothing)?
            if contextIsCore:
                if recordUpperleft:
                    F11[core_readcount] = x_i[:core_size]
                    F12[core_readcount] = x_i[core_size:]
                else:
                    # As w2id \in [ 0, noncore_size - 1 ], no offset is needed
                    F12[core_readcount] = x_i
            else:
                F21[noncore_readcount] = x_i

            """
            x_i_norm1 = np.sum(x_i)
            utrans = x_i_norm1 * k_u
            x_i = x_i * (1 - kappa) + utrans

            # the smoothing shoudn't change the norm1 of x_i
            # i.e. x_i_norm1 = np.sum(x_i)
            # normalization
            b_i = x_i / np.sum(x_i)
            """

            x_i /= neighborTotalOccur
            b_i = x_i * ( 1 - kappa ) + k_u
            g_i = np.log(b_i) - log_u

            if contextIsCore:
                if recordUpperleft:
                    G11[core_readcount] = g_i[:core_size]
                    G12[core_readcount] = g_i[core_size:]
                else:
                    # As w2id \in [ 0, noncore_size - 1 ], no offset is needed
                    G12[core_readcount] = g_i
            else:
                G21[noncore_readcount] = g_i

            contextWID += 1
            if contextIsCore:
                core_readcount += 1
            else:
                noncore_readcount += 1

            if orig_wid % 200 == 0:
                print "\r%d (%d core, %d noncore)\r" %( orig_wid, core_readcount, noncore_readcount ),
            if not coreMsg_printed and core_readcount == core_size:
                print "\n%d core words are all read." %(core_size)
                coreMsg_printed = True

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

    if recordUpperleft:
        G = [ G11, G12, G21 ]
        F = [ F11, F12, F21 ]
    else:
        G = [ G12, G21 ]
        F = [ F12, F21 ]

    return vocab_all, word2id_all, word2id_core, coreword_preIDs, G, F, u0

def loadUnigramFile(filename):
    UNI = open(filename)
    vocab_dict = {}
    wid = 1
    for line in UNI:
        line = line.strip()
        if line[0] == '#':
            continue
        fields = line.split("\t")
                             # id, freq, prob
        vocab_dict[ fields[0] ] = ( wid, int(fields[1]), np.exp(float(fields[2])) )
        wid += 1

    print "%d words loaded from unigram file %s" %(wid, filename)
    return vocab_dict

def loadExtraWordFile(filename):
    extraWords = {}
    with open(filename) as f:
        for line in f:
            w, wid = line.strip().split('\t')
            extraWords[w] = 1

    print "%d words loaded from extra word file %s" %( len(extraWords), filename)
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
    questWordIndices = [ model.word2id[x] for x in (a,a2,b) ]
    # b2 is effectively iterating through the vocab. The row is all the cosine values
    b2a2 = model.sim_row(a2)
    b2a  = model.sim_row(a)
    b2b  = model.sim_row(b)
    addsims = b2a2 - b2a + b2b

    addsims[questWordIndices] = -10000

    iadd = np.nanargmax(addsims)
    b2add  = model.vocab[iadd]

    # For debugging purposes
    ia = model.word2id[a]
    ia2 = model.word2id[a2]
    ib = model.word2id[b]
    ib2 = model.word2id[realb2]
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
def evaluate_sim(model, testsets, testsetNames, getAbsentWords=False, vocab_dict=None, cutPoint=-1 ):
    # words in absentModelID2Word and words in absentVocabWords don't overlap

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
def evaluate_ana(model, testsets, testsetNames, getAbsentWords=False, vocab_dict=None, cutPoint=-1 ):
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

# return a flag indicating whether installed mem is enough to computing a D*D dimensional Gramian matrix,
# plus installed amounts and required amounts of mem
# extraVarsRatio: the required mem of other existing variables (as a ratio of the decomposed matrix)
# e.g. in evaluate.py, only the Gramian (cosine) matrix is present, extraVarsRatio = 0
# in factorize.py, if there are 4 similar sized matrices other than the Gramian, 
# then extraVarsRatio=4 may be reasonable
def isMemEnoughGramian(D, extraVarsRatio=0):
    mem = virtual_memory()
    installedMemGB = round( mem.total * 1.0 / (1<<30) )
    # some overhead for np.array, so not divided by 1024^3
    requiredMemGB = D * D * 4.0 * ( extraVarsRatio + 1 ) / 1000000000
    
    # installed mem is enough
    if requiredMemGB <= installedMemGB:
        isEnough = 2
        
    # give a warning, will use some paging file and make the computer very slow
    elif requiredMemGB <= installedMemGB * 1.2:
        isEnough = 1
    # not enough
    else:
        isEnough = 0

    return isEnough, installedMemGB, requiredMemGB

# return a flag indicating whether installed mem is enough to computing eigendecomposition of a D*D matrix
# plus installed amounts and required amounts of mem
# extraVarsRatio: the required mem of other existing variables (as a ratio of the decomposed matrix)
# assume eigendecomposition alone takes 10 times of the mem of the decomposed matrix
# e.g. when only doing eigendecomposition of the matrix is present, extraVarsRatio = 0, allVarsRatio = 10
# if there are 4 similar sized matrices other than the decomposed matrix, then allVarsRatio = 10 + 4 = 14
# In factorize.py, when doing eigendecomposition, usually there are at least 5 other matrices of similar sizes
# so by default extraVarsRatio=5
def isMemEnoughEigen(D, extraVarsRatio=5):
    mem = virtual_memory()
    installedMemGB = round( mem.total * 1.0 / (1<<30) )
    # 15 is an empirical estimation. when D=30K, it takes around 50GB mem
    requiredMemGB = D * D * 4.0 * ( extraVarsRatio + 8 ) / 1000000000
    
    # installed mem is enough
    if requiredMemGB <= installedMemGB:
        isEnough = 2
        
    # give a warning, will use some paging file and make the computer very slow
    elif requiredMemGB <= installedMemGB * 1.2:
        isEnough = 1
    # not enough
    else:
        isEnough = 0

    return isEnough, installedMemGB, requiredMemGB

def extractSentenceWords(doc, remove_url=True, remove_punc="utf-8", min_length=1):
    if remove_punc:
        # ensure doc_u is in unicode
        if not isinstance(doc, unicode):
            encoding = remove_punc
            doc_u = doc.decode(encoding)
        else:
            doc_u = doc
        # remove unicode punctuation marks, keep ascii punctuation marks
        doc_u = doc_u.translate(unicode_punc_tbl)
        if not isinstance(doc, unicode):
            doc = doc_u.encode(encoding)
        else:
            doc = doc_u
            
    if remove_url:
        re_url = r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        doc = re.sub( re_url, "", doc )
            
    sentences = re.split( r"\s*[,;:`\"()?!{}]\s*|--+|\s*-\s+|''|\.\s|\.$|\.\.+|“|”", doc ) #"
    wc = 0
    wordsInSentences = []
    
    for sentence in sentences:
        if sentence == "":
            continue

        if not re.search( "[A-Za-z0-9]", sentence ):
            continue

        words = re.split( r"\s+\+|^\+|\+?[\-*\/&%=<>\[\]~\|\@\$]+\+?|\'\s+|\'s\s+|\'s$|\s+\'|^\'|\'$|\$|\\|\s+", sentence )

        words = filter( lambda w: w, words )

        if len(words) >= min_length:
            wordsInSentences.append(words)
            wc += len(words)

    #print "%d words extracted" %wc
    return wordsInSentences, wc
                            
def randomsample( X, n ):
    """ random.sample of the rows of X
        X may be sparse -- best csr
    """
    sampleix = random.sample( xrange( X.shape[0] ), int(n) )
    return X[sampleix]
                            
class VecModel:
    def __init__(self, V, vocab, word2id, vecNormalize=True, precompute_gramian=False):
        self.Vorig = V
        self.Vnorm = np.array( [ normF(x) for x in self.Vorig ], dtype=np.float32 )
        for i, w in enumerate(vocab):
            if self.Vnorm[i] == 0:
                print "WARN: %s norm is 0" %w
                # set to 1 to avoid "divided by 0 exception"
                self.Vnorm[i] = 1
                
        self.V = self.Vorig / self.Vnorm[:, None]
        self.word2id = word2id
        self.vecNormalize = vecNormalize
        self.vocab = vocab
        self.iterIndex = 0
        self.cosTable = None
        
        if precompute_gramian:
            isEnough, installedMemGB, requiredMemGB = isMemEnoughGramian( len(V) )
            if isEnough == 2:
                self.precomputeGramian()
            
    def __contains__(self, w):
        return w in self.word2id

    def __getitem__(self, w):
        if w not in self:
            return None
        else:
            if self.vecNormalize:
                return self.V[ self.word2id[w] ]
            else:
                return self.Vorig[ self.word2id[w] ]

    def orig(self, w):
        if w not in self:
            return None
        else:
            return self.Vorig[ self.word2id[w] ]

    def precomputeGramian(self):
        print "Precompute cosine matrix, will need %.1fGB RAM..." %( len(self.V) * len(self.V) * 4.0 / 1000000000 ),
        self.cosTable = np.dot( self.V, self.V.T )
        print "Done."

    def similarity(self, x, y):
        if x not in self or y not in self:
            return 0

        if self.vecNormalize:
            if self.cosTable is not None:
                ix = self.word2id[x]
                iy = self.word2id[y]
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
            if self.cosTable is not None:
                ix = self.word2id[x]
                return self.cosTable[ix]
            return self.V.dot(self[x])
            
        vx = self[x]
        # vector too short. the dot product similarity doesn't make sense
        if normF(vx) <= 1e-6:
            return np.zeros( len(self.vocab) )

        return self.V.dot(vx)

