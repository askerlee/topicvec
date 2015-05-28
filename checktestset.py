from gensim.models import word2vec
import getopt
import glob
import sys
import os.path
from utils import *

def loadBigramFile2(bigram_filename, topWordNum, extraWords):
    print "Loading bigram file '%s':" %bigram_filename
    BIGRAM = open(bigram_filename)
    lineno = 0
    vocab = []
    word2dim = {}
    
    try:
        header = BIGRAM.readline()
        lineno += 1
        match = re.match( "# (\d+) words, \d+ occurrences", header )
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
    
        match = re.match( "# (\d+) bigram occurrences", header)
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
        
        # Read the focus word list, build the word2dim mapping
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
 
    except ValueError, e:
        if len( e.args ) == 2:
            print "Unknown line %d:\n%s" %( e.args[0], e.args[1] )
        else:
            exc_type, exc_obj, tb = sys.exc_info()
            print "Source line %d: %s" %(tb.tb_lineno, e)
        exit(0)
    
    print
    BIGRAM.close()
    
    return vocab, word2dim
    
    
# default is current directory
testsetDir = "D:/Dropbox/doc2vec/omer2/testsets/ws/"
# if set to [], run all testsets
testsetNames = [ "bruni_men", "ws353_similarity", "ws353_relatedness", "radinsky_mturk" ] #, "luong_rare" ]
unigramFile = "top1grams-wiki.txt"
bigramFile = "top2grams-wiki.txt"
extraWordFile = "absentwords.txt"
getAbsentWords = True

if testsetDir[-1] != '/':
    testsetDir += '/'

if not os.path.isdir(testsetDir):
    print "ERR: Test set dir does not exist or is not a dir:\n" + testsetDir
    sys.exit(2)

extraWords = {}
if extraWordFile:
    with open(extraWordFile) as f:
        for line in f:
            words = line.strip().split(',')
            for w in words:
                extraWords[w] = 1
    
#vocab = loadUnigramFile(unigramFile)
vocab, word2dim = loadBigramFile2(bigramFile, 13000, extraWords)
vocab_size = len(vocab)
testsets = []

# We don't care about the test performance here. So use random vectors
V = np.random.randn( 10, vocab_size )
model = vecModel( V, word2dim )

vocab_dict = {}
for w in vocab:
    vocab_dict[w] = (word2dim[w], 0, 0)
    
if len(testsetNames) == 0:
    testsetNames = glob.glob( testsetDir + '*.txt' )
    if len(testsetNames) == 0:
        print "No testset ended with '.txt' is found in " + testsetDir
        sys.exit(2)
    testsetNames = map( lambda x: os.path.basename(x)[:-4], testsetNames )
        
for testsetName in testsetNames:
    testset = loadTestset( testsetDir + testsetName + ".txt" )
    testsets.append(testset)

spearmanCoeff, absentModelWordsID, absentModelWordsW, absentVocabWords = evaluate( model, testsets, testsetNames, vocab_dict, getAbsentWords )

if getAbsentWords:
    print "Absent words ID from the model:"
    print "ID:"
    print ",".join( map( lambda i: str(i), absentModelWordsID) )
    print "Words:"
    print ",".join(absentModelWordsW)
    
    if len(absentVocabWords) > 0:
        print "Absent words from the vocab:"
        print "\n".join( absentVocabWords )
    
    print

