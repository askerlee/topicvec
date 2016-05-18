import sys
import pdb
import os
import getopt
import time
from corpusLoader import *
import gensim
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix

def usage():
    print """Usage: ldaExp.py corpus_name"""

corpusName = sys.argv[1]

if corpusName == "20news":
    bow_filenames = [ "20news-train-11314.svm-bow.txt", "20news-test-7532.svm-bow.txt" ]
    topicNum = 100
else:
    bow_filenames = [ "reuters-train-5770.svm-bow.txt", "reuters-test-2255.svm-bow.txt" ]
    topicNum = 50

subcorpora = []
corpus = []
for filename in bow_filenames:
    bow_csr, labels = load_svmlight_file(filename)
    row_num, col_num = bow_csr.shape
    subcorpus = [ [] for i in xrange(row_num) ]
    row_idx, col_idx = bow_csr.nonzero()
    for i, x in enumerate(row_idx):
        y = col_idx[i]
        subcorpus[x].append( (y, bow_csr[x,y]) )
    corpus += subcorpus
    subcorpora.append( (subcorpus, labels) )
    print "%d docs loaded from '%s', containing %d words" %( row_num, filename, len(row_idx) )

print "Training LDA..."
startTime = time.time()
lda = gensim.models.ldamodel.LdaModel( corpus=corpus, num_topics=topicNum, passes=20 )
endTime = time.time()
print "Finished in %.1f seconds" %( endTime - startTime )

for i in xrange(2):
    lda_filename = bow_filenames[i].replace( "svm-bow", "svm-lda" )
    LDA = open( lda_filename, "w" )
    print "Saving topic proportions into '%s'..." %lda_filename
    
    subcorpus, labels = subcorpora[i]

    for d, bow in enumerate(subcorpus):
        label = labels[d]
        topic_props = lda.get_document_topics( bow, minimum_probability=0.001 )
        LDA.write( "%d" %label )
        for k, prop in topic_props:
            LDA.write(" %d:%.3f" %(k, prop) )
        LDA.write("\n")
    LDA.close()
    print "%d docs saved" %len(subcorpus)
