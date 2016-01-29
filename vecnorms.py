#!/usr/bin/python

# this simple script is to find patterns of the norms (L1) of the learned embeddings
from utils import *
import sys
import operator
import os
import getopt
import math
import pdb

def usage():
    print "Usage: vecnorms.py [-s -1 first_block_count -2 second_block_count ] embedding_filename"

def expectation(value_probs):
    accuProb = 0
    accuExp = 0
    for v, p in value_probs:
        accuExp += v * p
        accuProb += p

    return accuExp / accuProb

def var_div(value_probs):
    expect = expectation(value_probs)
    accuVar = 0
    accuProb = 0
    for v, p in value_probs:
        accuVar += (v - expect)**2 * p
        accuProb += p
    var = accuVar / accuProb
    div = math.sqrt(var)
    return var, div

if len(sys.argv) == 1:
    usage()
    sys.exit(1)

doSort = False
first_block_count = -1
second_block_count = -1
unigramFilename = 'top1grams-wiki.txt'

try:
    opts, args = getopt.getopt(sys.argv[1:],"s1:2:")
    if len(args) != 1:
        raise getopt.GetoptError("")
    embeddingFilename = args[0]
    for opt, arg in opts:
        if opt == '-s':
            doSort = True
        if opt == '-1':
            first_block_count = int(arg)
            print 'First block: 1-%d' %first_block_count
        if opt == '-2':
            second_block_count = int(arg)
            print 'Second block: %d-%d' %(first_block_count, second_block_count)
        if opt == '-u':
            # unigram file is used to get a full list of words,
            # and also to sort the absent words by their frequencies
            unigramFilename = arg
            
except getopt.GetoptError:
     usage()
     sys.exit(2)

vocab_prob = loadUnigramFile(unigramFilename)
V, vocab, word2id, skippedWords = load_embeddings( embeddingFilename, second_block_count )
warning("\nCompute norms...")

word2norm = {}
wordnorms = []
word_probs1 = []
word_probs2 = []

for i in xrange( len(V) ):
    w = vocab[i]
    if w not in vocab_prob:
        warning( "%s not in vocab, skip\n" %w )
        continue
    
    mag = norm1( V[i] )
    word2norm[w] = mag
    prob = vocab_prob[w][2]
    wordnorms.append( [ w, mag ] )
    if i < first_block_count:
        word_probs1.append( [ mag, prob ] )
    elif i < second_block_count:
        word_probs2.append( [ mag, prob ] )

warning("Done\n")

if len(word_probs1) > 0:
    var1, div1 = var_div(word_probs1)
    expect = expectation(word_probs1)
    print "First block: %d words, exp: %.2f, var: %.2f, div: %.2f" %( len(word_probs1), expect, var1, div1 )
if len(word_probs2) > 0:
    var2, div2 = var_div(word_probs2)
    expect = expectation(word_probs2)
    print "Second block: %d words, exp: %.2f, var: %.2f, div: %.2f" %( len(word_probs2), expect, var2, div2 )


if doSort:
    warning("Done\nSorting words ascendingly by norm...")
    # sort ascendingly by the norm length
    sorted_wordnorms = sorted( wordnorms, key=operator.itemgetter(1) )
    wordnorms = sorted_wordnorms
    
embeddingFilename = os.path.basename(embeddingFilename)
embeddingFilename = os.path.splitext(embeddingFilename)[0]

normFilename = "norms_" + embeddingFilename + "-%d.txt" %( len(V) )

warning( "Save norms into %s\n" %normFilename )
NORM = open(normFilename, "w")

wc = 0
for word_norm in wordnorms:
    word, norm = word_norm
    NORM.write( "%i %s: %.2f\n" %( word2id[word], word, norm ) )
    wc += 1

warning( "%d words saved\n" %wc )    
