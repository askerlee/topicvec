#!/usr/bin/python
import getopt
import glob
import sys
import os.path
from utils import *
import numpy as np
import copy
import pdb
import sys

def loadToeflTestset(toeflTestsetFilename):
    TOEFL = open(toeflTestsetFilename)
    toeflTestset = []
    for line in TOEFL:
        line = line.strip()
        words = line.split(" | ")
        toeflTestset.append(words)

    print "%d toefl test questions are loaded" %len(toeflTestset)
    return toeflTestset
    
embeddingDir = "./embeddings/"
modelFiles = [ "25000-180000-500-BLK-8.0.vec", "sparse.vec", "singular.vec",  
               "25000-180000-500-BLK-0.0.vec", "word2vec.vec", "glove.vec" ]

toeflTestsetFilename = "./testsets/ws/EN-TOEFL-80.txt"
isHyperwordsEmbed = False
hyperwordsType = None

def usage():
    print """Usage: evaluate-toefl.py [ -H -m model_file ]
Options:
  -m:    Path to the model file, a ".vec" or a Hyperwords embedding directory (with -H).
  -H:    Hyperwords embeddings type: PPMI or SVD."""
    
try:
    opts, args = getopt.getopt(sys.argv[1:],"m:H:")
    if len(args) != 0:
        raise getopt.GetoptError("")
    for opt, arg in opts:
        if opt == '-m':
            modelFiles = [ arg ]
            embeddingDir = ""
        if opt == '-H':
            isHyperwordsEmbed = True
            hyperwordsType = arg
        if opt == '-h':
            usage()
            sys.exit(0)

except getopt.GetoptError:
   usage()
   sys.exit(2)
       
vecNormalize = True
loadwordCutPoint = 180000

if loadwordCutPoint > 0:
    print "Load top %d words" %(loadwordCutPoint)

toeflTestset = loadToeflTestset(toeflTestsetFilename)

for m,modelFile in enumerate(modelFiles):
    modelFile = embeddingDir + modelFile
    if not isHyperwordsEmbed:
        V, vocab2, word2dim, skippedWords = load_embeddings( modelFile, loadwordCutPoint )
        model = VecModel(V, vocab2, word2dim, vecNormalize=vecNormalize)
    else:
        model = load_embeddings_hyper(modelFile, hyperwordsType)
        
    questionNum = 0
    correctNum = 0
    for toeflQuestion in toeflTestset:
        questionWord = toeflQuestion[0]
        maxID = -1
        maxsim = -100
        for i,w in enumerate( toeflQuestion[1:] ):
            sim = model.similarity( questionWord, w )
            if sim > maxsim:
                maxsim = sim
                maxID = i
                
        if maxID == 0:
            correctNum += 1
        else:
            question = copy.copy(toeflQuestion)
            question[maxID+1] = '(' + question[maxID+1] + ')'
            #if m == 0:
            #    pdb.set_trace()
            print "%s: %s, %s, %s, %s" %tuple(question)
        questionNum += 1
    print "%s: %d/%d=%.1f%%" %( modelFile, correctNum, questionNum, correctNum*100.0/questionNum )
    
    
