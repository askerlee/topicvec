import numpy as np
import scipy.linalg
from scipy.special import *
import getopt
import sys
from utils import *
import pdb
import time

embed_algs = { "PSDVec": "d:/corpus/embeddings/25000-180000-500-BLK-8.0.vec", 
                "word2vec": "d:/corpus/embeddings/word2vec2.vec",
                "CCA": "d:/corpus/embeddings/182800-500-CCA.vec"
             } 
                    # "50000-180000-500-BLK-8.0.vec" }
testsetDir = "./concept categorization"
testsetNames = [ "ap", "battig", "esslli" ]
maxID = -1

for algname, vecFilename in embed_algs.iteritems():
    print "Alg %s" %algname
    if vecFilename[-4:] == ".bin":
        V, vocab, word2ID, skippedWords_whatever = load_embeddings_bin(vecFilename, 400000)
    else:
        V, vocab, word2ID, skippedWords_whatever = load_embeddings(vecFilename, 400000)
    
    for testsetName in testsetNames:
        truthFilename = testsetDir + "/" + testsetName + ".txt"
        vecFilename = testsetDir + "/" + testsetName + "-" + algname + ".vec"
        labelFilename = testsetDir + "/" + testsetName + "-" + algname + ".label"
        
        FVEC = open(vecFilename, "w")
        ids = []
        
        FLABEL = open(labelFilename, "w")
        
        with open(truthFilename) as FT:
            # skip header
            FT.readline()
            for line in FT:
                line = line.strip()
                fields = line.split("\t")
                word, cat = fields[:2]
                    
                if word not in word2ID:
                    print "%s not in vocab" %word
                    continue
                else:
                    id = word2ID[word]
                    #print "%s: %d" %(word, id)
                    if id > maxID:
                        maxID = id
                    ids.append(id)
                    
                    FLABEL.write("%s\n" %cat)
            
            FVEC.write( "%d %d\n" %( len(ids), V.shape[1] ) )
            for id in ids:            
                v = V[id]
                FVEC.write("%.3f" %v[0])
                for d in v[1:]:
                    FVEC.write(" %.3f" %d)
                FVEC.write("\n")
        
            FLABEL.close()
            FVEC.close()
