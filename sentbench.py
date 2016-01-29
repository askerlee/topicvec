import os
import re
import subprocess

alg2vec = { "PSD-reg": "25000-180000-500-BLK-8.0.vec", 
            #"PSD-unreg": "25000-180000-500-BLK-0.0.vec", 
            #"word2vec": "word2vec2.vec",
            #"CCA": "182800-500-CCA.vec",
            "sparse":  "120000-sparse.vec"
          }

vecDir = "d:/corpus/embeddings"
liblinearDir = "D:/liblinear-2.1/windows"
trainExePath = liblinearDir + "/" + "train.exe"
predictExePath = liblinearDir + "/" + "predict.exe"
dataDir = "d:/corpus"
trainFiletrunk = "sent-train"
testFiletrunk = "sent-test"
dataGenScript = dataDir + "/corpus2liblinear.py"
dataGenConfig = dataDir + "/sent-gen.conf"
sentimentWordFile = dataDir + "/topSentWords500.txt"

# code below is just to count the words in sentimentWordFile.
# the count is used in file names
sentword2id = {}
bowSize = 0
if sentimentWordFile:
    SENT = open(sentimentWordFile)
    id = 0
    for line in SENT:
        word, freq = line.split("\t")
        sentword2id[word] = id
        id += 1
    bowSize = len(sentword2id)
    SENT.close()
     
# L1 or L2 regularization for the logistic regression solver
# Experiments show this option has little impact on the results
solverReg = 2
if solverReg == 1:
    solverType = "-s6"
elif solverReg == 2:
    solverType = "-s7"
    
for algName, vecFilename in alg2vec.items():
    print "%s:" %algName
    
    vecFullfilename = vecDir + "/" + vecFilename
    
    if sentimentWordFile:
        trainFilename = "%s/%s-%s-bow%d.txt" %( dataDir, trainFiletrunk, algName, bowSize )
        testFilename = "%s/%s-%s-bow%d.txt" %( dataDir, testFiletrunk, algName, bowSize )
    else:
        trainFilename = "%s/%s-%s.txt" %( dataDir, trainFiletrunk, algName )
        testFilename = "%s/%s-%s.txt" %( dataDir, testFiletrunk, algName )
        
    if not ( os.path.isfile(trainFilename) and os.path.isfile(testFilename) ):
        options = [ "python", dataGenScript, "-c", dataGenConfig, "-n", algName, \
                              "-v", vecFullfilename ]
        if sentimentWordFile:
            options.append("-s")
            options.append(sentimentWordFile)
            
        subprocess.call(options)

    if sentimentWordFile:
        modelFilename = "%s-bow%d.model" %( algName, bowSize )
        outputFilename = "%s-bow%d.output" %( algName, bowSize )
    else:
        modelFilename = "%s.model" %algName
        outputFilename = "%s.output" %algName
    
    print "Training using %s" %trainFilename
    subprocess.call( [ trainExePath, solverType, "-v10", trainFilename, modelFilename ] )
    print "Testing using %s" %testFilename
    subprocess.call( [ predictExePath, testFilename, modelFilename, outputFilename ] )
    print
    