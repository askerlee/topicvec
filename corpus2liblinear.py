import numpy as np
import getopt
import sys
from utils import *
import pdb
import time
import os
import json
import copy

def usage():
    print """Usage:\n  corpus2liblinear.py -d doc_dir -o output_file -v vec_file [ -s sent_file ] label
  corpus2liblinear.py -c config_file -n alg_name -v vec_file [ -s sent_file ]
Options:
  doc_dir:      Directory of the documents to convert.
  output_file:  File to save the extracted vectors.
  Label:        Label of documents. Must be 1/+1/-1.
  config_file:  File specifying multiple directories, labels and output files.
  vec_file:     File containing embedding vectors.
  alg_name:     Name of the embedding algorithm that generates vec_file. 
                Needed if onbly partial file name is specified in config_file.
  sent_file:    File containing a list of sentiment words.
"""

def parseConfigFile(configFilename):
    CONF = open(configFilename)
    dir_configs = []
    for line in CONF:
        line = line.strip()
        dir_config = json.loads(line)
        dir_configs.append(dir_config)
    return dir_configs
    
def getFileFeatures(filename, V, word2id, sentword2id, remove_stop=False):
    DOC = open(filename)
    doc = DOC.read()
    wordsInSentences, wc = extractSentenceWords(doc, 1)
    
    countedWC = 0
    outvocWC = 0
    stopwordWC = 0
    sentWC = 0
    
    wids = []
    wid2freq = {}
    BOWFeatureNum = len(sentword2id)
    BOWFreqs = np.zeros(BOWFeatureNum)
    
    for sentence in wordsInSentences:
        for w in sentence:
            w = w.lower()
            if remove_stop and w in stopwordDict:
                stopwordWC += 1
                continue
                
            if w in word2id:
                wid = word2id[w]
                wids.append( wid )
                
                if wid not in wid2freq:
                    wid2freq[wid] = 1
                else:
                    wid2freq[wid] += 1
                countedWC += 1
            else:
                outvocWC += 1
    
            if w in sentword2id:
                id = sentword2id[w]
                BOWFreqs[id] += 1
                sentWC += 1
                
    N0 = V.shape[1]
    avgv = np.zeros(N0)
    
    # avgv is the average embedding vector. Used in Tobias Schnabel et al. (2015) as the only features
    for wid, freq in wid2freq.items():
        avgv += np.log( freq + 1 ) * V[wid]
        
    #for wid in wids:
    #    avgv += V[wid]
    
    avgv = normalizeF(avgv)
    return avgv, BOWFreqs

def processDir( outFilename, docDir, label, appendToOutput, V, word2ID, sentword2id ):
    print "Process '%s' %s" %( docDir, label )
    
    if appendToOutput:
        OUT = open(outFilename, "a")
    else:
        OUT = open(outFilename, "w")
        
    filecount = 0
    
    for filename in os.listdir(docDir):
        OUT.write(label)
        fullFilename = docDir + "/" + filename
        avgv, BOWFreqs = getFileFeatures( fullFilename, V, word2ID, sentword2id  )
        for i,x in enumerate(avgv):
            OUT.write( " %d:%.4f" %( i+1, x ) )
        # i == N0 - 1 here, dimensionality of the embedding vector
        i += 1
        if BOWFreqs.shape[0] > 0:
            for freq in BOWFreqs:
                if freq > 0:
                    OUT.write( " %d:%d" %( i+1, freq ) )
                i += 1
                
        OUT.write("\n")
        filecount += 1
        if filecount % 500 == 0:
            print "\r%d\r" %filecount,
    
    if appendToOutput:
        writeMode = "appended to" 
    else:
        writeMode = "written into"
    print "%d files processed and %s '%s'" %( filecount, writeMode, outFilename )
    
    OUT.close()
            
def main():
    vecFilename = "25000-180000-500-BLK-8.0.vec"
    algname = None
    topword_cutoff = -1
    topSentWord_cutoff = -1
    
    configFilename = ""
    label = None
    appendToOutput = False
    sentimentWordFile = None
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"d:o:v:c:n:s:1ah")
        if( len(args) == 1 ):
            if args[0] != "1" and args[0] != "+1":
                raise getopt.GetoptError( "Unknown free argument '%s'" %args[0] )
            label = "+1"
        elif( len(args) > 1 ):
            raise getopt.GetoptError( "Too many free arguments '%s'" %args )
            
        for opt, arg in opts:
            if opt == '-1':
                label = "-1"
            
            if opt == '-c':
                configFilename = arg
            if opt == '-s':
                sentimentWordFile = arg
            
            if opt == '-n':
                algname = arg
            if opt == '-d':
                docDir = arg
            if opt == '-d':
                docDir = arg
            if opt == '-o':
                outFilename = arg
            if opt == '-v':
                vecFilename = arg
            if opt == '-a':
                appendToOutput = True    
            if opt == '-h':
                usage()
                sys.exit(0)

    except getopt.GetoptError, e:
        if len(e.args) == 1:
            print "Option error: %s" %e.args[0]
        usage()
        sys.exit(2)

    sentword2id = {}
    bowSize = 0
    if sentimentWordFile:
        SENT = open(sentimentWordFile)
        id = 0
        for line in SENT:
            word, freq = line.split("\t")
            sentword2id[word] = id
            id += 1
            # if topSentWord_cutoff == -1, this equality is never satisfied, so no cut off
            if id == topSentWord_cutoff:
                break
        bowSize = len(sentword2id)
        print "%d sentiment words loaded" %(bowSize)  
    
    if configFilename:
        dir_configs = parseConfigFile(configFilename)
        for conf in dir_configs:
            if 'outFilenameTrunk' in conf:
                if not algname:
                    print "-n alg_name is needed to generate full output file name"
                    usage()
                    sys.exit(2)
                
                if sentimentWordFile:
                    conf['outFilename'] = "%s-%s-bow%d.txt" %( conf['outFilenameTrunk'], algname, bowSize )
                else:
                    conf['outFilename'] = "%s-%s.txt" %( conf['outFilenameTrunk'], algname )

    elif not label:
        print "No config file nor label is specified"
        usage()
        sys.exit(0)
    else:
        dir_config = { 'dir': docDir, 'outFilename': outFilename, 
                        'label': label, 'isAppend': appendToOutput }
        dir_configs = [ dir_config ]      
                      
    V, vocab, word2ID, skippedWords_whatever = load_embeddings( vecFilename, topword_cutoff )

    for conf in dir_configs:
        processDir( conf['outFilename'], conf['docDir'], conf['label'], 
                    conf['appendToOutput'], V, word2ID, sentword2id )
            
if __name__ == '__main__':
    main()
    