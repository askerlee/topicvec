import numpy as np
import getopt
import sys
from utils import *
import pdb
import time
import os
import json

def usage():
    print """Usage:\n  topsentwords.py -c config_file -l f1,f2... -o out_file -n count
Options:
  config_file:  Same config file used by corpus2liblinear.py, 
                which specifying multiple document directories.
  f1,f2:        Files containg lists of interesting words.
  out_file:     Output file to save top interesting words. 
                Default: 'topwords.txt'
  count:        Top k words that will be counted. Default: 1000.
"""

def parseConfigFile(configFilename):
    CONF = open(configFilename)
    dir_configs = []
    for line in CONF:
        line = line.strip()
        dir_config = json.loads(line)
        dir_configs.append(dir_config)
    return dir_configs

def getListWordCount( docPath, word2freq ):
    DOC = open(docPath)
    doc = DOC.read()
    wordsInSentences, wc = extractSentenceWords(doc, 1)
    
    interestingWc = 0
    for sentence in wordsInSentences:
        for w in sentence:
            w = w.lower()
            if w in word2freq:
                word2freq[w] += 1
                interestingWc += 1

    return wc, interestingWc
    
def processDir( docDir, word2freq ):
    print "Processing '%s'" %( docDir )
    
    filecount = 0
    totalwc = 0
    totalInterestingWc = 0
    
    for filename in os.listdir(docDir):
        docPath = docDir + "/" + filename
        wc, interestingWc = getListWordCount( docPath, word2freq )
        
        totalwc += wc
        totalInterestingWc += interestingWc
        filecount += 1
        
        if filecount % 500 == 0:
            print "\r%d\r" %filecount,
    
    print "%d files scanned, totally %d words, %d are interesting" %( filecount, totalwc, totalInterestingWc )
            
def main():
    topword_cutoff = 1000
    
    configFilename = None
    listFilenames = None
    outFilename = "topwords.txt"
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"c:l:o:n:h")
            
        for opt, arg in opts:
            if opt == '-c':
                configFilename = arg
            if opt == '-o':
                outFilename = arg
            if opt == '-n':
                topword_cutoff = int(arg)
            if opt == '-l':
                listFilenames = arg.split(",")
            if opt == '-h':
                usage()
                sys.exit(0)

    except getopt.GetoptError, e:
        if len(e.args) == 1:
            print "Option error: %s" %e.args[0]
        usage()
        sys.exit(2)
    
    if not configFilename or not listFilenames:
        usage()
        sys.exit(2)
        
    dir_configs = parseConfigFile(configFilename)
    
    word2freq = {}
    
    totalwc = 0
    for listFilename in listFilenames:
        filewc = 0
        LIST = open(listFilename)
        for line in LIST:
            if line[0] == ';':
                continue
            line = line.strip()
            if not line:
                continue
            word2freq[line] = 0
            filewc += 1
            totalwc += 1
        print "%d words loaded from '%s'" %( filewc, listFilename )
    
    print "%d words loaded from %d files" %( totalwc, len(listFilenames) )        

    for conf in dir_configs:
        processDir( conf['docDir'], word2freq )

    words = sorted( word2freq.keys(), key=lambda w: word2freq[w], reverse=True )
    topwords = words[:topword_cutoff]
    OUT = open(outFilename, "w")
    for w in topwords:
        OUT.write( "%s\t%d\n" %( w, word2freq[w] ) )
    print "%d words written into '%s'" %( len(topwords), outFilename )
                    
if __name__ == '__main__':
    main()
    